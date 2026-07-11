"""대화 파인튜닝(SFT) — 사전학습된 모델을 "질문에 답하는 모델"로 바꾼다.

pretrain.py와 거의 같지만 세 가지가 다르다:
  1. 사전학습 체크포인트에서 가중치를 이어받아 시작한다.
  2. 데이터가 (연속 토큰 스트림)이 아니라 (대화 예시들)이다. 예시를
     max_seq_len에 맞춰 자르거나 패딩해 배치로 만든다.
  3. loss를 답변 토큰에만 준다 (loss_mask). 질문을 외우게 하지 않는다.

파인튜닝이라 스텝 수도 학습률도 작다 (기존 지식을 망가뜨리지 않도록).

마음 유사 기제의 학습도 여기서 한다:
  --mood-dim D : 기분 벡터. 배치의 절반을 2-pass로 학습 — 먼저 문맥의
                 은닉 평균을 기분 관측으로 압축하고, 그 기분을 주입한 채
                 답변을 학습한다. "문맥의 압축된 느낌에 조건화하는 법"을
                 배우는 것. 나머지 절반은 mood 없이 학습해 첫 턴(기분이
                 아직 없는 상태)에도 강건하게 만든다.
  --latent K   : Coconut 잠재 사고. 답변 첫 토큰 전에 은닉 상태를 말 없이
                 K번 되먹이는 경로를 커리큘럼(초반 30% -> 후반 70% 확률)으로
                 학습한다. K=0 배치를 항상 섞어 잠재 스텝 없는 추론도
                 유효하게 유지한다.
검증 분할(마지막 --val-frac)을 떼어 주기적으로 val loss를 보고,
가장 좋았던 시점의 가중치를 저장한다.

사용법:
    python -m train.sft --init checkpoints/ckpt_best.pt --data data/bin/sft.npz
    python -m train.sft --init ... --mood-dim 64                 # 기분 벡터
    python -m train.sft --init ... --latent 2                    # 잠재 사고
    python -m train.sft --init ... --n-pause 4                   # pause 데이터로 SFT 시 기록
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.gpt import GPT, ModelConfig
from train.config import pick_amp_dtype
from train.pretrain import _save


def make_batch(picks, boundaries, ids, mask, block, pad_id, device):
    """예시 인덱스 picks로 (x, y, m) 배치를 만든다.
    각 예시는 block+1 길이로 패딩/절단한다. 패딩 위치의 target은 -1(무시)."""
    batch_size = len(picks)
    x = np.full((batch_size, block), pad_id, dtype=np.int64)
    y = np.full((batch_size, block), -1, dtype=np.int64)
    m = np.zeros((batch_size, block), dtype=np.float32)
    for row, p in enumerate(picks):
        s, e = boundaries[p], boundaries[p + 1]
        seg_ids = ids[s:e].astype(np.int64)
        seg_mask = mask[s:e].astype(np.float32)
        n = min(len(seg_ids) - 1, block)  # 마지막 토큰은 정답으로만 쓰임
        x[row, :n] = seg_ids[:n]
        y[row, :n] = seg_ids[1:n + 1]
        m[row, :n] = seg_mask[1:n + 1]   # 위치 t의 정답(t+1)이 답변 토큰일 때만 1
    to = lambda a: torch.from_numpy(a).to(device)
    return to(x), to(y), to(m)


# ---------------------------------------------------------------------------
# 기분 벡터: 2-pass 학습
# ---------------------------------------------------------------------------
def mood_from_context(model, x, y, m, h=None):
    """1-pass: 문맥(사용자 질문 부분)의 은닉 평균을 기분 관측으로 압축한다.
    은닉 계산은 no_grad(비쌈 + 여기로 역전파할 필요 없음)지만, 읽기 헤드
    mood_read에는 그래디언트가 흐르게 해 "무엇을 기분으로 읽을지"를 배운다.
    h를 주면(역피드백과 1-pass 공유) 다시 계산하지 않는다."""
    if h is None:
        with torch.no_grad():
            h = model.hidden_states(x)                 # (B, T, C)
    sel = ((y != -1) & (m == 0)).float().unsqueeze(-1)  # 실제 토큰 중 답변이 아닌 곳
    pool = (h * sel).sum(1) / sel.sum(1).clamp(min=1)   # (B, C)
    return torch.tanh(model.mood_read(pool.detach()))   # (B, mood_dim)


def ws_from_context(model, x, y, m, h=None):
    """워크스페이스 2-pass 학습(§C1): 문맥의 은닉 평균을 슬롯 상태로 압축한다.
    mood와 대칭 — 세션엔 멀티턴 상태가 없으니, 문맥을 슬롯으로 압축해 조건화하는
    법(ws_write)과 방송으로 읽는 법(ws_read)을 함께 배운다."""
    if h is None:
        with torch.no_grad():
            h = model.hidden_states(x)
    sel = ((y != -1) & (m == 0)).float().unsqueeze(-1)
    pool = (h * sel).sum(1) / sel.sum(1).clamp(min=1)   # (B, C)
    return torch.tanh(model.ws_write(pool.detach()))    # (B, slots*dim)


# ---------------------------------------------------------------------------
# 잠재 사고(Coconut): 배치 구성과 loss
# ---------------------------------------------------------------------------
def make_latent_batch(picks, boundaries, ids, mask, a_id, pad_id, device,
                      block=0, k=0):
    """예시들을 <|assistant|> 직후에서 갈라 (접두부, 답변부) 배치로 만든다.

    접두부는 왼쪽 패딩(모든 행의 <|assistant|>가 마지막 열에 오도록),
    답변부는 오른쪽 패딩. 이렇게 정렬해야 잠재 스텝과 답변부가 배치 전체에서
    같은 열에서 시작해 캐시/마스크를 한 번에 처리할 수 있다.

    왼쪽 패딩이 RoPE를 깨지 않는 이유: RoPE는 상대 거리만 보는데, 한 행 안의
    실제 토큰들은 여전히 연속된 열에 있어 상대 거리가 전부 보존된다.
    패딩 열은 attn_mask로 어텐션에서 제외한다.

    block(=model max_seq_len)을 주면 전역 위치 예산 P + k + S <= block 을
    강제한다. RoPE 테이블이 block 행까지만 있는데 run_from_pos 의 슬라이스는
    범위를 넘어도 조용히 짧아지기만 해서, 넘는 순간 apply_rope 의 곱에서
    shape 불일치로 터진다. P 는 배치 내 최장 접두부, S 는 최장 답변부라
    서로 다른 예시에서 오므로, 예시 하나하나가 block 이하여도 합은 넘을 수
    있다 — 그래서 예시 단위가 아니라 배치 단위로 잘라야 한다.
    """
    # 접두부 예산: 잠재 k스텝 + 답변 최소 1토큰 자리를 남긴다
    max_pre = (block - k - 1) if block else 10 ** 9

    rows = []
    for p in picks:
        s, e = boundaries[p], boundaries[p + 1]
        seg_ids = ids[s:e].astype(np.int64)
        seg_mask = mask[s:e].astype(np.float32)
        a_pos = np.where(seg_ids == a_id)[0]
        if len(a_pos) == 0:
            continue  # 형식이 깨진 예시는 건너뜀
        # 멀티턴이면 assistant가 여러 번 나온다 — 마지막 답변 앞에서 가른다.
        # "대화 전체를 읽고 → 잠재 사고 → 마지막 답변" 구조가 latent의 의미와
        # 맞고, 앞선 턴은 접두부(문맥)로 들어간다. 단일턴이면 [0]==[-1].
        split = int(a_pos[-1]) + 1                    # 접두부는 <|assistant|> 포함
        if split > max_pre:
            # 접두부가 예산을 넘으면 왼쪽(가장 오래된 문맥)을 버린다 —
            # 질문 뒷부분과 <|assistant|>는 보존되므로 학습 신호는 유지된다
            cut = split - max_pre
            seg_ids, seg_mask, split = seg_ids[cut:], seg_mask[cut:], max_pre
        rows.append((seg_ids, seg_mask, split))

    if not rows:
        return None
    B = len(rows)
    P = max(r[2] for r in rows)                       # 접두부 길이 (왼쪽 패딩 후)
    S = max(len(r[0]) - r[2] - 1 for r in rows)       # 답변부 입력 길이
    if block:
        S = min(S, block - k - P)                     # 남은 위치 예산만큼만 (>=1 보장)
    prefix = np.full((B, P), pad_id, dtype=np.int64)
    pad_len = np.zeros(B, dtype=np.int64)
    suffix_x = np.full((B, S), pad_id, dtype=np.int64)
    suffix_y = np.full((B, S), -1, dtype=np.int64)
    suffix_m = np.zeros((B, S), dtype=np.float32)
    first_y = np.zeros(B, dtype=np.int64)             # 마지막 잠재 스텝의 정답 = 답변 첫 토큰

    for i, (seg_ids, seg_mask, split) in enumerate(rows):
        pl = P - split
        pad_len[i] = pl
        prefix[i, pl:] = seg_ids[:split]
        first_y[i] = seg_ids[split]
        sx = seg_ids[split:-1]                        # 각 위치 t가 t+1을 맞힌다
        n = min(len(sx), S)                           # 예산 초과분(답변 꼬리)은 잘라냄
        suffix_x[i, :n] = sx[:n]
        suffix_y[i, :n] = seg_ids[split + 1:split + 1 + n]
        suffix_m[i, :n] = seg_mask[split + 1:split + 1 + n]

    to = lambda a: torch.from_numpy(a).to(device)
    return to(prefix), to(pad_len), to(suffix_x), to(suffix_y), to(suffix_m), to(first_y)


def latent_loss(model, prefix, pad_len, suffix_x, suffix_y, suffix_m, first_y, k):
    """잠재 사고 경로: 접두부(캐시 유지) -> 잠재 k스텝 -> 답변부(직사각 마스크).

    추론(generate)과 똑같은 순서로 처리하되, 접두부와 답변부는 병렬로
    처리해 학습 속도를 지킨다. loss = 답변 첫 토큰(마지막 잠재 스텝이 맞힘)
    + 답변부 각 토큰. attn_mask는 True=참조 허용.
    """
    B, P = prefix.shape
    S = suffix_x.size(1)
    dev = prefix.device
    cols = torch.arange(P, device=dev)
    real = cols[None, :] >= pad_len[:, None]          # (B, P) 패딩이 아닌 열

    # 1) 접두부: causal + 패딩 차단, KV 캐시에 쌓는다 (그래디언트 유지)
    pre_mask = (cols[None, None, None, :] <= cols[None, None, :, None]) \
        & real[:, None, None, :]
    # 패딩 열의 쿼리는 참조할 곳이 하나도 없으면 softmax가 NaN이 되어 배치
    # 전체로 번진다 — 최소한 자기 자신은 보게 한다 (출력은 어차피 버려짐)
    diag = torch.eye(P, dtype=torch.bool, device=dev)
    pre_mask = pre_mask | diag[None, None, :, :]
    caches = model.new_caches()
    h = model.run_from_pos(model.tok_emb(prefix), 0, caches, attn_mask=pre_mask)
    h_last = h[:, -1, :]                              # <|assistant|> 위치의 은닉

    # 2) 잠재 스텝: 은닉을 말 없이 자기 입력으로 k번 되먹인다
    for t in range(k):
        lat_mask = torch.ones(B, 1, 1, P + t + 1, dtype=torch.bool, device=dev)
        lat_mask[:, 0, 0, :P] = real
        x_lat = model.latent_proj(h_last).unsqueeze(1)
        h_last = model.run_from_pos(x_lat, P + t, caches, attn_mask=lat_mask)[:, -1, :]

    # 3) 답변 첫 토큰: 마지막 잠재 스텝의 은닉이 맞혀야 한다 (항상 학습 대상)
    loss_first = F.cross_entropy(model.lm_head(h_last), first_y, reduction="sum")

    # 4) 답변부: 직사각 causal — 전역 위치 P+k+i는 j <= P+k+i까지 참조 가능
    cols_all = torch.arange(P + k + S, device=dev)
    rows_q = torch.arange(S, device=dev)
    suf_mask = (cols_all[None, None, None, :] <= (P + k + rows_q)[None, None, :, None]) \
        .expand(B, 1, S, P + k + S).clone()
    suf_mask[:, 0, :, :P] &= real[:, None, :]
    h_suf = model.run_from_pos(model.tok_emb(suffix_x), P + k, caches, attn_mask=suf_mask)
    logits = model.lm_head(h_suf)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), suffix_y.reshape(-1),
                         ignore_index=-1, reduction="none")
    msum = suffix_m.reshape(-1)
    return (loss_first + (ce * msum).sum()) / (B + msum.sum()).clamp(min=1)


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_val(model, val_idx, boundaries, ids, mask, block, batch_size, pad_id,
                 device, use_mood=False, use_ws=False):
    """검증 예시 전체의 masked loss 평균. use_mood/use_ws면 2-pass(주입)로 잰다.
    역피드백은 켜져 있으면(model.cfg.feedback) 항상 반영 — 추론과 같은 조건."""
    model.eval()
    losses = []
    for i in range(0, len(val_idx), batch_size):
        picks = val_idx[i:i + batch_size]
        x, y, m = make_batch(picks, boundaries, ids, mask, block, pad_id, device)
        h = (model.hidden_states(x)
             if (use_mood or use_ws or model.cfg.feedback) else None)
        mood = mood_from_context(model, x, y, m, h=h) if use_mood else None
        ws = ws_from_context(model, x, y, m, h=h) if use_ws else None
        fh = h if model.cfg.feedback else None
        _, loss = model(x, y, loss_mask=m, mood=mood, feedback_h=fh, ws=ws)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def estimate_ece(model, val_idx, boundaries, ids, mask, block, batch_size, pad_id,
                 device, bins=10):
    """검증셋 답변 토큰에서 ECE(Expected Calibration Error)를 계산한다 (§C3).

    확신도 헤드를 1급 지표로 승격 — 정식 리포트는 tools/eval_conf.py지만,
    학습 중 주기적으로 찍어 캘리브레이션 추이를 본다. 낮을수록 잘 보정됨.
    """
    model.eval()
    confs, corrs = [], []
    for i in range(0, len(val_idx), batch_size):
        picks = val_idx[i:i + batch_size]
        x, y, m = make_batch(picks, boundaries, ids, mask, block, pad_id, device)
        h = model.hidden_states(x)
        if model.cfg.feedback:
            h = model.hidden_states(x, feedback_h=h)   # 추론과 같은 2-pass
        logits = model.lm_head(h)
        conf = torch.sigmoid(model.conf_head(h).squeeze(-1))
        correct = (logits.argmax(-1) == y).float()
        sel = ((y != -1).float() * m).bool()
        confs.append(conf[sel])
        corrs.append(correct[sel])
    model.train()
    conf = torch.cat(confs).cpu().numpy()
    correct = torch.cat(corrs).cpu().numpy()
    if len(conf) == 0:
        return float("nan")
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        s = (conf >= lo) & (conf < hi if b < bins - 1 else conf <= hi)
        if s.sum() == 0:
            continue
        ece += s.mean() * abs(conf[s].mean() - correct[s].mean())
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="사전학습 체크포인트")
    ap.add_argument("--data", default="data/bin/sft.npz")
    ap.add_argument("--out", default="checkpoints/sft.pt")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto",
                    help="AMP dtype (auto=장치 능력으로 T4 fp16 / Ampere+ bf16 자동)")
    # --- 마음 유사 기제 ---
    ap.add_argument("--mood-dim", type=int, default=0, help="기분 벡터 차원 (0=off)")
    ap.add_argument("--latent", type=int, default=0, help="잠재 사고 스텝 수 (0=off)")
    ap.add_argument("--n-pause", type=int, default=0,
                    help="데이터에 넣은 pause 수 — 체크포인트에 기록해 chat.py가 따라 하게 한다")
    ap.add_argument("--feedback", action="store_true",
                    help="역피드백: 직전 토큰의 최종 은닉을 다음 토큰 입력에 방송 (2-pass)")
    ap.add_argument("--conf", action="store_true",
                    help="확신도 헤드: '다음 토큰을 맞힐 것인가'를 스스로 예측 (메타인지)")
    ap.add_argument("--workspace-slots", type=int, default=0,
                    help="워크스페이스 슬롯 수 (0=off) — GWT 지속 작업공간")
    ap.add_argument("--workspace-dim", type=int, default=0,
                    help="슬롯 하나의 차원 (0=d_model)")
    ap.add_argument("--attn-schema", action="store_true",
                    help="주의 도식 헤드: 레이어별 어텐션 엔트로피를 은닉에서 예측 (AST)")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json",
                    help="--latent 사용 시 <|assistant|> 위치를 찾는 데 필요")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--eval-interval", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    ck = torch.load(args.init, map_location=args.device)
    cfg = ModelConfig(**ck["model_config"])
    # 기능 플래그를 켜서 모델을 만든다 — 새 파라미터(FiLM 등)는 항등 초기화라
    # 켜는 순간에는 동작이 변하지 않고, 학습으로만 활성화된다
    if args.mood_dim:
        cfg.mood_dim = args.mood_dim
    if args.latent:
        cfg.n_latent = args.latent
    if args.n_pause:
        cfg.n_pause = args.n_pause
    if args.feedback:
        cfg.feedback = True
    if args.conf:
        cfg.conf_head = True
    if args.workspace_slots:
        cfg.workspace_slots = args.workspace_slots
        cfg.workspace_dim = args.workspace_dim
    if args.attn_schema:
        cfg.attn_schema = True
    model = GPT(cfg).to(args.device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"strict=False 로드: missing={missing}, unexpected={unexpected}")
    print(f"사전학습 가중치 로드: {args.init} ({model.num_params() / 1e6:.1f}M)")

    # SFT에서는 loop 반복 횟수를 최대값으로 고정한다 — 잠재 사고 경로가
    # 여러 forward에 걸쳐 같은 캐시를 쓰므로 실행 구조가 일정해야 한다
    model._loop_override = cfg.n_loop

    d = np.load(args.data)
    ids, mask, boundaries = d["ids"], d["mask"], d["boundaries"]
    n_examples = len(boundaries) - 1
    n_val = max(int(n_examples * args.val_frac), 1)
    train_idx = np.arange(0, n_examples - n_val)
    val_idx = np.arange(n_examples - n_val, n_examples)
    steps = int(len(train_idx) * args.epochs / args.batch_size)
    warmup = max(steps // 20, 10)
    print(f"{len(train_idx):,}개 학습 / {n_val:,}개 검증 예시, {steps:,} 스텝 예정")

    a_id = None
    if args.latent:
        from tokenizer.bpe import BPETokenizer
        a_id = BPETokenizer.load(args.tokenizer).encode_special("<|assistant|>")

    pad_id = int(ids[0])  # 아무 토큰이나 무방 — target=-1이라 loss에 안 잡힘
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    dtype = pick_amp_dtype(args.device) if args.dtype == "auto" else args.dtype
    if args.device == "cuda":
        print(f"AMP dtype: {dtype}")
    use_amp = args.device == "cuda" and dtype in ("float16", "bfloat16")
    amp_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and dtype == "float16"))

    best_val = float("inf")
    saved_best = False
    model.train()
    for step in range(steps):
        # 코사인 LR (워밍업 포함)
        if step < warmup:
            lr = args.lr * (step + 1) / warmup
        else:
            r = (step - warmup) / max(steps - warmup, 1)
            lr = 0.1 * args.lr + 0.5 * (1 + math.cos(math.pi * r)) * 0.9 * args.lr
        for g in opt.param_groups:
            g["lr"] = lr

        picks = np.random.choice(train_idx, size=args.batch_size, replace=False)

        # --- 학습 경로 선택 ---
        progress = step / max(steps, 1)
        p_latent = 0.3 if progress < 1 / 3 else 0.7  # 잠재 커리큘럼: 점점 자주
        path = "plain"
        if args.latent and random.random() < p_latent:
            path = "latent"
        elif cfg.mood_dim and random.random() < 0.5:
            path = "mood"
        elif cfg.workspace_slots and random.random() < 0.5:
            path = "ws"

        def compute_loss():
            if path == "latent":
                # 역피드백은 이 경로에 얹지 않는다 — 잠재 스텝의 입력 자체가
                # 이미 전대역 피드백이라 중복이다
                batch = make_latent_batch(picks, boundaries, ids, mask,
                                          a_id, pad_id, args.device,
                                          block=cfg.max_seq_len, k=args.latent)
                if batch is not None:
                    return latent_loss(model, *batch, k=args.latent)
            x, y, m = make_batch(picks, boundaries, ids, mask,
                                 cfg.max_seq_len, pad_id, args.device)
            # 1-pass 은닉은 기분/워크스페이스/역피드백이 공유한다 — 같이 켜도 비용 동일
            h = None
            if cfg.feedback or path in ("mood", "ws"):
                with torch.no_grad():
                    h = model.hidden_states(x)
            mood = mood_from_context(model, x, y, m, h=h) if path == "mood" else None
            ws = ws_from_context(model, x, y, m, h=h) if path == "ws" else None
            fh = h if cfg.feedback else None
            _, loss = model(x, y, loss_mask=m, mood=mood, feedback_h=fh, ws=ws)
            return loss

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = compute_loss()
        else:
            loss = compute_loss()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            extra = ""
            if cfg.conf_head and getattr(model, "_last_conf_loss", None) is not None:
                extra += f" | conf {model._last_conf_loss.item():.3f}"
            if cfg.attn_schema and getattr(model, "_last_schema_loss", None) is not None:
                extra += f" | schema {model._last_schema_loss.item():.3f}"
            print(f"step {step:>5}/{steps} | loss {loss.item():.3f} | lr {lr:.2e} | {path}{extra}")

        if step > 0 and step % args.eval_interval == 0:
            vloss = estimate_val(model, val_idx, boundaries, ids, mask,
                                 cfg.max_seq_len, args.batch_size, pad_id, args.device)
            line = f"  >> val loss {vloss:.3f} (best {best_val:.3f})"
            if cfg.mood_dim:
                vm = estimate_val(model, val_idx, boundaries, ids, mask,
                                  cfg.max_seq_len, args.batch_size, pad_id,
                                  args.device, use_mood=True)
                line += f" | mood 주입 시 {vm:.3f}"
            if cfg.workspace_slots:
                vw = estimate_val(model, val_idx, boundaries, ids, mask,
                                  cfg.max_seq_len, args.batch_size, pad_id,
                                  args.device, use_ws=True)
                line += f" | ws 주입 시 {vw:.3f}"
            if cfg.conf_head:
                ece = estimate_ece(model, val_idx, boundaries, ids, mask,
                                   cfg.max_seq_len, args.batch_size, pad_id, args.device)
                line += f" | ECE {ece:.3f}"
            print(line)
            if vloss < best_val:
                best_val = vloss
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                _save(args.out, model, opt, step, best_val,
                      type("C", (), {"model": cfg})())  # _save는 cfg.model만 참조
                saved_best = True

    if not saved_best:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _save(args.out, model, opt, steps - 1, best_val,
              type("C", (), {"model": cfg})())
    print(f"SFT 완료 -> {args.out} (best val {best_val:.3f})")


if __name__ == "__main__":
    main()
