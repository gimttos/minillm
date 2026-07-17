"""사전학습 루프 — 모델에게 "다음 토큰 맞히기"를 시키는 곳.

학습이란 결국 이 반복이다:
    1. 데이터에서 (입력, 정답) 배치를 뽑는다  (정답 = 입력을 한 칸 민 것)
    2. 모델이 예측하고 loss(틀린 정도)를 계산한다  (forward)
    3. loss를 줄이는 방향으로 모든 가중치를 아주 조금 민다  (backward + step)
수십만 번 반복하면 모델은 한국어의 통계적 규칙을 스스로 익힌다.

체크포인트 재개(resume)를 지원한다 — 무료 클라우드는 세션이 끊기므로
필수다. --resume 을 주면 마지막 체크포인트에서 이어서 학습한다.

사용법:
    python -m train.pretrain --preset tiny                    # 로컬 검증
    python -m train.pretrain --preset full                    # 클라우드(base)
    python -m train.pretrain --preset full --resume           # 이어서
    python -m train.pretrain --preset full --optimizer muon   # Muon 하이브리드
    python -m train.pretrain --preset full --target-tokens 500000000  # 예산 스윕
"""

import argparse
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from model.gpt import GPT
from train.config import get_config, pick_amp_dtype
from train.muon import build_optimizer_bundle


def get_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    """긴 토큰 배열에서 무작위 위치 batch_size개를 뽑아 (x, y)를 만든다.
    y는 x를 한 칸 민 것 — 위치 t의 정답은 t+1의 실제 토큰이다."""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size].astype(np.int64) for i in ix])
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":
        # non_blocking 전송으로 GPU가 노는 시간을 줄인다
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


class Prefetcher:
    """백그라운드 스레드로 다음 배치를 미리 만들어 큐에 채운다 (§A5).

    이 모델은 작고 빨라 동기식 배치 준비(파이썬 루프 + np.stack + pin + 전송)가
    GPU를 놀린다. 배치 생성을 학습 스텝과 겹쳐 GPU 활용률을 올린다.
    CUDA 스트림까지는 불필요(과설계 금지) — 큐(maxsize=2) 하나면 충분.
    """

    def __init__(self, data, block, batch, device, seed=0):
        self.data, self.block, self.batch, self.device = data, block, batch, device
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.exc = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _make(self):
        ix = self.rng.integers(0, len(self.data) - self.block - 1, size=self.batch)
        x = np.stack([self.data[i:i + self.block].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block].astype(np.int64) for i in ix])
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.device == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x, y

    def _worker(self):
        try:
            while not self._stop.is_set():
                batch = self._make()
                while not self._stop.is_set():
                    try:
                        self.q.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as e:              # 스레드 예외를 메인으로 전파
            self.exc = e
            self.q.put(None)

    def get(self):
        item = self.q.get()
        if item is None:
            raise self.exc or RuntimeError("prefetch 스레드가 죽었습니다")
        x, y = item
        if self.device == "cuda":
            return (x.to(self.device, non_blocking=True),
                    y.to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)

    def close(self):
        self._stop.set()


def lr_at(step: int, cfg) -> float:
    """워밍업(선형 증가) 후 코사인 감소. 초반 폭주를 막고 후반에 미세조정."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, data, cfg):
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.model.max_seq_len, cfg.batch_size, cfg.device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full",
                    choices=["tiny", "tiny-loop", "full", "large", "xl", "xxl",
                             "full-loop"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default="", help="이 체크포인트 가중치로 시작(재개 아님)")
    ap.add_argument("--target-tokens", type=int, default=0,
                    help="토큰 예산 오버라이드 (0=프리셋 값). max_steps를 유도")
    ap.add_argument("--optimizer", default="", choices=["", "adamw", "muon"],
                    help="옵티마이저 오버라이드 (기본=프리셋)")
    ap.add_argument("--dtype", default="", help="AMP dtype 오버라이드 (기본=auto)")
    args = ap.parse_args()

    cfg = get_config(args.preset)
    if args.target_tokens:
        cfg.target_tokens = args.target_tokens
    if args.optimizer:
        cfg.optimizer = args.optimizer
    if args.dtype:
        cfg.dtype = args.dtype

    # --- 토큰 예산 → max_steps 유도 (§A1) ---
    cfg.max_steps = cfg.resolve_max_steps()
    if cfg.target_tokens:
        print(f"토큰 예산 {cfg.target_tokens/1e9:.2f}B tokens -> {cfg.max_steps} steps "
              f"(유효배치 {cfg.batch_size*cfg.grad_accum}×{cfg.model.max_seq_len})")

    # --- AMP dtype 자동 선택 (§A2) ---
    if cfg.dtype == "auto":
        cfg.dtype = pick_amp_dtype(cfg.device)
    if cfg.device == "cuda":
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        print(f"AMP dtype: {cfg.dtype} (cap {cap[0]}.{cap[1]})")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.memmap(Path(cfg.data_dir) / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(Path(cfg.data_dir) / "val.bin", dtype=np.uint16, mode="r")

    model = GPT(cfg.model).to(cfg.device)
    print(f"파라미터 수: {model.num_params() / 1e6:.1f}M | 옵티마이저: {cfg.optimizer}")

    optimizer = build_optimizer_bundle(model, cfg)

    start_step = 0
    best_val = float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=cfg.device)
        # 프리셋을 바꿔 놓고 --resume 을 주면 다른 아키텍처의 가중치를 얹으려다
        # 수십 줄짜리 shape 에러가 쏟아진다. 원인을 한 줄로 말해 준다.
        old = ck.get("model_config", {})
        new = vars(cfg.model)
        diff = {k: (old.get(k), new[k]) for k in ("d_model", "n_layers", "n_heads",
                                                  "ffn_hidden", "vocab_size", "max_seq_len")
                if k in old and old[k] != new[k]}
        if diff:
            raise SystemExit(
                f"재개 실패: {ckpt_path} 는 다른 아키텍처입니다 — "
                + ", ".join(f"{k} {o}->{n}" for k, (o, n) in diff.items())
                + f"\n  이 체크포인트는 다른 프리셋의 것입니다. 새 프리셋으로 처음부터"
                  f" 학습하려면 --resume 없이 실행하세요 (out_dir={out_dir}).")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])   # 종류 불일치 시 내부에서 새로 시작
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"재개: step {start_step}부터 (지금까지 best_val={best_val:.3f})")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=cfg.device)
        # strict=False: 기분 벡터(FiLM) 등 새 기능의 파라미터가 체크포인트에
        # 없어도 로드된다 (새 파라미터는 항등 초기화 상태로 시작)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"  strict=False 로드: missing={missing}, unexpected={unexpected}")
        print(f"가중치 초기화: {args.init_from}")

    # mixed precision: GPU에서 fp16/bf16으로 계산해 속도·메모리 이득
    use_amp = cfg.device == "cuda" and cfg.dtype in ("float16", "bfloat16")
    amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and cfg.dtype == "float16"))

    if cfg.compile:
        t_c = time.time()
        model = torch.compile(model)
        print(f"torch.compile 준비(첫 스텝에서 컴파일; 여기선 래핑만 {time.time()-t_c:.1f}s)")

    prefetch = Prefetcher(train_data, cfg.model.max_seq_len, cfg.batch_size,
                          cfg.device, seed=cfg.seed)

    model.train()
    t0 = time.time()
    try:
        for step in range(start_step, cfg.max_steps):
            # Muon과 AdamW의 서로 다른 base_lr에 공통 코사인 계수를 곱한다
            scale = lr_at(step, cfg) / cfg.learning_rate
            optimizer.set_lr_scale(scale)

            # --- grad accumulation: 작은 배치 여러 번의 그래디언트를 모아 큰 배치 흉내 ---
            for micro in range(cfg.grad_accum):
                x, y = prefetch.get()
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)
                loss = loss / cfg.grad_accum
                scaler.scale(loss).backward()

            for opt in optimizer.optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for opt in optimizer.optimizers:
                scaler.step(opt)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_interval == 0:
                dt = time.time() - t0
                t0 = time.time()
                per = dt / cfg.log_interval if step > start_step else dt
                lrs = " ".join(f"{k} {v:.2e}" for k, v in optimizer.current_lrs().items())
                print(f"step {step:>6} | loss {loss.item() * cfg.grad_accum:.3f} "
                      f"| {lrs} | {per * 1000:.0f} ms/step")

            if step > 0 and step % cfg.eval_interval == 0:
                vloss = estimate_loss(model, val_data, cfg)
                print(f"  >> val loss {vloss:.3f} (best {best_val:.3f})")
                if vloss < best_val:
                    best_val = vloss
                    _save(out_dir / "ckpt_best.pt", model, optimizer, step, best_val, cfg)

            if step > 0 and step % cfg.save_interval == 0:
                _save(ckpt_path, model, optimizer, step, best_val, cfg)
    finally:
        prefetch.close()

    _save(ckpt_path, model, optimizer, cfg.max_steps - 1, best_val, cfg)
    print("학습 완료.")


def _save(path, model, optimizer, step, best_val, cfg):
    # torch.compile은 원본을 _orig_mod에 감싼다 — 저장은 원본 state_dict로
    raw = getattr(model, "_orig_mod", model)
    torch.save({
        "model": raw.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_config": vars(cfg.model),
    }, path)


if __name__ == "__main__":
    main()
