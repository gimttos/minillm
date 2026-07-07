"""대화 파인튜닝(SFT) — 사전학습된 모델을 "질문에 답하는 모델"로 바꾼다.

pretrain.py와 거의 같지만 세 가지가 다르다:
  1. 사전학습 체크포인트에서 가중치를 이어받아 시작한다.
  2. 데이터가 (연속 토큰 스트림)이 아니라 (대화 예시들)이다. 예시를
     max_seq_len에 맞춰 자르거나 패딩해 배치로 만든다.
  3. loss를 답변 토큰에만 준다 (loss_mask). 질문을 외우게 하지 않는다.

파인튜닝이라 스텝 수도 학습률도 작다 (기존 지식을 망가뜨리지 않도록).

사용법:
    python -m train.sft --init checkpoints/ckpt_best.pt --data data/bin/sft.npz
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from model.gpt import GPT, ModelConfig
from train.pretrain import _save


def make_batch(examples, boundaries, ids, mask, block, batch_size, pad_id, device):
    """무작위 예시 batch_size개를 골라 (x, y, m) 배치를 만든다.
    각 예시는 block+1 길이로 패딩/절단한다. 패딩 위치의 target은 -1(무시)."""
    x = np.full((batch_size, block), pad_id, dtype=np.int64)
    y = np.full((batch_size, block), -1, dtype=np.int64)
    m = np.zeros((batch_size, block), dtype=np.float32)
    picks = np.random.randint(0, len(boundaries) - 1, size=batch_size)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="사전학습 체크포인트")
    ap.add_argument("--data", default="data/bin/sft.npz")
    ap.add_argument("--out", default="checkpoints/sft.pt")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)

    ck = torch.load(args.init, map_location=args.device)
    cfg = ModelConfig(**ck["model_config"])
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    print(f"사전학습 가중치 로드: {args.init} ({model.num_params() / 1e6:.1f}M)")

    d = np.load(args.data)
    ids, mask, boundaries = d["ids"], d["mask"], d["boundaries"]
    n_examples = len(boundaries) - 1
    steps = int(n_examples * args.epochs / args.batch_size)
    warmup = max(steps // 20, 10)
    print(f"{n_examples:,}개 예시, {steps:,} 스텝 예정")

    pad_id = int(ids[0])  # 아무 토큰이나 무방 — target=-1이라 loss에 안 잡힘
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    use_amp = args.device == "cuda"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and args.dtype == "float16"))

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

        x, y, m = make_batch(n_examples, boundaries, ids, mask,
                             cfg.max_seq_len, args.batch_size, pad_id, args.device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, loss = model(x, y, loss_mask=m)
        else:
            _, loss = model(x, y, loss_mask=m)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(f"step {step:>5}/{steps} | loss {loss.item():.3f} | lr {lr:.2e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _save(args.out, model, opt, steps - 1, 0.0,
          type("C", (), {"model": cfg})())  # _save는 cfg.model만 참조
    print(f"SFT 완료 -> {args.out}")


if __name__ == "__main__":
    main()
