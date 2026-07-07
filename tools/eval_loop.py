"""loop 반복 횟수(n_loop)별 val loss·속도 비교.

확률적 반복 횟수로 학습한 모델은 하나의 가중치로 여러 반복 횟수에서
동작한다. 이 스크립트는 "한 번 더 생각하면 실제로 더 잘 맞히는가"를
숫자로 보여 준다. (학습 최대치보다 큰 값은 외삽 — 보통은 나빠지지만
직접 확인해 볼 가치가 있다.)

사용법:
    python -m tools.eval_loop --ckpt checkpoints/ckpt_best.pt --n-loops 1,2,3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model.gpt import GPT, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/bin/val.bin")
    ap.add_argument("--n-loops", default="1,2,3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)

    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = ModelConfig(**ck["model_config"])
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"모델: {args.ckpt} (학습 설정 loop {cfg.loop_start}..{cfg.loop_end} x{cfg.n_loop})")

    data = np.memmap(Path(args.data), dtype=np.uint16, mode="r")
    block = cfg.max_seq_len

    # 모든 n_loop이 같은 배치를 보도록 미리 뽑아 둔다 (공정한 비교)
    batches = []
    for _ in range(args.iters):
        ix = np.random.randint(0, len(data) - block - 1, size=args.batch_size)
        x = np.stack([data[i:i + block].astype(np.int64) for i in ix])
        y = np.stack([data[i + 1:i + 1 + block].astype(np.int64) for i in ix])
        batches.append((torch.from_numpy(x).to(args.device),
                        torch.from_numpy(y).to(args.device)))

    for n in (int(s) for s in args.n_loops.split(",")):
        model._loop_override = n
        losses = []
        t0 = time.time()
        with torch.no_grad():
            for x, y in batches:
                _, loss = model(x, y)
                losses.append(loss.item())
        dt = (time.time() - t0) / len(batches)
        print(f"n_loop={n} | val loss {sum(losses) / len(losses):.4f} "
              f"| {dt * 1000:.0f} ms/batch")


if __name__ == "__main__":
    main()
