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
    python -m train.pretrain --preset full                    # 클라우드
    python -m train.pretrain --preset full --resume           # 이어서
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from model.gpt import GPT
from train.config import get_config


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
    ap.add_argument("--preset", default="full", choices=["tiny", "full"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default="", help="이 체크포인트 가중치로 시작(재개 아님)")
    args = ap.parse_args()

    cfg = get_config(args.preset)
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
    print(f"파라미터 수: {model.num_params() / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
    )

    start_step = 0
    best_val = float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", best_val)
        print(f"재개: step {start_step}부터 (지금까지 best_val={best_val:.3f})")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=cfg.device)
        model.load_state_dict(ck["model"])
        print(f"가중치 초기화: {args.init_from}")

    # mixed precision: GPU에서 fp16/bf16으로 계산해 속도·메모리 이득
    use_amp = cfg.device == "cuda" and cfg.dtype in ("float16", "bfloat16")
    amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(use_amp and cfg.dtype == "float16"))

    if cfg.compile:
        model = torch.compile(model)

    model.train()
    t0 = time.time()
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # --- grad accumulation: 작은 배치 여러 번의 그래디언트를 모아 큰 배치 흉내 ---
        for micro in range(cfg.grad_accum):
            x, y = get_batch(train_data, cfg.model.max_seq_len, cfg.batch_size, cfg.device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            per = dt / cfg.log_interval if step > start_step else dt
            print(f"step {step:>6} | loss {loss.item() * cfg.grad_accum:.3f} "
                  f"| lr {lr:.2e} | {per * 1000:.0f} ms/step")

        if step > 0 and step % cfg.eval_interval == 0:
            vloss = estimate_loss(model, val_data, cfg)
            print(f"  >> val loss {vloss:.3f} (best {best_val:.3f})")
            if vloss < best_val:
                best_val = vloss
                _save(out_dir / "ckpt_best.pt", model, optimizer, step, best_val, cfg)

        if step > 0 and step % cfg.save_interval == 0:
            _save(ckpt_path, model, optimizer, step, best_val, cfg)

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
