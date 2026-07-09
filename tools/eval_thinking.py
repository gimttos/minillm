"""잠재(latent) vs 속말(pause) — 같은 연산량 k 비교 — D3.

이 프로젝트의 대표 실험: 연속 잠재 사고가 이산 필러 속말보다 나은가.
출력으로 환원되지 않는 내부 연산이 실제로 이득인지, 같은 base에서 가른
`--latent k` 모델과 `--n-pause k` 모델을 **같은 검증셋·시드**로 비교한다.
어느 쪽이 이기든 수치로 결론내는 것이 목적.

지표: 답변 토큰 masked val loss + 답변 top-1 정확도.

사용법:
    python -m tools.eval_thinking --data data/bin/sft.npz \
        --ckpts k0=checkpoints/sft.pt latent=checkpoints/sft_latent2.pt \
                pause=checkpoints/sft_pause4.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train.sft import make_batch
from tools._common import load_model, load_sft_val


@torch.no_grad()
def score(model, cfg, val_idx, boundaries, ids, mask, pad_id, batch_size, device):
    """답변 토큰 masked loss와 top-1 정확도. 추론과 같은 조건(feedback 2-pass)."""
    tot_loss, tot_tok, tot_correct = 0.0, 0, 0
    import torch.nn.functional as F
    for i in range(0, len(val_idx), batch_size):
        picks = val_idx[i:i + batch_size]
        x, y, m = make_batch(picks, boundaries, ids, mask, cfg.max_seq_len,
                             pad_id, device)
        h = model.hidden_states(x)
        if cfg.feedback:
            h = model.hidden_states(x, feedback_h=h)
        logits = model.lm_head(h)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             ignore_index=-1, reduction="none")
        mf = m.reshape(-1)
        tot_loss += float((ce * mf).sum())
        tot_tok += float(mf.sum())
        correct = (logits.argmax(-1) == y).float().reshape(-1)
        tot_correct += float((correct * mf).sum())
    return tot_loss / max(tot_tok, 1), tot_correct / max(tot_tok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bin/sft.npz")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="이름=경로 쌍들 (예: k0=... latent=... pause=...)")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", nargs="?", const="eval_out/thinking.json", default="")
    args = ap.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)
    val_idx, boundaries, ids, mask, pad_id = load_sft_val(args.data, args.val_frac)

    print(f"검증 예시 {len(val_idx):,}개 (같은 셋·시드로 공정 비교)")
    print(f"{'이름':>10} | {'k(설정)':>10} | {'val loss':>9} | {'답변 top-1':>9}")
    results = {}
    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        model, cfg = load_model(path, args.device)
        k = f"lat{cfg.n_latent}/pau{cfg.n_pause}"
        loss, acc = score(model, cfg, val_idx, boundaries, ids, mask, pad_id,
                          args.batch_size, args.device)
        results[name] = {"ckpt": path, "n_latent": cfg.n_latent,
                         "n_pause": cfg.n_pause, "val_loss": loss, "acc": acc}
        print(f"{name:>10} | {k:>10} | {loss:>9.4f} | {acc:>9.3f}")

    best = min(results, key=lambda n: results[n]["val_loss"])
    print(f"\n최저 val loss: {best} ({results[best]['val_loss']:.4f})")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
