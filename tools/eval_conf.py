"""확신도 헤드의 calibration 평가 — "자기 예측이 실제로 맞는가?"

SFT 검증 예시에서 답변 토큰마다 (모델의 확신도, 실제로 맞혔는가)를 모아
확신도 구간별 실제 정답률 표(reliability table)를 찍는다.
확신도 0.9 구간에서 실제로 ~90%를 맞힌다면, 모델은 자기 상태를 읽고
자기 수행을 예측하는 기능적 메타인지를 갖춘 것이다.

사용법:
    python -m tools.eval_conf --ckpt checkpoints/sft_conf.pt --data data/bin/sft.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.gpt import GPT, ModelConfig
from train.sft import make_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/bin/sft.npz")
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="sft.py와 같은 값이어야 학습에 안 쓴 예시로 평가된다")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", nargs="?", const="eval_out/reliability.json", default="",
                    help="신뢰도 다이어그램을 JSON으로 저장 (기본 eval_out/reliability.json)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device)
    cfg = ModelConfig(**ck["model_config"])
    assert cfg.conf_head, "확신도 헤드가 없는 체크포인트입니다 (--conf로 SFT 필요)"
    model = GPT(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    model._loop_override = cfg.n_loop

    d = np.load(args.data)
    ids, mask, boundaries = d["ids"], d["mask"], d["boundaries"]
    n_examples = len(boundaries) - 1
    n_val = max(int(n_examples * args.val_frac), 1)
    val_idx = np.arange(n_examples - n_val, n_examples)
    pad_id = int(ids[0])

    confs, corrects = [], []
    with torch.no_grad():
        for i in range(0, len(val_idx), args.batch_size):
            picks = val_idx[i:i + args.batch_size]
            x, y, m = make_batch(picks, boundaries, ids, mask,
                                 cfg.max_seq_len, pad_id, args.device)
            h = model.hidden_states(x)
            fh = h if cfg.feedback else None
            if fh is not None:
                h = model.hidden_states(x, feedback_h=fh)  # 추론과 같은 2-pass
            logits = model.lm_head(h)
            conf = torch.sigmoid(model.conf_head(h).squeeze(-1))
            correct = (logits.argmax(-1) == y).float()
            valid = (y != -1).float() * m                  # 답변 토큰만
            sel = valid.bool()
            confs.append(conf[sel].cpu())
            corrects.append(correct[sel].cpu())

    conf = torch.cat(confs).numpy()
    correct = torch.cat(corrects).numpy()
    print(f"답변 토큰 {len(conf):,}개 | 평균 확신도 {conf.mean():.3f} "
          f"| 실제 정답률 {correct.mean():.3f}")
    print(f"{'확신도 구간':>14} | {'개수':>7} | {'실제 정답률':>10}")
    ece = 0.0  # Expected Calibration Error: |확신 - 실제|의 가중 평균
    diagram = []  # 신뢰도 다이어그램 데이터 (--save)
    for b in range(args.bins):
        lo, hi = b / args.bins, (b + 1) / args.bins
        sel = (conf >= lo) & (conf < hi if b < args.bins - 1 else conf <= hi)
        if sel.sum() == 0:
            continue
        acc = correct[sel].mean()
        ece += sel.mean() * abs(conf[sel].mean() - acc)
        print(f"  [{lo:.1f}, {hi:.1f}) | {int(sel.sum()):>7,} | {acc:>10.3f}")
        diagram.append({"lo": lo, "hi": hi, "count": int(sel.sum()),
                        "mean_conf": float(conf[sel].mean()), "accuracy": float(acc)})
    print(f"ECE (낮을수록 잘 보정됨): {ece:.4f}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"n": len(conf), "mean_conf": float(conf.mean()),
                       "accuracy": float(correct.mean()), "ece": float(ece),
                       "bins": diagram}, f, ensure_ascii=False, indent=2)
        print(f"신뢰도 다이어그램 저장: {out}")


if __name__ == "__main__":
    main()
