"""주의 도식 정확도 (AST 검증) — D6.

C2의 attn_schema 헤드가 만든 "자기 어텐션 상태 모델"이 실제 어텐션을
예측하는지 검증한다. 검증셋에서 수동 계산한 실제 레이어별 어텐션 엔트로피
vs 도식 헤드 예측의 상관·오차를 재고, 무작위/영 예측 baseline과 비교한다.

주의도식(AST) 지표에 대응.

사용법:
    python -m tools.eval_schema --ckpt checkpoints/sft_schema.pt --data data/bin/sft.npz
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from train.sft import make_batch
from tools._common import load_model, load_sft_val, require


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/bin/sft.npz")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", nargs="?", const="eval_out/schema.json", default="")
    args = ap.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)
    model, cfg = load_model(args.ckpt, args.device)
    require(cfg.attn_schema, "attn_schema 헤드가 없는 체크포인트입니다 (--attn-schema로 SFT 필요)")
    val_idx, boundaries, ids, mask, pad_id = load_sft_val(args.data, args.val_frac)

    preds, tgts = [], []
    with torch.no_grad():
        for i in range(0, len(val_idx), args.batch_size):
            picks = val_idx[i:i + args.batch_size]
            x, y, m = make_batch(picks, boundaries, ids, mask, cfg.max_seq_len,
                                 pad_id, args.device)
            T = x.size(1)
            tgt = model.attention_entropy(x) / math.log(max(T, 2))   # (B,T,K)
            pred = model.attn_schema_head(model.hidden_states(x))     # (B,T,K)
            sel = (y != -1).unsqueeze(-1).expand_as(tgt)              # 실제 위치만
            preds.append(pred[sel].cpu())
            tgts.append(tgt[sel].cpu())

    pred = torch.cat(preds).numpy()
    tgt = torch.cat(tgts).numpy()
    mae = float(np.abs(pred - tgt).mean())
    # 상관 (전체 원소 평면화)
    corr = float(np.corrcoef(pred, tgt)[0, 1]) if pred.std() > 0 else 0.0
    # baseline: 타깃 평균으로만 예측 (도식이 평균 이상을 하는가)
    base_mae = float(np.abs(tgt - tgt.mean()).mean())

    print(f"검증 원소 {len(pred):,}개 (위치×레이어)")
    print(f"도식 예측 MAE : {mae:.4f}   (baseline=평균예측 {base_mae:.4f})")
    print(f"상관계수      : {corr:.3f}")
    print("-> " + ("도식이 실제 어텐션을 유의하게 예측" if mae < base_mae and corr > 0.1
                   else "baseline 대비 이득 미미"))

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"mae": mae, "baseline_mae": base_mae, "corr": corr,
                   "n": len(pred)}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
