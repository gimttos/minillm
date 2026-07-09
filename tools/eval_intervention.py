"""개입 인과성 — activation patching ("거미 -> 개미") — D1.

내부 상태가 출력과 단지 상관인지, 아니면 **인과적으로** 지배하는지 가른다
(J-space 핵심 실험의 자가 재현). 개념 A와 B의 평균 은닉 차
    Δ = mean_h(A) - mean_h(B)
를 구해 프롬프트의 레이어 L 은닉에 α·Δ를 더하고, 목표 토큰의 로짓이
α에 따라 단조·인과적으로 변하는지 본다. 무작위 방향(대조군)보다 효과가
커야 "그 방향이 진짜 개념 축"이라는 증거가 된다.

전역작업공간(GWT)의 방송·점화 지표에 대응.

사용법:
    python -m tools.eval_intervention --ckpt checkpoints/sft.pt \
        --concept-a 거미 --concept-b 개미 --targets "8,6" --layer 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tokenizer.bpe import BPETokenizer
from tools._common import load_model


def layer_hidden(model, ids, layer):
    """레이어 L 블록 출력의 마지막 위치 은닉 (Δ 계산용)."""
    captured = {}

    def hook(_m, _in, out):
        captured["h"] = out[:, -1, :].detach()

    hd = model.blocks[layer].register_forward_hook(hook)
    with torch.no_grad():
        model.hidden_states(ids)
    hd.remove()
    return captured["h"]


def target_logits(model, ids, target_ids, layer=None, vec=None):
    """프롬프트 끝 다음토큰 로짓 중 target_ids만. vec가 주어지면 레이어 L
    블록 출력에 vec를 더한 채(개입) 계산한다."""
    hd = None
    if vec is not None:
        def hook(_m, _in, out):
            return out + vec.view(1, 1, -1)
        hd = model.blocks[layer].register_forward_hook(hook)
    with torch.no_grad():
        h = model.hidden_states(ids)
        logits = model.lm_head(h[:, -1, :]).squeeze(0)
    if hd is not None:
        hd.remove()
    return logits[target_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--concept-a", default="거미")
    ap.add_argument("--concept-b", default="개미")
    ap.add_argument("--prompt", default="그것의 다리 개수는")
    ap.add_argument("--targets", default="8,6", help="관찰할 목표 토큰(쉼표)")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--alphas", default="-4,-2,0,2,4")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", nargs="?", const="eval_out/intervention.json", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = load_model(args.ckpt, args.device)
    L = min(args.layer, cfg.n_layers - 1)
    tok = BPETokenizer.load(args.tokenizer)

    def enc(s):
        return torch.tensor([tok.encode(s)], device=args.device)

    # 개념 축 Δ: A/B 표현을 여러 문맥에 넣어 평균 은닉 차를 구한다
    ctx = ["", "나는 ", "저기 ", "무서운 "]
    ha = torch.stack([layer_hidden(model, enc(c + args.concept_a), L) for c in ctx]).mean(0)
    hb = torch.stack([layer_hidden(model, enc(c + args.concept_b), L) for c in ctx]).mean(0)
    delta = (ha - hb).squeeze(0)
    rand = torch.randn_like(delta)
    rand = rand / rand.norm() * delta.norm()   # 같은 크기의 무작위 방향(대조군)

    target_ids = [tok.encode(t)[0] for t in args.targets.split(",")]
    prompt = enc(args.prompt)
    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"레이어 {L}에 Δ={args.concept_a}-{args.concept_b} 주입 | 목표 {args.targets}")
    print(f"{'alpha':>6} | {'Δ 개입 (목표 로짓)':>28} | {'무작위 개입':>22}")
    rows = []
    for a in alphas:
        lg = target_logits(model, prompt, target_ids, L, a * delta).tolist()
        rg = target_logits(model, prompt, target_ids, L, a * rand).tolist()
        rows.append({"alpha": a, "delta": lg, "random": rg})
        f = lambda v: "[" + ", ".join(f"{x:+.2f}" for x in v) + "]"
        print(f"{a:>6.1f} | {f(lg):>28} | {f(rg):>22}")

    # 인과성 요약: 목표0 - 목표1 로짓차가 α에 따라 단조로 변하면 인과적
    diffs = [r["delta"][0] - r["delta"][1] for r in rows]
    rand_diffs = [r["random"][0] - r["random"][1] for r in rows]
    swing = max(diffs) - min(diffs)
    rand_swing = max(rand_diffs) - min(rand_diffs)
    print(f"\nΔ 개입 로짓차 변화폭 {swing:.2f} vs 무작위 {rand_swing:.2f} "
          f"-> {'개념 방향이 유의' if swing > rand_swing * 1.5 else '차이 미미'}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"layer": L, "targets": args.targets, "rows": rows,
                   "swing": swing, "random_swing": rand_swing},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
