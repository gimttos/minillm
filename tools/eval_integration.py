"""통합/재귀 프로브 (RPT/IIT 근사) — D4.

loop·feedback이 만드는 정보 통합의 **방향성 신호**를 잰다(진짜 Φ 아님, 명시).
한 위치의 은닉을 섭동하고, 그 변화가 뒤 위치들로 얼마나 멀리 전파되는지
측정한다. 순전파-only는 국소적이고, 재귀(loop/feedback)는 더 전역적이어야
한다는 가설을 숫자로 확인한다.

재귀처리(RPT) 지표에 대응.

사용법:
    python -m tools.eval_integration --ckpt checkpoints/sft_loop_fb.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tokenizer.bpe import BPETokenizer
from tools._common import load_model


def final_hidden(model, ids, feedback=False, perturb=None):
    """final_norm 은닉 (1,T,C). perturb=(layer,pos,vec)면 그 위치를 섭동."""
    hd = None
    if perturb is not None:
        layer, pos, vec = perturb
        def hook(_m, _in, out):
            out = out.clone()
            out[:, pos, :] = out[:, pos, :] + vec
            return out
        hd = model.blocks[layer].register_forward_hook(hook)
    with torch.no_grad():
        h = model.hidden_states(ids)
        if feedback:
            h = model.hidden_states(ids, feedback_h=h)
    if hd is not None:
        hd.remove()
    return h


def propagation(model, ids, layer, pos, feedback, seed=0):
    """position pos 섭동이 뒤 위치로 전파되는 정도. 반환 (평균 전파거리, 도달)."""
    torch.manual_seed(seed)
    C = model.cfg.d_model
    vec = torch.randn(C, device=ids.device)
    vec = vec / vec.norm() * 3.0
    h0 = final_hidden(model, ids, feedback)
    h1 = final_hidden(model, ids, feedback, perturb=(layer, pos, vec))
    change = (h1 - h0).norm(dim=-1).squeeze(0)   # (T,)
    T = change.numel()
    downstream = torch.arange(T, device=ids.device) - pos
    m = (downstream > 0).float() * change
    denom = m.sum().clamp(min=1e-9)
    mean_dist = float((m * downstream.float()).sum() / denom)
    reach = int((change[pos + 1:] > change.max() * 0.05).sum()) if pos + 1 < T else 0
    return mean_dist, reach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="옛날 옛날 아주 먼 옛날에 작은 마을이 하나 있었어요.")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", nargs="?", const="eval_out/integration.json", default="")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    tok = BPETokenizer.load(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt)], device=args.device)
    L = min(args.layer, cfg.n_layers - 1)
    pos = ids.size(1) // 3          # 앞쪽 위치를 섭동해 뒤로 전파를 관찰

    print(f"프롬프트 길이 {ids.size(1)} | 레이어 {L} 위치 {pos} 섭동")
    print(f"{'조건':>16} | {'평균 전파거리':>12} | {'도달 위치수':>10}")

    conditions = []
    # loop off vs on (학습 최대치가 >1일 때만 의미)
    if cfg.n_loop > 1:
        for n in (1, cfg.n_loop):
            model._loop_override = n
            conditions.append((f"loop x{n}", False))
    else:
        model._loop_override = cfg.n_loop
        conditions.append(("loop off", False))
    # feedback off vs on (켜진 체크포인트일 때)
    if cfg.feedback:
        model._loop_override = cfg.n_loop
        conditions += [("feedback off", False), ("feedback on", True)]

    results = {}
    for name, fb in conditions:
        if name.startswith("loop"):
            model._loop_override = int(name.split("x")[-1]) if "x" in name else cfg.n_loop
        dist, reach = propagation(model, ids, L, pos, fb)
        results[name] = {"mean_distance": dist, "reach": reach}
        print(f"{name:>16} | {dist:>12.3f} | {reach:>10}")

    print("\n(재귀가 클수록 전파 거리·도달이 커야 국소->전역 통합 신호)")
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
