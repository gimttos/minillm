"""상태 지속 일관성 (mood / workspace) — D5.

"왜 상태를 갈아끼우면 유지가 안 됐나"를 수치화한다. 좋은 지속 상태라면:
  1. 민감성 : 상태를 다르게 주면 출력이 유의하게 갈린다 (같은 입력이라도).
  2. 안정성 : 그래도 발산하지 않는다 (상태를 갱신해도 노름이 폭주 안 함).
  3. 복귀   : 중립 대화를 흘리면 상태가 기준선으로 감쇠·복귀한다.
  4. 재현성 : 같은 상태 -> 같은(greedy) 출력.

예측처리(느린 prior) 지표에 대응. mood나 workspace 중 켜진 것을 잰다.

사용법:
    python -m tools.eval_state --ckpt checkpoints/sft_mood.pt
    python -m tools.eval_state --ckpt checkpoints/sft_ws.pt --save
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tokenizer.bpe import BPETokenizer
from tools._common import load_model, require

PROMPTS = [
    "오늘 날씨가 참 좋네요.",
    "요즘 제일 힘든 게 뭐예요?",
    "좋아하는 노래 하나 추천해 줄래?",
    "주말에 뭐 하고 지냈어?",
]


def next_dist(model, cfg, ids, mood=None, ws=None):
    """프롬프트 끝에서의 다음 토큰 분포 (greedy 판단·KL용)."""
    with torch.no_grad():
        h = model.hidden_states(ids, mood=mood, ws=ws)
        return torch.softmax(model.lm_head(h[:, -1, :]), dim=-1).squeeze(0)


def kl(p, q, eps=1e-9):
    return float((p * ((p + eps).log() - (q + eps).log())).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--n-states", type=int, default=5, help="비교할 무작위 상태 수")
    ap.add_argument("--turns", type=int, default=8, help="복귀 관찰용 중립 갱신 횟수")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", nargs="?", const="eval_out/state.json", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = load_model(args.ckpt, args.device)
    require(cfg.mood_dim > 0 or cfg.workspace_slots > 0,
            "mood도 workspace도 없는 체크포인트입니다 (--mood-dim / --workspace-slots로 SFT 필요)")
    tok = BPETokenizer.load(args.tokenizer)

    kind = "mood" if cfg.mood_dim > 0 else "workspace"
    dim = cfg.mood_dim if kind == "mood" else cfg.workspace_slots * (cfg.workspace_dim or cfg.d_model)
    print(f"상태 종류: {kind} (dim={dim})")

    prompts = [torch.tensor([tok.encode(p)], device=args.device) for p in PROMPTS]

    # --- 1) 민감성: 상태별 다음토큰 분포가 baseline(0) 대비 얼마나 갈리나 ---
    def dist(ids, state):
        if kind == "mood":
            return next_dist(model, cfg, ids, mood=state)
        return next_dist(model, cfg, ids, ws=state)

    zero = torch.zeros(1, dim, device=args.device)
    states = [torch.randn(1, dim, device=args.device) * 0.5 for _ in range(args.n_states)]

    kls, rand_kls = [], []
    for ids in prompts:
        base = dist(ids, zero)
        for s in states:
            kls.append(kl(dist(ids, s), base))
        # 대조군: 상태는 0인데 두 번 재면 KL≈0 이어야 (재현성)
        rand_kls.append(kl(dist(ids, zero), base))
    sens = float(np.mean(kls))
    repro = float(np.mean(rand_kls))
    print(f"민감성  : 상태 주입 시 평균 KL(vs 0) = {sens:.4f}  (클수록 상태가 출력을 가름)")
    print(f"재현성  : 같은 상태(0) 반복 KL       = {repro:.2e}  (0에 가까워야 함)")

    # --- 2) 안정성·복귀: 무작위 상태에서 중립 갱신을 반복하며 노름 궤적 ---
    # 갱신은 update_mood/update_workspace를 쓴다 — 실제 대화의 감쇠와 동일.
    model._turn_hidden_mean = model.hidden_states(prompts[0]).mean(1)  # 중립 관측 고정
    traj = []
    s = states[0].clone()
    for _ in range(args.turns):
        s = model.update_mood(s) if kind == "mood" else model.update_workspace(s)
        traj.append(float(s.norm()))
    diverged = traj[-1] > traj[0] * 1.5
    print(f"안정성  : 노름 궤적 {[round(t,3) for t in traj]}")
    print(f"복귀    : {'발산 경고!' if diverged else 'OK (폭주 없음)'}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"kind": kind, "dim": dim, "sensitivity_kl": sens,
                   "reproducibility_kl": repro, "norm_trajectory": traj,
                   "diverged": diverged}, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
