"""Muon 옵티마이저 (Newton–Schulz 직교화) + AdamW 하이브리드.

왜 Muon인가
===========
SGD/Adam은 그래디언트를 좌표별로만 스케일한다. Muon은 2D 가중치 행렬의
업데이트를 **직교화**(대략 특이값을 전부 1로) 해서 모든 방향으로 고르게
민다 — 스피드런에서 같은 목표 loss까지 스텝 수가 ~1.3~2배 줄어드는 표준
기법이다. 직교화는 SVD 대신 Newton–Schulz 반복 5스텝으로 근사한다.

무엇에 쓰고 무엇에 안 쓰나 (§A3)
================================
- Muon 대상 = **블록 내부의 2D 가중치 행렬** (attn/ffn). 여기서만 이득.
- AdamW 대상 = 나머지 전부: 임베딩(=tied lm_head), RMSNorm, bias, 그리고
  마음 기제 헤드(mood/latent/feedback/conf/workspace/attn_schema...).
  헤드는 항등/제로 init 동역학을 지켜야 하므로 직교화에 넣지 않는다.

modded-nanoGPT(KellerJordan)의 구현을 참조하되 **단일 GPU/비분산**으로
단순화했다(분산 all-gather 제거).
"""

from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton–Schulz 5차 반복으로 G의 (근사) 직교화 행렬을 구한다.

    G = U S V^T 일 때 U V^T (특이값을 1로) 에 수렴한다. 계수 (a,b,c)는
    modded-nanoGPT의 값으로, 특이값이 [0,1]에 몰려도 빠르게 1로 밀어올린다.
    GPU에서는 bf16으로 계산해 싸게(정밀도는 이 근사에 충분), CPU에서는 fp32.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(work_dtype)
    transposed = G.size(0) > G.size(1)
    if transposed:                    # 세로로 긴 행렬은 눕혀 계산 (반복이 더 안정)
        X = X.T
    X = X / (X.norm() + 1e-7)          # 스펙트럼을 [0,1]로 정규화
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """2D 가중치 행렬 전용 Muon. momentum 후 업데이트를 직교화해 적용한다."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            wd, nesterov, ns = group["weight_decay"], group["nesterov"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                u = zeropower_via_newtonschulz5(g, steps=ns)
                # RMS 스케일 보정: 직교화 후 업데이트의 원소 크기를 행렬 모양에
                # 맞춰 보정해 AdamW와 비슷한 유효 스텝 크기를 갖게 한다.
                u = u * (max(1.0, g.size(0) / g.size(1)) ** 0.5)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# 하이브리드 번들: Muon(블록 2D) + AdamW(나머지)를 하나처럼 다룬다
# ---------------------------------------------------------------------------
_MUON_SUFFIXES = (
    "attn.wqkv.weight", "attn.wo.weight",
    "ffn.w_gate.weight", "ffn.w_up.weight", "ffn.w_down.weight",
)
# AdamW로 가는 2D 파라미터 중 "예상된" 것들 — 경고를 내지 않을 이름 조각.
_KNOWN_ADAMW_2D = (
    "tok_emb", "lm_head", "mood_film", "mood_read", "latent_proj",
    "feedback_proj", "conf_head", "workspace", "ws_read", "ws_write",
    "attn_schema",
)


class OptimizerBundle:
    """여러 옵티마이저를 하나의 인터페이스로 감싼다.

    - LR 스케줄: 각 그룹의 base_lr에 공통 스케일을 곱한다(코사인 계수).
    - state_dict/load_state_dict: {"muon":..., "adamw":...} 형태로 저장.
      옵티마이저 종류가 바뀐 예전 체크포인트를 로드하면 state 불일치이므로
      가중치만 살리고 옵티마이저는 새로 시작한다(경고). resume이 죽지 않게.
    """

    def __init__(self, named_optimizers: dict[str, torch.optim.Optimizer]):
        self.named = named_optimizers
        self.optimizers = list(named_optimizers.values())
        for opt in self.optimizers:
            for g in opt.param_groups:
                g.setdefault("base_lr", g["lr"])

    def set_lr_scale(self, scale: float):
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

    def current_lrs(self) -> dict[str, float]:
        return {name: opt.param_groups[0]["lr"] for name, opt in self.named.items()}

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {name: opt.state_dict() for name, opt in self.named.items()}

    def load_state_dict(self, sd: dict):
        # 예전 단일 AdamW 체크포인트({"state":..,"param_groups":..})는 키가 다르다.
        if set(sd.keys()) != set(self.named.keys()):
            print("  옵티마이저 구성이 체크포인트와 다름 — 옵티마이저 state 새로 시작")
            return
        for name, opt in self.named.items():
            try:
                opt.load_state_dict(sd[name])
            except (ValueError, KeyError) as e:
                print(f"  옵티마이저 '{name}' state 불일치 ({e}) — 새로 시작")


def build_optimizer_bundle(model, cfg) -> OptimizerBundle:
    """cfg.optimizer 에 따라 AdamW-단일 또는 Muon+AdamW 하이브리드를 만든다.

    이름 기반 화이트리스트로 파라미터를 가른다(정규식 하드코딩 금지):
    블록 내부 2D 행렬만 Muon, 나머지는 AdamW.
    """
    betas = (cfg.beta1, cfg.beta2)
    raw = getattr(model, "_orig_mod", model)  # torch.compile 래핑 대비

    if cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            [p for p in raw.parameters() if p.requires_grad],
            lr=cfg.learning_rate, betas=betas, weight_decay=cfg.weight_decay,
        )
        return OptimizerBundle({"adamw": opt})

    if cfg.optimizer != "muon":
        raise ValueError(f"알 수 없는 optimizer: {cfg.optimizer}")

    muon_params, adamw_params = [], []
    seen = set()
    for name, p in raw.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if name.endswith(_MUON_SUFFIXES):
            muon_params.append(p)
        else:
            adamw_params.append(p)
            if p.ndim == 2 and not any(k in name for k in _KNOWN_ADAMW_2D):
                print(f"  경고: 2D 파라미터 '{name}'가 Muon/화이트리스트 어디에도"
                      f" 안 걸림 — AdamW로 처리")

    muon = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95,
                weight_decay=cfg.weight_decay)
    # AdamW 대상(임베딩/헤드/norm/bias)은 weight_decay 0 — 항등/제로 init 동역학
    # 과 작은 헤드를 감쇠로 흔들지 않는다.
    adamw = torch.optim.AdamW(adamw_params, lr=cfg.learning_rate,
                              betas=betas, weight_decay=0.0)
    print(f"Muon 대상 {sum(p.numel() for p in muon_params)/1e6:.1f}M "
          f"/ AdamW 대상 {sum(p.numel() for p in adamw_params)/1e6:.1f}M")
    return OptimizerBundle({"muon": muon, "adamw": adamw})
