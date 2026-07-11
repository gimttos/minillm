"""Drive → 내부 채널 라우팅 (워크스트림 G2).

경계 규칙
=========
밖에서 안으로 가는 유일한 통로 = mood 벡터 / workspace 슬롯 / latent step 수.
drive를 모델 입력 토큰으로 바꾸거나 출력 logits를 밖에서 게이팅하지 않는다.

왜 mood인가
===========
mood는 매 블록 FiLM으로 forward에 스며든다. drive가 mood에 가산되면
"사고의 기질"이 바뀌고, 그 결과 출력이 바뀐다 — 모델 위에 덧씌우는 게 아니라
모델 안으로 녹아드는 경로(Part 1.3 "갈아끼우면 유지 안 됨" 문제의 구조적 해법).

학습 없는 결정적 투영
====================
drive_head는 Phase 4 이후 선택. 지금은 4차원 drive를 mood_dim으로
고정 패턴 확장한다(시드 불변). 켠 직후 scale이 작아 baseline을 크게 흔들지 않음.
"""

from __future__ import annotations

import torch

from runtime.drive import DriveState


def apply_drive_to_mood(
    mood: torch.Tensor | None,
    drive: DriveState,
    mood_dim: int,
    device: torch.device | str = "cpu",
    scale: float = 0.3,
) -> torch.Tensor | None:
    """drive 벡터를 mood에 가산한 새 텐서를 반환한다.

    mood가 None이고 mood_dim>0이면 0에서 시작해 drive 편향만 싣는다.
    mood_dim==0이면 라우팅 불가 → 입력 mood를 그대로 반환.
    """
    if mood_dim <= 0:
        return mood

    bias = _drive_bias(drive, mood_dim, device, scale)
    if mood is None:
        return bias
    # shape 맞추기: mood (B, mood_dim) 또는 (mood_dim,)
    if mood.dim() == 1:
        mood = mood.unsqueeze(0)
    return mood + bias.to(device=mood.device, dtype=mood.dtype)


def latent_steps_from_drive(
    base_latent: int,
    drive: DriveState,
    max_extra: int = 2,
) -> int:
    """curiosity가 높을수록 말하기 전 잠재 스텝을 늘린다.

    curiosity 0 → base, 1 → base+max_extra. 나머지는 건드리지 않는다.
    base가 0(latent 미학습)이면 0을 유지 — 학습 안 된 경로를 억지로 켜지 않음.
    """
    if base_latent <= 0 or max_extra <= 0:
        return base_latent
    c = float(drive.levels.get("curiosity", 0.0))
    extra = int(round(c * max_extra))
    return base_latent + extra


def workspace_nudge_from_drive(
    ws: torch.Tensor | None,
    drive: DriveState,
    scale: float = 0.05,
) -> torch.Tensor | None:
    """선택: maintenance/curiosity를 workspace 앞 몇 차원에 약하게 더한다.
    scale이 매우 작아 슬롯을 망가뜨리지 않는다. ws가 None이면 None."""
    if ws is None or ws.numel() == 0:
        return ws
    v = drive.as_vector()
    n = min(len(v), ws.shape[-1])
    out = ws.clone()
    bias = torch.tensor(v[:n], device=ws.device, dtype=ws.dtype) * scale
    out[..., :n] = out[..., :n] + bias
    return out


def _drive_bias(
    drive: DriveState,
    mood_dim: int,
    device: torch.device | str,
    scale: float,
) -> torch.Tensor:
    """4-drive → mood_dim 결정적 확장.

    앞 4차원에 drive 레벨을 넣고, 나머진 작은 주기 패턴으로 채운다.
    학습 가중치 없이 재현 가능 — 같은 drive면 같은 편향.
    """
    vec = drive.as_vector()  # len 4
    bias = torch.zeros(1, mood_dim, device=device)
    n = min(len(vec), mood_dim)
    for i in range(n):
        bias[0, i] = vec[i]
    # 남는 차원: drive 성분의 약한 선형결합 (고정 계수)
    if mood_dim > n:
        for i in range(n, mood_dim):
            bias[0, i] = 0.15 * vec[i % len(vec)] * ((-1) ** i)
    return bias * scale
