"""DriveState — 외부에서 계산하는 항상성 동인 (워크스트림 G1).

왜 모델 밖인가
==============
drive 라벨 데이터가 없어 30M에서 drive_head를 학습할 수 없다.
그래서 벽시계·이벤트 관측으로 바깥에서 계산하고, 그 결과는 반드시
내부 채널(mood / workspace / latent step)을 통해서만 사고에 스며든다.
모델을 우회해 출력을 게이팅하는 래퍼는 금지(불변식 7, Part 1.3).

동역학
======
각 drive는 관측 신호 + 시간 경과로 상승하고, discharge로 감쇠한다(항상성).
"턴 단위 갱신"이 아니라 **벽시계 기반** — 대화가 없어도 시간이 지나면 변한다.

초기 4종
--------
- curiosity   : 입력 적음 / 유휴 / 내부 불확실성 → 탐색 욕구
- rest        : 장시간 활동 후 → 쉬고 싶음
- social      : 무대화 지속 → 말을 걸고 싶음
- maintenance : 시간 경과·파일 이벤트 등 → 정리/점검
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_KINDS = ("curiosity", "rest", "social", "maintenance")

# discharge 한 번이 남기는 잔존 비율 (대략 "한 이벤트면 꽤 가라앉는다")
DEFAULT_DISCHARGE_FRACTION = 0.35


@dataclass
class DriveObs:
    """한 틱의 감각 관측 (G5). 없는 필드는 0/False로 두면 순수 시간 상승만 적용."""
    idle_sec: float = 0.0          # 마지막 사용자 입력 이후
    activity_sec: float = 0.0      # 이번 세션 누적 활동(대화) 시간
    silence_sec: float = 0.0       # 마지막 대화(어느 쪽이든) 이후
    hour: float = 12.0             # 로컬 시각 0~23
    file_event: bool = False       # 감시 파일 mtime 변화 등
    uncertainty: float = 0.0       # 0~1, conf 헤드 등에서 온 내부 불확실성


@dataclass
class DriveState:
    """4종 drive 레벨(0~1)과 벽시계 갱신."""

    kinds: tuple[str, ...] = DEFAULT_KINDS
    levels: dict[str, float] = field(default_factory=dict)
    # discharge 후 반감기(초): 길수록 천천히 다시 찬다기보다, discharge 규모 힌트
    halflife_sec: dict[str, float] = field(default_factory=dict)
    # 시간당 기본 상승량 (관측 보너스 전)
    rise_per_hour: dict[str, float] = field(default_factory=dict)
    last_tick: float = field(default_factory=time.time)

    def __post_init__(self):
        for k in self.kinds:
            self.levels.setdefault(k, 0.0)
            self.halflife_sec.setdefault(k, {
                "curiosity": 600, "rest": 3600,
                "social": 1800, "maintenance": 900,
            }.get(k, 1800))
            self.rise_per_hour.setdefault(k, {
                "curiosity": 0.35, "rest": 0.20,
                "social": 0.40, "maintenance": 0.15,
            }.get(k, 0.25))

    # ------------------------------------------------------------------
    # 갱신
    # ------------------------------------------------------------------
    def tick(self, now: float | None = None, obs: DriveObs | None = None) -> dict[str, float]:
        """벽시계 경과 + 관측 신호로 drive를 상승시킨다. 레벨은 [0, 1]로 유계."""
        now = time.time() if now is None else now
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now
        if dt == 0.0 and obs is None:
            return dict(self.levels)

        obs = obs or DriveObs()
        hours = dt / 3600.0

        # 관측 보너스: 시간당 상승에 곱해질 계수(≥1). 신호 없으면 1.
        bonus = {
            "curiosity": 1.0
                + min(obs.idle_sec / 1800.0, 2.0)          # 30분 유휴 → +1
                + min(obs.uncertainty, 1.0),               # 내부 불확실
            "rest": 1.0
                + min(obs.activity_sec / 3600.0, 2.0),     # 1시간 활동 → +1
            "social": 1.0
                + min(obs.silence_sec / 1800.0, 2.5),      # 무대화
            "maintenance": 1.0
                + (0.5 if obs.file_event else 0.0)
                + _night_bonus(obs.hour),                  # 새벽 점검 감
        }

        for k in self.kinds:
            rise = self.rise_per_hour[k] * hours * bonus.get(k, 1.0)
            self.levels[k] = _clamp01(self.levels[k] + rise)
        return dict(self.levels)

    def discharge(self, kind: str, fraction: float = DEFAULT_DISCHARGE_FRACTION) -> None:
        """이벤트 후 해당 drive를 감쇠 (항상성 충족).

        fraction=0.35 → 약 65% 남김. 여러 번 부르면 더 가라앉는다.
        """
        if kind not in self.levels:
            return
        frac = _clamp01(fraction)
        self.levels[kind] = _clamp01(self.levels[kind] * (1.0 - frac))

    def discharge_many(self, kinds: Iterable[str],
                       fraction: float = DEFAULT_DISCHARGE_FRACTION) -> None:
        for k in kinds:
            self.discharge(k, fraction)

    def decay_toward_zero(self, dt_sec: float) -> None:
        """선택: 시간만 흐를 때 지수 감쇠(반감기 사용). 보통 tick 상승이
        주 경로라 필수는 아니지만, 테스트·안정성용으로 둔다."""
        for k in self.kinds:
            hl = max(self.halflife_sec[k], 1.0)
            factor = math.exp(-math.log(2) * dt_sec / hl)
            self.levels[k] = _clamp01(self.levels[k] * factor)

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------
    def as_vector(self) -> list[float]:
        return [self.levels[k] for k in self.kinds]

    def to_dict(self) -> dict:
        return {
            "kinds": list(self.kinds),
            "levels": dict(self.levels),
            "halflife_sec": dict(self.halflife_sec),
            "rise_per_hour": dict(self.rise_per_hour),
            "last_tick": self.last_tick,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DriveState":
        kinds = tuple(d.get("kinds", DEFAULT_KINDS))
        return cls(
            kinds=kinds,
            levels={k: float(d.get("levels", {}).get(k, 0.0)) for k in kinds},
            halflife_sec={k: float(v) for k, v in d.get("halflife_sec", {}).items()},
            rise_per_hour={k: float(v) for k, v in d.get("rise_per_hour", {}).items()},
            last_tick=float(d.get("last_tick", time.time())),
        )

    def __repr__(self) -> str:
        parts = " ".join(f"{k}={self.levels[k]:.2f}" for k in self.kinds)
        return f"DriveState({parts})"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _night_bonus(hour: float) -> float:
    """새벽(0~6시)에 maintenance가 조금 더 찬다 — '정리하고 싶다' 감각의 최소 대리."""
    h = hour % 24
    return 0.4 if h < 6 else 0.0
