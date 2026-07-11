"""ProactiveEngine — drive 임계 + 쿨다운 + DND (워크스트림 G4).

프레이밍
========
proactive 출력은 지표 증거가 아니다(ELIZA 경계, Part 1.4).
경험적 질감용이며, "먼저 말을 건다"는 행위성(agency)의 **부분·외부** 구현.

안전장치 (없으면 성가셔서 제작자 의욕이 먼저 죽음)
====================================================
1. drive 임계 초과 시에만
2. 마지막 proactive 이후 cooldown_sec 경과
3. do_not_disturb 시간대면 침묵
4. max_per_hour 상한

발화 자체는 chat.py가 내부 latent 사고 → generate 로 수행한다.
이 모듈은 "지금 말해도 되는가 / 어떤 drive 때문인가"만 판정.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from runtime.drive import DriveState


@dataclass
class ProactiveEngine:
    thresholds: dict[str, float] = field(default_factory=lambda: {
        "social": 0.7, "curiosity": 0.65,
    })
    cooldown_sec: float = 900.0
    do_not_disturb: list[list[str]] = field(default_factory=lambda: [["23:30", "09:00"]])
    max_per_hour: int = 2
    last_proactive_ts: float = 0.0
    recent_proactive_ts: list[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ProactiveEngine":
        p = cfg.get("proactive", {})
        return cls(
            thresholds=dict(p.get("thresholds", {"social": 0.7, "curiosity": 0.65})),
            cooldown_sec=float(p.get("cooldown_sec", 900)),
            do_not_disturb=list(p.get("do_not_disturb", [["23:30", "09:00"]])),
            max_per_hour=int(p.get("max_per_hour", 2)),
        )

    def should_speak(
        self,
        drive: DriveState,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """(발화 여부, 트리거 kind 또는 거절 사유).

        사유 문자열: kind 이름이면 발화 승인, 그 외는 거절 이유.
        """
        now = time.time() if now is None else now

        if self.in_dnd(now):
            return False, "dnd"

        if self.last_proactive_ts and (now - self.last_proactive_ts) < self.cooldown_sec:
            return False, "cooldown"

        # 최근 1시간 발화 수
        hour_ago = now - 3600.0
        self.recent_proactive_ts = [t for t in self.recent_proactive_ts if t >= hour_ago]
        if len(self.recent_proactive_ts) >= self.max_per_hour:
            return False, "rate_limit"

        # 임계를 넘는 drive 중 가장 높은 것
        best_kind, best_val = "", -1.0
        for kind, thr in self.thresholds.items():
            val = float(drive.levels.get(kind, 0.0))
            if val >= thr and val > best_val:
                best_kind, best_val = kind, val

        if not best_kind:
            return False, "below_threshold"

        return True, best_kind

    def mark_spoke(self, kind: str = "", now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.last_proactive_ts = now
        self.recent_proactive_ts.append(now)

    def in_dnd(self, now: float | None = None) -> bool:
        """do_not_disturb 구간이면 True. 자정 넘김 구간(23:30–09:00) 지원."""
        now = time.time() if now is None else now
        minutes = _local_minutes(now)
        for window in self.do_not_disturb:
            if len(window) != 2:
                continue
            start = _parse_hhmm(window[0])
            end = _parse_hhmm(window[1])
            if start <= end:
                if start <= minutes < end:
                    return True
            else:
                # 자정 넘김: start..24h 또는 0..end
                if minutes >= start or minutes < end:
                    return True
        return False

    def to_dict(self) -> dict:
        return {
            "last_proactive_ts": self.last_proactive_ts,
            "recent_proactive_ts": list(self.recent_proactive_ts),
        }

    def load_dict(self, d: dict) -> None:
        self.last_proactive_ts = float(d.get("last_proactive_ts", 0.0))
        self.recent_proactive_ts = [float(t) for t in d.get("recent_proactive_ts", [])]


def _local_minutes(ts: float) -> int:
    dt = datetime.fromtimestamp(ts)
    return dt.hour * 60 + dt.minute


def _parse_hhmm(s: str) -> int:
    """'23:30' → 분 단위 정수."""
    parts = s.strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h * 60 + m
