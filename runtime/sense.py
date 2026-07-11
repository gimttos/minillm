"""가벼운 감각 어댑터 (워크스트림 G5).

일반화된 감각 변환은 비목표. 지금 쓰는 관측만:
  - 현재 시각(시간대)
  - 유휴/무대화 시간
  - 선택: 감시 경로 파일 mtime 변화

drive.tick()에 넘길 DriveObs만 만든다. 모델 입력 토큰으로 바꾸지 않는다.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from runtime.drive import DriveObs


def observe(
    now: float | None = None,
    last_user_ts: float = 0.0,
    last_activity_ts: float = 0.0,
    activity_sec: float = 0.0,
    watch_paths: list[str] | None = None,
    prev_mtimes: dict[str, float] | None = None,
    uncertainty: float = 0.0,
) -> DriveObs:
    """현재 시점의 관측 묶음."""
    now = time.time() if now is None else now
    hour = datetime.fromtimestamp(now).hour + datetime.fromtimestamp(now).minute / 60.0

    idle = max(0.0, now - last_user_ts) if last_user_ts else 0.0
    silence = max(0.0, now - last_activity_ts) if last_activity_ts else 0.0

    file_event = False
    if watch_paths and prev_mtimes is not None:
        for p in watch_paths:
            try:
                m = Path(p).stat().st_mtime
            except OSError:
                continue
            prev = prev_mtimes.get(p)
            if prev is not None and m > prev:
                file_event = True
            prev_mtimes[p] = m

    return DriveObs(
        idle_sec=idle,
        activity_sec=activity_sec,
        silence_sec=silence,
        hour=hour,
        file_event=file_event,
        uncertainty=uncertainty,
    )
