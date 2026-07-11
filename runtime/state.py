"""StateManager — mood+workspace+drive+events 메모리 상주 상태 (G3).

설계
====
- 상태는 RAM에 두고, 실제 입력 이벤트 + 느린 idle 타이머에만 갱신·저장한다
  (busy-loop 폴링 금지 — 배터리·발열이 진짜 비용).
- 저장은 기존 mood-file 패턴 확장: 단일 .pt에 mood/ws/drive/events를 묶음.
  새 DB를 재발명하지 않는다.
- 부팅/재시작 시 복원하면 대화 기질이 이어진다.

런타임 정책(임계·쿨다운·저장주기)은 여기 config에만 둔다.
ModelConfig에 넣지 않는다 (Part 3.3).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from runtime.drive import DriveObs, DriveState
from runtime.route import apply_drive_to_mood, latent_steps_from_drive, workspace_nudge_from_drive
from runtime.sense import observe


# ---------------------------------------------------------------------------
# 설정 로드
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "drive": {
        "kinds": ["curiosity", "rest", "social", "maintenance"],
        "discharge_halflife_sec": {
            "curiosity": 600, "rest": 3600,
            "social": 1800, "maintenance": 900,
        },
        "rise_per_hour": {
            "curiosity": 0.35, "rest": 0.20,
            "social": 0.40, "maintenance": 0.15,
        },
        "mood_scale": 0.3,
        "curiosity_latent_extra": 2,
    },
    "proactive": {
        "thresholds": {"social": 0.7, "curiosity": 0.65},
        "cooldown_sec": 900,
        "do_not_disturb": [["23:30", "09:00"]],
        "max_per_hour": 2,
    },
    "state": {
        "idle_tick_sec": 120,
        "recent_events_maxlen": 64,
    },
    "runtime": {
        "lifecycle": "session",
    },
}


def load_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    """JSON 런타임 config 로드. 없거나 실패하면 기본값.

    yaml 대신 JSON — 외부 의존성 없이(requirements에 PyYAML 없음).
    """
    cfg = json.loads(json.dumps(_DEFAULT_CONFIG))  # deep copy
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    with p.open(encoding="utf-8") as f:
        user = json.load(f)
    _deep_update(cfg, user)
    return cfg


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """세션 상태: mood + workspace + drive + recent_events."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        mood_dim: int = 0,
        workspace_size: int = 0,
        device: str | torch.device = "cpu",
        state_path: str = "",
        watch_paths: list[str] | None = None,
    ):
        self.cfg = cfg or load_runtime_config()
        self.mood_dim = mood_dim
        self.workspace_size = workspace_size
        self.device = device
        self.state_path = state_path
        self.watch_paths = list(watch_paths or [])

        dcfg = self.cfg.get("drive", {})
        kinds = tuple(dcfg.get("kinds") or ("curiosity", "rest", "social", "maintenance"))
        self.drive = DriveState(
            kinds=kinds,
            levels={k: 0.0 for k in kinds},
            halflife_sec=dict(dcfg.get("discharge_halflife_sec", {})),
            rise_per_hour=dict(dcfg.get("rise_per_hour", {})),
        )

        self.mood: torch.Tensor | None = (
            torch.zeros(1, mood_dim, device=device) if mood_dim > 0 else None
        )
        self.ws: torch.Tensor | None = (
            torch.zeros(1, workspace_size, device=device) if workspace_size > 0 else None
        )

        maxlen = int(self.cfg.get("state", {}).get("recent_events_maxlen", 64))
        self.events: deque[dict] = deque(maxlen=maxlen)

        now = time.time()
        self.last_user_ts = now
        self.last_activity_ts = now
        self.session_start = now
        self.activity_sec = 0.0
        self._file_mtimes: dict[str, float] = {}

        if state_path and Path(state_path).exists():
            self.load(state_path)

    # ------------------------------------------------------------------
    # 이벤트
    # ------------------------------------------------------------------
    def on_user_input(self, text: str = "") -> None:
        """사용자 입력: social/curiosity discharge + 이벤트 기록 + tick."""
        now = time.time()
        self.activity_sec += max(0.0, now - self.last_activity_ts) * 0.0  # 입력 자체는 활동 시작
        self.last_user_ts = now
        self.last_activity_ts = now
        self._record("user_input", {"n": len(text)})
        self.drive.discharge("social", 0.5)
        self.drive.discharge("curiosity", 0.35)
        self.tick(now)

    def on_assistant_reply(self, n_tokens: int = 0) -> None:
        """봇 응답 후: rest 소량 상승 경로용 활동 누적, curiosity 일부 해소."""
        now = time.time()
        # 응답 생성 시간을 활동으로 잡기 어려우니 토큰 수 기반 근사
        self.activity_sec += min(n_tokens * 0.05, 120.0)
        self.last_activity_ts = now
        self._record("assistant_reply", {"n_tokens": n_tokens})
        self.drive.discharge("curiosity", 0.2)
        self.drive.discharge("social", 0.15)
        # 장시간 활동이면 rest가 tick에서 찬다
        self.tick(now)

    def on_proactive(self, kind: str = "") -> None:
        """proactive 발화 후 해당 drive discharge."""
        now = time.time()
        self.last_activity_ts = now
        self._record("proactive", {"kind": kind})
        if kind:
            self.drive.discharge(kind, 0.55)
        else:
            self.drive.discharge_many(["social", "curiosity"], 0.4)
        self.tick(now)

    def tick(self, now: float | None = None) -> dict[str, float]:
        """idle/감각 관측과 함께 drive 갱신."""
        now = time.time() if now is None else now
        obs = observe(
            now=now,
            last_user_ts=self.last_user_ts,
            last_activity_ts=self.last_activity_ts,
            activity_sec=self.activity_sec,
            watch_paths=self.watch_paths,
            prev_mtimes=self._file_mtimes,
        )
        if obs.file_event:
            self._record("file_event", {})
            self.drive.discharge("maintenance", 0.3)  # 인지했으면 조금 해소
        return self.drive.tick(now, obs)

    def idle_tick(self) -> dict[str, float]:
        """느린 idle 타이머 콜백 — 폴링 최소화 전제에서 주기적으로만 호출."""
        levels = self.tick()
        if self.state_path:
            self.save(self.state_path)
        return levels

    # ------------------------------------------------------------------
    # 내부 채널로 내보내기 (G2)
    # ------------------------------------------------------------------
    def routed_mood(self) -> torch.Tensor | None:
        """모델에 넣을 mood = 세션 mood + drive 편향."""
        scale = float(self.cfg.get("drive", {}).get("mood_scale", 0.3))
        return apply_drive_to_mood(
            self.mood, self.drive, self.mood_dim, self.device, scale=scale,
        )

    def routed_workspace(self) -> torch.Tensor | None:
        return workspace_nudge_from_drive(self.ws, self.drive, scale=0.05)

    def routed_latent(self, base_latent: int) -> int:
        extra = int(self.cfg.get("drive", {}).get("curiosity_latent_extra", 2))
        return latent_steps_from_drive(base_latent, self.drive, max_extra=extra)

    def mood_arg(self) -> torch.Tensor | None:
        """chat.py 규율: 전부 0이면 None (SFT '기분 없음' 경로와 동일)."""
        m = self.routed_mood()
        if m is None or m.abs().max().item() == 0:
            return None
        return m

    def ws_arg(self) -> torch.Tensor | None:
        w = self.routed_workspace()
        if w is None or w.abs().max().item() == 0:
            return None
        return w

    # ------------------------------------------------------------------
    # 모델 갱신 반영 (턴 종료 후 update_mood/ws 결과)
    # ------------------------------------------------------------------
    def set_mood(self, mood: torch.Tensor | None) -> None:
        if mood is not None and self.mood_dim > 0:
            self.mood = mood.detach()

    def set_workspace(self, ws: torch.Tensor | None) -> None:
        if ws is not None and self.workspace_size > 0:
            self.ws = ws.detach()

    # ------------------------------------------------------------------
    # 저장 / 복원
    # ------------------------------------------------------------------
    def save(self, path: str | Path | None = None) -> None:
        path = path or self.state_path
        if not path:
            return
        payload = {
            "version": 1,
            "mood": None if self.mood is None else self.mood.cpu(),
            "workspace": None if self.ws is None else self.ws.cpu(),
            "drive": self.drive.to_dict(),
            "events": list(self.events),
            "last_user_ts": self.last_user_ts,
            "last_activity_ts": self.last_activity_ts,
            "activity_sec": self.activity_sec,
            "session_start": self.session_start,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(data, dict) or "drive" not in data:
            # 구형 mood-file (순수 텐서) 호환
            if torch.is_tensor(data) and self.mood_dim > 0:
                self.mood = data.to(self.device)
            return
        if data.get("mood") is not None and self.mood_dim > 0:
            self.mood = data["mood"].to(self.device)
        if data.get("workspace") is not None and self.workspace_size > 0:
            self.ws = data["workspace"].to(self.device)
        if data.get("drive"):
            self.drive = DriveState.from_dict(data["drive"])
        maxlen = int(self.cfg.get("state", {}).get("recent_events_maxlen", 64))
        self.events = deque(data.get("events") or [], maxlen=maxlen)
        self.last_user_ts = float(data.get("last_user_ts", time.time()))
        self.last_activity_ts = float(data.get("last_activity_ts", time.time()))
        self.activity_sec = float(data.get("activity_sec", 0.0))
        self.session_start = float(data.get("session_start", time.time()))

    def _record(self, kind: str, payload: dict) -> None:
        self.events.append({
            "t": time.time(),
            "kind": kind,
            **payload,
        })

    def summary(self) -> str:
        d = self.drive
        parts = [f"{k}={d.levels[k]:.2f}" for k in d.kinds]
        mn = 0.0 if self.mood is None else float(self.mood.norm())
        wn = 0.0 if self.ws is None else float(self.ws.norm())
        return f"drive[{' '.join(parts)}] mood‖{mn:.3f} ws‖{wn:.3f}"
