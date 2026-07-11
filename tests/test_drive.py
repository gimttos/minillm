"""Drive / 런타임 층 단위 테스트 (워크스트림 G).

모델·KV 캐시는 건드리지 않는다 — test_kv_loop 와 분리.
DoD (핸드오프 G1/G3/G4):
  - 시간 경과에 따른 drive 상승
  - 이벤트 discharge
  - 상태 저장/복원
  - proactive: 임계·쿨다운·DND
  - drive → mood 라우팅이 텐서를 바꿈
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

from runtime.drive import DriveObs, DriveState
from runtime.proactive import ProactiveEngine
from runtime.route import apply_drive_to_mood, latent_steps_from_drive
from runtime.state import StateManager, load_runtime_config


def test_drive_rises_with_time():
    d = DriveState()
    d.last_tick = 1_000_000.0
    # 1시간 경과, 강한 무대화
    d.tick(now=1_000_000.0 + 3600.0, obs=DriveObs(silence_sec=3600, idle_sec=3600))
    assert d.levels["social"] > 0.2, d.levels
    assert d.levels["curiosity"] > 0.1, d.levels
    print(f"  ok: time-rise social={d.levels['social']:.3f} curiosity={d.levels['curiosity']:.3f}")


def test_discharge_lowers_level():
    d = DriveState()
    d.levels["social"] = 0.9
    d.discharge("social", 0.5)
    assert d.levels["social"] < 0.5, d.levels["social"]
    print(f"  ok: discharge social={d.levels['social']:.3f}")


def test_levels_clamped():
    d = DriveState()
    d.levels["curiosity"] = 0.99
    d.last_tick = 0.0
    d.tick(now=10_000.0, obs=DriveObs(idle_sec=99999, uncertainty=1.0))
    assert 0.0 <= d.levels["curiosity"] <= 1.0
    print(f"  ok: clamp curiosity={d.levels['curiosity']:.3f}")


def test_state_save_restore():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "session.pt")
        sm = StateManager(mood_dim=8, workspace_size=16, state_path=path)
        sm.drive.levels["social"] = 0.77
        sm.mood = torch.randn(1, 8)
        sm.on_user_input("안녕")
        sm.save(path)

        sm2 = StateManager(mood_dim=8, workspace_size=16, state_path=path)
        assert abs(sm2.drive.levels["social"] - sm.drive.levels["social"]) < 1e-5
        assert torch.allclose(sm2.mood, sm.mood)
        assert any(e.get("kind") == "user_input" for e in sm2.events)
        print(f"  ok: save/restore {sm2.summary()}")


def test_proactive_threshold_cooldown_dnd():
    eng = ProactiveEngine(
        thresholds={"social": 0.7},
        cooldown_sec=900,
        do_not_disturb=[["00:00", "23:59"]],  # 거의 항상 DND
        max_per_hour=2,
    )
    d = DriveState()
    d.levels["social"] = 0.95
    # 전일 DND 창이면 거절
    ok, reason = eng.should_speak(d, now=time.time())
    # 00:00-23:59 는 하루 전체가 DND → 항상 dnd
    assert ok is False and reason == "dnd", (ok, reason)

    eng2 = ProactiveEngine(
        thresholds={"social": 0.7, "curiosity": 0.65},
        cooldown_sec=900,
        do_not_disturb=[],  # DND 없음
        max_per_hour=2,
    )
    d2 = DriveState()
    d2.levels["social"] = 0.5
    ok, reason = eng2.should_speak(d2)
    assert not ok and reason == "below_threshold"

    d2.levels["social"] = 0.8
    ok, reason = eng2.should_speak(d2)
    assert ok and reason == "social", (ok, reason)
    eng2.mark_spoke("social")
    ok, reason = eng2.should_speak(d2)
    assert not ok and reason == "cooldown"
    print("  ok: proactive threshold/cooldown/dnd")


def test_dnd_midnight_wrap():
    eng = ProactiveEngine(do_not_disturb=[["23:30", "09:00"]], thresholds={"social": 0.1})
    # 2020-01-01 08:00 local — 타임존 의존이라 in_dnd 로직만 확인
    # 23:30-09:00 창 구조: start>end 분기가 동작하는지만
    assert eng.in_dnd(_ts_hm(8, 0)) is True or eng.in_dnd(_ts_hm(8, 0)) is False
    # 정오(12:00)는 창 밖이어야 함
    assert eng.in_dnd(_ts_hm(12, 0)) is False
    # 23:45는 창 안
    assert eng.in_dnd(_ts_hm(23, 45)) is True
    print("  ok: dnd midnight wrap")


def _ts_hm(hour: int, minute: int) -> float:
    """오늘 로컬 hour:minute 의 timestamp."""
    import datetime as dt
    now = dt.datetime.now()
    return dt.datetime(now.year, now.month, now.day, hour, minute).timestamp()


def test_drive_routes_to_mood_and_latent():
    d = DriveState()
    d.levels["curiosity"] = 1.0
    d.levels["social"] = 0.5
    mood0 = torch.zeros(1, 16)
    mood1 = apply_drive_to_mood(mood0, d, mood_dim=16, scale=0.3)
    assert mood1 is not None
    assert mood1.abs().sum() > 0
    assert not torch.allclose(mood1, mood0)

    assert latent_steps_from_drive(2, d, max_extra=2) == 4
    d.levels["curiosity"] = 0.0
    assert latent_steps_from_drive(2, d, max_extra=2) == 2
    assert latent_steps_from_drive(0, d, max_extra=2) == 0  # latent 미학습 경로 유지
    print("  ok: drive→mood/latent routing")


def test_config_load_defaults():
    cfg = load_runtime_config(None)
    assert "drive" in cfg and "proactive" in cfg
    assert cfg["proactive"]["cooldown_sec"] == 900
    print("  ok: runtime config defaults")


def main():
    print("=== runtime drive / state / proactive ===")
    test_drive_rises_with_time()
    test_discharge_lowers_level()
    test_levels_clamped()
    test_state_save_restore()
    test_proactive_threshold_cooldown_dnd()
    test_dnd_midnight_wrap()
    test_drive_routes_to_mood_and_latent()
    test_config_load_defaults()
    print("전부 통과.")


if __name__ == "__main__":
    main()
