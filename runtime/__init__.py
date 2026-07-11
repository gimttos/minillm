"""런타임 층 — 모델 밖, 학습 불필요.

Drive / 상태 지속 / proactive 트리거는 전부 여기 있다.
모델(forward) 안 기제와 연결되는 유일한 통로는
mood 벡터 · workspace 슬롯 · latent step 수다 (불변식 7).
"""

from runtime.drive import DriveState
from runtime.state import StateManager, load_runtime_config
from runtime.proactive import ProactiveEngine
from runtime.route import apply_drive_to_mood, latent_steps_from_drive

__all__ = [
    "DriveState",
    "StateManager",
    "ProactiveEngine",
    "load_runtime_config",
    "apply_drive_to_mood",
    "latent_steps_from_drive",
]
