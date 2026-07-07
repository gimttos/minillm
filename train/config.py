"""학습 설정 모음.

- tiny : 로컬 CPU에서 파이프라인이 도는지 검증하는 초소형 설정 (몇 분)
- full : 클라우드 T4 GPU에서 실제로 쓸 ~30M 설정

학습 스크립트는 --preset tiny|full 로 골라 쓴다.
"""

from dataclasses import dataclass, field, asdict

from model.gpt import ModelConfig


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)

    # 데이터 / 체크포인트 경로
    data_dir: str = "data/bin"
    out_dir: str = "checkpoints"

    # 배치: 실제 배치 = batch_size * grad_accum (GPU 메모리에 맞춰 나눠 처리)
    batch_size: int = 24
    grad_accum: int = 20            # -> 유효 배치 480 시퀀스
    max_steps: int = 40000
    warmup_steps: int = 1000

    learning_rate: float = 6e-4
    min_lr: float = 6e-5            # cosine 스케줄의 최저점
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    eval_interval: int = 500        # val loss + 샘플 생성 주기
    eval_iters: int = 100
    log_interval: int = 20
    save_interval: int = 1000

    device: str = "cuda"            # 클라우드 기본값; tiny에서 cpu로 덮어씀
    dtype: str = "bfloat16"         # T4는 fp16, A100/신형은 bf16. cpu는 float32
    compile: bool = True            # torch.compile — GPU에서만 이득

    seed: int = 1337


def get_config(preset: str) -> TrainConfig:
    if preset == "full":
        return TrainConfig()

    if preset == "tiny":
        # 로컬 CPU에서 수 분 내로 도는 초소형 설정. 목적은 성능이 아니라
        # "코드가 학습되긴 하는가"를 오버핏으로 확인하는 것.
        return TrainConfig(
            model=ModelConfig(
                vocab_size=16384, d_model=128, n_layers=2, n_heads=4,
                ffn_hidden=352, max_seq_len=128,
            ),
            batch_size=8, grad_accum=1, max_steps=200, warmup_steps=20,
            eval_interval=50, eval_iters=20, log_interval=10, save_interval=100,
            device="cpu", dtype="float32", compile=False,
            learning_rate=1e-3,
        )

    raise ValueError(f"알 수 없는 preset: {preset}")
