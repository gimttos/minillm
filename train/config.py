"""학습 설정 모음.

- tiny : 로컬 CPU에서 파이프라인이 도는지 검증하는 초소형 설정 (몇 분)
- full : 클라우드 T4 GPU에서 실제로 쓸 ~30M 설정 (base — loop off, compile on)
- full-loop : loop를 사전학습에 넣고 싶을 때의 예전 full (권장 경로 아님)

학습 스크립트는 --preset tiny|full 로 골라 쓴다.

권장 경로(§A4): `full`(base) 사전학습 → SFT에서 loop/mood/latent/... 부여.
제일 비싼 사전학습 단계에서 loop(1.5배 연산)와 torch.compile 비활성을 걷어내
속도를 벌고, 마음 기제는 30분~1시간짜리 SFT에서 붙인다.
"""

from dataclasses import dataclass, field, asdict

from model.gpt import ModelConfig


def pick_amp_dtype(device: str) -> str:
    """장치에 맞는 AMP 연산 dtype을 고른다.

    T4(Turing, capability 7.5)는 bf16 텐서코어가 없어 autocast(bf16)이 가속을
    못 받는다 — 이 경우 fp16이 정답(GradScaler 경로로). Ampere(cap>=8) 이상만
    bf16이 하드웨어로 빠르다. CPU는 fp32.
    """
    if device != "cuda":
        return "float32"
    try:
        import torch
        if not torch.cuda.is_available():
            return "float32"
        cap = torch.cuda.get_device_capability(0)[0]
    except Exception:
        return "float16"
    return "bfloat16" if cap >= 8 else "float16"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)

    # 데이터 / 체크포인트 경로
    data_dir: str = "data/bin"
    out_dir: str = "checkpoints"

    # 배치: 실제 배치 = batch_size * grad_accum (GPU 메모리에 맞춰 나눠 처리)
    batch_size: int = 24
    grad_accum: int = 20            # -> 유효 배치 480 시퀀스
    # 토큰 예산 기반 학습(§A1): target_tokens > 0 이면 max_steps를 여기서 유도한다.
    # 30M 모델은 ~1B 토큰에서 포화 근처 — 그 이상은 시간만 낭비.
    target_tokens: int = 0          # 0이면 아래 max_steps를 그대로 사용
    max_steps: int = 40000
    warmup_steps: int = 1000

    learning_rate: float = 6e-4
    min_lr: float = 6e-5            # cosine 스케줄의 최저점
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # 옵티마이저(§A3): "adamw"(기존) | "muon"(Muon+AdamW 하이브리드)
    optimizer: str = "adamw"
    muon_lr: float = 0.02           # Muon 대상(블록 2D 행렬)의 시작 LR

    eval_interval: int = 500        # val loss + 샘플 생성 주기
    eval_iters: int = 100
    log_interval: int = 20
    save_interval: int = 1000

    device: str = "cuda"            # 클라우드 기본값; tiny에서 cpu로 덮어씀
    # "auto"면 장치 능력으로 T4=fp16 / Ampere+=bf16 / cpu=fp32 자동 선택(§A2).
    # 명시 문자열("bfloat16" 등)을 주면 그 값을 우선.
    dtype: str = "auto"
    compile: bool = True            # torch.compile — GPU에서만 이득

    seed: int = 1337

    def resolve_max_steps(self) -> int:
        """target_tokens가 설정돼 있으면 유효배치 토큰 수로 max_steps를 유도.
        예: 1.0B / (24*20*512=245760) ≈ 4069 스텝."""
        if self.target_tokens > 0:
            per_step = self.batch_size * self.grad_accum * self.model.max_seq_len
            return max(self.target_tokens // per_step, 1)
        return self.max_steps


def get_config(preset: str) -> TrainConfig:
    if preset == "full":
        # base 사전학습: loop off + compile on (§A4). 마음 기제는 SFT로 부여.
        # vocab 16392 = 특수 토큰 12개 예약분 포함 (tokenizer/bpe.py 참조).
        # 토큰 예산 1B(§A1): 유효배치 480×512 기준 약 4,000스텝.
        return TrainConfig(
            model=ModelConfig(vocab_size=16392),   # loop off (loop_* 기본 0)
            batch_size=24, grad_accum=20,          # 유효 480, loop 없어 메모리 여유
            target_tokens=1_000_000_000,
            warmup_steps=250,                      # ~ 스텝의 6%
            # Kaggle 12h 세션이 언제 끊길지 모른다 — 자주 저장해 재개 손실을 줄인다.
            eval_interval=100, save_interval=100,
            compile=True,                          # loop off라 그래프가 고정 → compile 가능
        )

    if preset == "full-loop":
        # loop를 사전학습에 넣는 예전 경로 (권장 아님 — full 후 SFT를 쓸 것).
        # 중간 4블록(2~5)을 그룹째 2회 통과 -> 실효 깊이 12, 연산 1.5배.
        # 확률적 n_loop 때문에 그래프가 매번 달라져 torch.compile을 끈다.
        return TrainConfig(
            model=ModelConfig(
                vocab_size=16392,
                loop_start=2, loop_end=6, n_loop=2,
            ),
            batch_size=16, grad_accum=30,          # loop 메모리 ~1.5배 → 유효 480 유지
            target_tokens=1_000_000_000,
            warmup_steps=250,
            eval_interval=100, save_interval=100,
            compile=False,
        )

    if preset == "tiny":
        # 로컬 CPU에서 수 분 내로 도는 초소형 설정. 목적은 성능이 아니라
        # "코드가 학습되긴 하는가"를 오버핏으로 확인하는 것.
        return TrainConfig(
            model=ModelConfig(
                vocab_size=16392, d_model=128, n_layers=2, n_heads=4,
                ffn_hidden=352, max_seq_len=128,
            ),
            batch_size=8, grad_accum=1, max_steps=200, warmup_steps=20,
            eval_interval=50, eval_iters=20, log_interval=10, save_interval=100,
            device="cpu", dtype="float32", compile=False,
            learning_rate=1e-3,
        )

    if preset == "tiny-loop":
        # loop 경로의 로컬 검증용: 4층 중 가운데 2층(1~2)을 2회 반복.
        # 2층짜리 tiny로는 loop가 의미 있게 검증되지 않아 층을 4로 늘렸다.
        return TrainConfig(
            model=ModelConfig(
                vocab_size=16392, d_model=128, n_layers=4, n_heads=4,
                ffn_hidden=352, max_seq_len=128,
                loop_start=1, loop_end=3, n_loop=2,
            ),
            batch_size=8, grad_accum=1, max_steps=200, warmup_steps=20,
            eval_interval=50, eval_iters=20, log_interval=10, save_interval=100,
            device="cpu", dtype="float32", compile=False,
            learning_rate=1e-3,
        )

    raise ValueError(f"알 수 없는 preset: {preset}")
