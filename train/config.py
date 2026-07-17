"""학습 설정 모음.

- tiny  : 로컬 CPU에서 파이프라인이 도는지 검증하는 초소형 설정 (몇 분)
- full  : T4 시절의 ~34M 설정 (base — loop off, compile on)
- large : 4090급 GPU용 ~98M 설정 (아래 설명)
- full-loop : loop를 사전학습에 넣고 싶을 때의 예전 full (권장 경로 아님)

학습 스크립트는 --preset tiny|full|large 로 골라 쓴다.

권장 경로(§A4): base 사전학습 → SFT에서 loop/mood/latent/... 부여.
제일 비싼 사전학습 단계에서 loop(1.5배 연산)와 torch.compile 비활성을 걷어내
속도를 벌고, 마음 기제는 30분~1시간짜리 SFT에서 붙인다.

## 왜 large를 더했나 (34M의 천장)

34M(`full`)은 **Kaggle 무료 T4라는 제약에서 나온 숫자**였다. 4090(24GB)으로
옮기면서 그 제약이 사라졌고, 실제로 `full`의 사전학습은 4090에서 45분밖에
걸리지 않았다 — 하드웨어가 놀고 있었다.

그리고 34M은 1B 토큰에서 이미 포화다(Chinchilla 기준 ~700M이면 충분). 즉
**토큰을 더 넣어도 안 늘고, 병목은 파라미터 수**다. 실전 대화에서 문법은
완벽한데 의미가 이어지지 않는 것(“수험생인데도 저보다 어린 학생이 있을까요?”)이
정확히 이 한계다. 마음 기제 자체는 34M에서도 성립했으므로(워크스페이스 인과
기여 Δ+0.148, 확신도 ECE 0.022), large는 "기제"가 아니라 "말이 통하는가"를
겨냥한 확장이다.

vocab(16392)과 max_seq_len(512)은 그대로라 **토크나이저·패킹된 .bin을 다시
만들 필요가 없다**. 다만 체크포인트 shape이 달라지므로 SFT는 새 base에서
다시 돌려야 한다.
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

    if preset in ("xl", "xxl"):
        # ===== 스케일업 세대 (vocab 32k · context 1024) =====
        #
        # 왜 이 세대가 생겼나: 98M/2B에서 문법은 완벽한데 의미가 이어지지 않았다.
        # "말이 통하는 느낌"의 바닥은 base 규모(파라미터 × 토큰)가 정하고, SFT로는
        # 그 천장을 못 올린다. 마음 기제는 34M에서 이미 성립했으므로(워크스페이스
        # 인과 기여 Δ+0.148, ECE 0.022) 이 세대는 "기제"가 아니라 "언어능력"을
        # 겨냥한다 — 기제를 관찰하려면 그걸 얹을 언어가 먼저 있어야 하니까.
        #
        # 이전 세대(full/large)와 달라지는 두 가지, 둘 다 대화 품질에 직결된다:
        #  - vocab 16392 -> 32768: 한국어에 16k는 작다. 같은 글을 ~15~20% 적은
        #    토큰으로 담아 학습 시간이 줄고 표현도 좋아진다.
        #  - max_seq_len 512 -> 1024: 512는 대화 4~6턴이면 찬다. 문맥 길이는
        #    "대화가 통하는 느낌"에 파라미터만큼 중요하다. RoPE라 테이블만 늘면 된다.
        #
        # 이 둘 때문에 **토크나이저와 .bin을 새로 만들어야 한다**(CLAUDE.md의
        # vocab 16392 규약은 이전 세대 체크포인트용으로 그대로 유효하다 — 옛 모델은
        # 자기 model_config를 들고 있어 계속 로드된다). 경로를 분리해 공존시킨다.
        #
        # 유효 배치는 240 시퀀스 × 1024 = 245,760 토큰/스텝으로 이전과 같게 맞췄다
        # (LR 스케줄 감각을 재사용하기 위해).
        big = preset == "xxl"
        return TrainConfig(
            model=ModelConfig(
                vocab_size=32768,
                max_seq_len=1024,
                # xl : ~350M (GPT-2 medium 계열, head_dim 64)
                # xxl: ~1.1B  (head_dim 128)
                d_model=2048 if big else 1024,
                n_layers=20 if big else 24,
                n_heads=16,
                ffn_hidden=5632 if big else 2816,   # ≈ 8/3 × d_model, 128의 배수
            ),
            data_dir="data/bin_32k",                # 32k 토크나이저로 새로 패킹한 것
            out_dir="checkpoints_xxl" if big else "checkpoints_xl",
            # 시퀀스가 2배 길어졌으니 마이크로배치를 줄여 메모리를 맞춘다.
            # (유효 배치 240 시퀀스는 유지 — Muon이 모멘텀 1개만 들어 1B도 24GB에 들어간다)
            batch_size=4 if big else 12,
            grad_accum=60 if big else 20,
            target_tokens=20_000_000_000 if big else 8_000_000_000,  # ≈ 파라미터당 20토큰
            warmup_steps=2000 if big else 1000,     # ~ 스텝의 3~6%
            eval_interval=500, save_interval=500,   # 길게 도는 학습이라 재개 손실 관리
            compile=True,
        )

    if preset == "large":
        # 4090급(24GB) base 사전학습: ~98M. 34M의 천장을 넘기 위한 확장.
        #
        # 형태는 GPT-2 small 계열(768 / 12층 / 12헤드, head_dim 64)로 잡았다 —
        # 검증된 비율이고, head_dim이 full(64)과 같아 RoPE 배선이 그대로다.
        # ffn_hidden 2048 ≈ 8/3 × 768 (SwiGLU 관례).
        #
        # 토큰 예산 2B: Chinchilla 어림(파라미터당 ~20토큰)으로 98M × 20 ≈ 2B.
        # 유효배치 480×512=245,760 토큰/스텝 -> 약 8,100스텝.
        # 코퍼스(위키+나무 121만 문서)가 2B에 못 미치면 2~3에폭 반복이 되는데,
        # 이 규모에서 소수 에폭 반복은 새 데이터와 거의 동등하다고 알려져 있다.
        #
        # vocab·max_seq_len이 full과 같아 tokenizer/.bin을 다시 만들 필요가 없다.
        # 단 체크포인트 shape이 달라지므로 SFT는 이 base에서 다시 돌려야 한다.
        return TrainConfig(
            model=ModelConfig(
                vocab_size=16392,
                d_model=768, n_layers=12, n_heads=12, ffn_hidden=2048,
            ),
            # 34M(full)과 체크포인트가 섞이지 않게 별도 폴더 — 34M 결과(워크스페이스
            # 인과 기여·확신도 캘리브레이션)는 보존 가치가 있고, --resume 이 엉뚱한
            # 아키텍처를 집어 들지도 않는다.
            out_dir="checkpoints_large",
            batch_size=24, grad_accum=20,          # 유효 480 유지 (4090 24GB에 여유)
            target_tokens=2_000_000_000,
            warmup_steps=500,                      # ~ 스텝의 6%
            eval_interval=200, save_interval=200,  # 재개 손실을 줄이되 4090이라 덜 잦게
            compile=True,                          # loop off라 그래프 고정 → compile 가능
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
