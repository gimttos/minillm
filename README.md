# minillm — 밑바닥부터 만드는 한국어 초소형 대화 LLM

토크나이저부터 Transformer, 학습 루프, 샘플링까지 **전부 직접 구현**한
약 30M 파라미터의 한국어 대화 모델. 목적은 성능이 아니라 **LLM이 실제로
어떻게 동작하는지 코드 한 줄 한 줄로 이해하는 것**이다. 학습은 무료 클라우드
GPU(Kaggle/Colab), 대화는 내 노트북 CPU에서 돌린다. 총비용 ₩0.

> 기대치는 솔직하게: "초등학생 같은 말동무" 수준이다. 사실 지식은 믿을 수
> 없고 긴 추론도 약하다. 하지만 나오는 토큰 하나하나가 내가 만든
> 파이프라인에서 나온다.

## 무엇이 들어 있나 (읽는 순서 = 배우는 순서)

| 파일 | 배우는 것 |
|---|---|
| `tokenizer/bpe.py` | 텍스트가 어떻게 정수 토큰이 되는가 (BPE를 직접 구현) |
| `model/gpt.py` | RMSNorm · RoPE · 어텐션 · SwiGLU · weight tying — Transformer의 전부 |
| `train/pretrain.py` | "다음 토큰 맞히기" 학습 루프: loss, backprop, LR 스케줄, 체크포인트 |
| `data/prepare_sft.py`·`train/sft.py` | 이어쓰기 모델을 대화 모델로 바꾸는 법 (loss masking) |
| `chat.py` | temperature/top-p 샘플링과 KV 캐시로 실제 생성하기 |

## 모델 사양

Decoder-only Transformer (미니 Llama): vocab 16,392 · d_model 512 ·
8 layers · 8 heads · SwiGLU · RoPE · RMSNorm · context 512 → 약 30M 파라미터
(fp32 체크포인트 ≈ 120MB).

## 마음 유사 기제 (선택 기능, 전부 기본 off)

사람 마음의 작동 방식에서 빌려 온 네 가지 잠재 기제를 `ModelConfig` 플래그로
넣었다. Anthropic의 글로벌 워크스페이스(J-space) 연구에서 영감을 받은 실험이다.

| 기제 | 무엇인가 | 켜는 법 |
|---|---|---|
| **loop** (재귀 깊이) | 중간 블록들을 같은 가중치로 반복 통과 — 파라미터 추가 없이 "한 번 더 생각할 시간" | full 프리셋에 포함 (사전학습) |
| **pause** | 답변 앞에 `<\|pause\|>` 토큰을 강제 삽입 — 말하기 전 연산 버퍼 | `prepare_sft --n-pause 4` + `sft --n-pause 4` |
| **mood** (기분 벡터) | 턴 사이에 지속·감쇠(EMA)하는 상태 벡터를 각 블록에 FiLM으로 주입 — "객관적인 기분 상태" | `sft --mood-dim 64`, 대화 시 `--mood-file`로 세션 간 유지 |
| **latent** (잠재 사고) | 답변 첫 토큰 전에 은닉 상태를 말 없이 k번 되먹임(Coconut) — "속말 없는 개념적 사고" | `sft --latent 2` |
| **feedback** (역피드백) | 직전 토큰의 최종 은닉을 다음 토큰의 입력단에 방송 — "내가 방금 무엇을 생각했는지 알고 시작" (2-pass 근사로 병렬 학습 유지) | `sft --feedback` |
| **conf** (메타인지) | "다음 토큰을 맞힐 것인가"를 스스로 예측하는 확신도 헤드. 추론에서 확신이 낮으면 잠재 스텝을 더 밟는 적응적 사고 | `sft --conf`, 대화 시 `--adaptive-latent 0.5` |
| **workspace** (GWT 작업공간) | 세션 내내 지속되며 모든 블록이 읽는(전역 방송) 소수의 작업공간 슬롯 — 턴 단위 latent를 넘어선 제한용량 워크스페이스 | `sft --workspace-slots 4`, 대화 시 `--workspace-file`로 세션 간 유지 |
| **attn_schema** (AST 주의도식) | 자기 어텐션 상태(레이어별 엔트로피)를 은닉에서 예측하는 보조 헤드 — "자기 주의의 단순 모델". 본체 logits는 불변(detach 절연) | `sft --attn-schema` |

### 런타임 층 (모델 밖 — 지속 상태 에이전트)

학습 없이 동작하는 층. drive는 **mood / latent / workspace 내부 채널로만**
사고에 스며들고, 출력 우회 게이팅은 하지 않는다. 정책(임계·쿨다운·DND)은
`runtime/config.json`에만 둔다 (`ModelConfig` 금지).

| 구성 | 무엇인가 | 켜는 법 |
|---|---|---|
| **drive** | curiosity/rest/social/maintenance — 벽시계·이벤트로 상승, discharge로 감쇠 | `--state-file` 사용 시 자동 |
| **state** | mood+workspace+drive+events 통합 저장/복원 | `chat --state-file session.pt` |
| **proactive** | 임계+쿨다운+DND 준수 후 먼저 말 걸기 (지표 증거 아님) | `chat --proactive` |

체크포인트가 자기 설정(`model_config`)을 내장하므로 `chat.py`는 자동으로
올바르게 동작한다. 핵심 실험 둘: **latent vs pause** — 연속적인 잠재 사고가
이산적인 필러 "속말"보다 나은지를 같은 연산량(k)에서 val loss로 비교,
그리고 **calibration** (`tools/eval_conf.py`) — 확신도 0.9 구간에서 실제로
~90%를 맞히면 기능적 메타인지가 성립한 것이다.
정확성은 `python -m tests.test_kv_loop`가 지킨다 (캐시 증분 ≡ 일괄 처리).
런타임 층은 `python -m tests.test_drive`.

> **권장 학습 경로**: 제일 비싼 사전학습은 `full`(base — loop off + compile on)로
> 빠르게 끝내고, 마음 기제(loop/mood/latent/workspace/attn_schema/...)는
> 30분~1시간짜리 SFT에서 붙인다. loop를 사전학습부터 넣고 싶으면 `full-loop`
> 프리셋. 인지과학 지표(Butlin et al.)와 각 기제·검증 도구의 대응은
> [`INDICATORS.md`](INDICATORS.md) 참고.

## 빠른 시작

### 0. 설치
```bash
pip install -r requirements.txt
```

### 1. 로컬에서 코드가 도는지 먼저 검증 (CPU, 몇 분)
tiny 프리셋은 성능이 아니라 "학습이 되긴 하는가"를 오버핏으로 확인하는 용도다.
```bash
# 소량 데이터로 파이프라인 전체 리허설
python -m data.download --max-docs 3000
python -m tokenizer.train_tokenizer --input data/raw/corpus.txt --vocab-size 4096 --sample-mb 20
python -m data.pack --input data/raw/corpus.txt --tokenizer tokenizer/tokenizer.json
python -m train.pretrain --preset tiny        # loss가 뚝뚝 떨어지면 정상
```

### 2. 진짜 학습은 클라우드에서 (무료 GPU)
`notebooks/train_cloud.ipynb`를 Kaggle(권장, 주 30h 무료) 또는 Colab에 올려
위에서부터 실행한다. 세션이 끊기면 `--resume`으로 이어서 학습된다.
- 사전학습(base): 토큰예산 1B(~4000스텝). fp16 자동·Muon·compile로 하룻밤~이틀.
  ```bash
  python -m data.download --source mix                 # 위키+나무위키 혼합
  python -m tokenizer.train_tokenizer --input data/raw/corpus.txt --sample-mb 200
  python -m data.pack --input data/raw/corpus.txt --tokenizer tokenizer/tokenizer.json
  python -m train.pretrain --preset full --optimizer muon --resume
  ```
- SFT(30분~1시간) — base에 마음 기제를 조합해 부여:
  ```bash
  python -m train.sft --init checkpoints/ckpt_best.pt --data data/bin/sft.npz \
      --mood-dim 64 --latent 2 --conf              # 조합 예시
  python -m train.sft --init checkpoints/ckpt_best.pt --data data/bin/sft.npz \
      --workspace-slots 4 --attn-schema            # 새 기제
  ```

### 3. 로컬에서 대화
클라우드에서 만든 `checkpoints/sft.pt`와 `tokenizer/tokenizer.json`을 내려받고:
```bash
python chat.py --ckpt checkpoints/sft.pt
# 워크스페이스를 세션 간 유지하며 슬롯 상태 관찰:
python chat.py --ckpt checkpoints/sft.pt --workspace-file ws.pt --show-workspace
# 통합 상태(mood+ws+drive) + proactive (쿨다운·DND 준수):
python chat.py --ckpt checkpoints/sft.pt --state-file session.pt --proactive --show-drive
# SFT 전에 "이어쓰기"만 시험해 보려면:
python chat.py --ckpt checkpoints/ckpt_best.pt --raw
```

> Windows 콘솔에서 한글이 깨지면 `set PYTHONUTF8=1` 후 실행.

## 검증 하네스 (철학 → 실행 eval)

각 마음 기제가 "그럴듯한 말"이 아니라 실제로 기능하는지를 수치로 확인하는
도구들. 대응 기제가 켜진 체크포인트가 필요하며, `--save`로 `eval_out/`에 JSON을
남긴다. 지표 매핑은 [`INDICATORS.md`](INDICATORS.md).

| 도구 | 무엇을 재나 | 대응 지표 |
|---|---|---|
| `tools/eval_conf.py` | 확신도 캘리브레이션(ECE·신뢰도 다이어그램) | HOT |
| `tools/eval_state.py` | mood/workspace 상태의 민감성·안정성·복귀·재현성 | 예측처리 |
| `tools/eval_intervention.py` | 개념축 Δ 주입의 인과성(α 단조성 vs 무작위) | GWT |
| `tools/eval_thinking.py` | latent vs pause vs k=0 (같은 검증셋 loss·정확도) | GWT |
| `tools/eval_integration.py` | 섭동 전파 반경으로 loop/feedback 통합 신호 | RPT |
| `tools/eval_schema.py` | attn_schema 예측 vs 실제 어텐션(MAE·상관) | AST |
| `tools/eval_loop.py` | n_loop별 val loss·속도 | RPT |
| `tests/test_drive.py` | drive 상승·discharge·상태복원·proactive 쿨다운/DND | agency(부분) |

```bash
python -m tools.eval_conf --ckpt checkpoints/sft_conf.pt --data data/bin/sft.npz --save
python -m tools.eval_thinking --data data/bin/sft.npz \
    --ckpts k0=checkpoints/sft.pt latent=checkpoints/sft_latent2.pt pause=checkpoints/sft_pause4.pt
python -m tests.test_drive
```

## 파이프라인 한눈에

```
위키·대화 데이터 ──(download)──> corpus.txt
   corpus.txt   ──(train_tokenizer)──> tokenizer.json   [토큰 사전]
   corpus.txt   ──(pack)──> train.bin / val.bin          [토큰 스트림]
   train.bin    ──(pretrain, 클라우드)──> ckpt_best.pt    [한국어의 결]
   ckpt_best    ──(sft, 클라우드)──> sft.pt               [대화하는 법]
   sft.pt       ──(chat.py, 로컬 CPU)──> 대화
```

## 어떻게 검증하나

- **구현이 옳은가**: tiny 프리셋으로 단일 배치를 오버핏시켜 loss가 0 근처로
  떨어지는지 본다. 떨어지면 forward/backward 배선이 정확한 것이다.
- **학습이 되는가**: val loss 곡선이 내려가고, 중간중간 생성 샘플이 점점
  한국어다워지는지 눈으로 확인한다.
- **쓸 만한가**: `chat.py`로 인사·일상 질문·간단한 코멘트 요청을 해 본다.
