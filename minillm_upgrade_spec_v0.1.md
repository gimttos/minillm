# minillm 업그레이드 핸드오프 스펙 v0.1

> Claude Code용 작업 지시서. 대상 레포: `gimttos/minillm` (밑바닥부터 만드는 한국어 초소형 대화 LLM, ~30M).
> 이 문서 하나로 모든 작업을 순서대로 수행할 수 있게 작성됨. 각 작업은 **목적 / 대상 파일 / 구현 / 완료 기준**을 갖는다.

---

## 0. 읽는 법 · 작업 원칙

- 작업은 워크스트림 A~F로 나뉜다. **§4의 게이트 순서**대로 진행한다.
- 각 작업 끝의 **완료 기준(DoD)**을 통과하지 못하면 다음으로 넘어가지 않는다.
- 모델/캐시 코드를 건드리는 모든 작업은 **반드시 `python -m tests.test_kv_loop` 통과**로 끝낸다. 새 기제를 추가하면 이 테스트에 케이스도 추가한다.
- 커밋은 작업 단위로 잘게. 커밋 메시지에 작업 ID(예: `A3: Muon+AdamW 하이브리드 옵티마이저`)를 넣는다.
- 한 작업이 기존 동작을 바꾸면, 바꾸기 전에 baseline 수치(val loss, ms/step)를 로그로 남기고 전후를 비교한다.

## 1. 배경 · 목표

현재 상태: 아키텍처·최적화(SDPA/RoPE/RMSNorm/SwiGLU/weight tying/KV캐시/grad accum/cosine)는 잘 되어 있고, 마음 유사 기제(loop·mood·latent·feedback·conf)도 항등 초기화·detach 절연까지 구현되어 있음. **문제는 코드 효율이 아니라 (1) 계획된 학습량이 가진 자원의 ~10배라 무료 Kaggle 쿼터가 먼저 끊기는 것, (2) 마음 기제를 이론적 지표에 맞춰 확장·검증할 체계가 없는 것.**

이번 업그레이드의 목표:
1. **학습을 한 자릿수 배 압축** — 몇 주짜리 → 하룻밤~이틀짜리.
2. **데이터를 나무위키로 확장** (말동무/서브컬처 결).
3. **마음 기제를 인지과학 지표(Butlin et al.)에 맞춰 확장** — 지속 워크스페이스(GWT), 주의 도식(AST).
4. **철학을 실행 가능한 eval로** — 개입 인과성·캘리브레이션·잠재 vs pause·통합·상태 지속성.

비목표(이번엔 안 함): 신체성(embodiment) 구현, RL 기반 행위성(agency), 모델 대형화, 웹 UI.

## 2. 절대 불변식 (어기면 안 되는 것)

기존 `CLAUDE.md` 규칙을 그대로 승계하고, 이번 스펙에서 강화한다:

1. **모든 신규 기능은 `ModelConfig` 플래그, 기본값 off.** 예전 체크포인트가 최신 코드에서 그대로 로드·동작해야 한다(`strict=False` 경로 보장).
2. **항등/제로 초기화 원칙.** 기능을 켠 직후 출력이 바뀌면 안 된다(FiLM 제로 init, `latent_proj` eye init, `feedback_proj` 제로 init과 동일한 패턴).
3. **체크포인트는 자기 설정(`model_config`)을 내장.** `chat.py`는 플래그 없이 자동으로 올바르게 동작.
4. **정확성 게이트:** 모델/캐시 코드를 건드리면 `python -m tests.test_kv_loop` 통과(캐시 증분 ≡ 일괄 처리, 접두부 캐시+직사각 마스크 ≡ 일괄).
5. **vocab 호환:** 기존 특수 토큰 ID(16380~16383)는 불변. 이번 신규 기제는 **새 토큰을 도입하지 않는다**(전부 벡터/헤드 수준). 따라서 `.bin` 재패킹 불필요.
6. **추론은 CPU fp32, 램 ~1GB 제약.** 매 토큰 비용을 곱하는 변경(loop, workspace)은 신중히, 턴당 한 번인 비용(mood, latent)은 관대하게.

## 3. 환경 제약

- **학습:** Kaggle 무료 GPU. 기본 T4(Turing, capability 7.5 → **하드웨어 bf16 없음**), 세션 12h·주 30h, `--resume` 필수.
- **추론:** VRAM 없는 노트북 CPU fp32.
- **총비용 ₩0 유지.**
- 네트워크로 받는 데이터/패키지: HuggingFace `datasets`, PyPI만.

## 4. 작업 개요 · 게이트 순서

```
게이트 1 (속도 기반) : A2 fp16자동  → A1 토큰예산 → A3 Muon → A4 base/SFT분리 → A5 프리페치 → A6 토크나이저샘플
게이트 2 (데이터)     : B1 나무위키 다운로더 → B2 혼합/재패킹 확인
게이트 3 (재베이스라인): 게이트1·2 반영해 base 사전학습 1회 완주 → 수치 기록
게이트 4 (마음 확장)  : C3 conf 상시화 → C1 워크스페이스 슬롯 → C2 주의 도식
게이트 5 (검증)       : D2(기존)→ D5 상태지속 → D1 개입 → D3 잠재vspause → D4 통합 → D6 도식정확도
게이트 6 (문서/런북)  : E 지표매핑 → F 런북·노트북·README 갱신
```

의존성: A3는 A2/A1 이후. C1·C2는 게이트3의 base 체크포인트가 있어야 SFT로 검증 가능. D 계열은 대응하는 C 기제가 켜진 체크포인트가 필요.

---

## 5. 워크스트림 A — 학습 속도 / 연산 압축

### A1. 토큰 예산 1/10 (max_steps 재설정)
**목적.** 30M 모델은 ~1B 토큰에서 포화 근처. 현재 `full`은 40,000스텝 × 245,760토큰 ≈ **9.8B 토큰**으로 10배 과다. 시간을 10배 낭비하는데 val loss 이득은 미미.
**대상.** `train/config.py`.
**구현.**
- `TrainConfig`에 `target_tokens: int = 1_000_000_000` 필드 추가. `max_steps`를 상수로 두지 말고 `target_tokens // (batch_size * grad_accum * max_seq_len)`로 유도하는 헬퍼를 `get_config` 안에서 계산해 세팅.
- `full` 프리셋: `target_tokens=1_000_000_000` → 유효배치 480×512 기준 약 4,000스텝. `warmup_steps`는 스텝의 5~8%(약 250)로. `min_lr`는 유지.
- CLI에서 `--target-tokens`로 덮어쓸 수 있게 `pretrain.py`에 인자 추가(스윕용).
**완료 기준.** `python -m train.pretrain --preset full`이 계산된 max_steps를 로그로 출력(예: `target 1.0B tokens -> 4069 steps`). tiny 프리셋은 영향 없음.

### A2. T4 fp16 자동 선택
**목적.** T4는 bf16 텐서코어가 없어 `autocast(bf16)`이 가속을 못 받음. fp16이 정답(GradScaler 경로는 이미 존재).
**대상.** `train/config.py`, `train/pretrain.py`, `train/sft.py`.
**구현.**
- 공용 헬퍼 `pick_amp_dtype(device)` 신설(예: `train/amputil.py` 또는 `config.py`): `torch.cuda.get_device_capability(0)[0] >= 8`이면 `"bfloat16"`, 아니면 `"float16"`. CPU면 `"float32"`.
- `config.py`의 `dtype` 하드코딩(`"bfloat16"`)을 이 헬퍼 결과로 대체. 단, 명시적 CLI 지정이 있으면 그 값을 우선.
- `pretrain.py`/`sft.py`는 이미 `dtype=="float16"`일 때만 GradScaler를 켜므로 로직 그대로 두되, dtype 출처만 헬퍼로.
**완료 기준.** T4에서 실행 시 로그에 `AMP dtype: float16 (T4, cap 7.5)` 류 출력. 동일 스텝에서 bf16 대비 ms/step이 감소.

### A3. Muon + AdamW 하이브리드 옵티마이저
**목적.** 목표 loss까지 스텝 수 ~1.3–2배 감소(스피드런 표준 옵티마이저).
**대상.** 신규 `train/muon.py`, 수정 `train/pretrain.py`(+선택 `train/sft.py`).
**구현.**
- `train/muon.py`: Newton–Schulz 직교화 기반 Muon 구현. 표준 5-스텝 NS(bf16 계산), momentum(β≈0.95), per-parameter 업데이트 후 RMS 스케일 보정. 공개 modded-nanoGPT 구현을 참조하되 **단일 GPU/비분산**으로 단순화(분산 all-gather 제거).
- **파라미터 그룹핑(중요):**
  - **Muon 대상 = 블록 내부의 2D 가중치 행렬:** `attn.wqkv.weight`, `attn.wo.weight`, `ffn.w_gate.weight`, `ffn.w_up.weight`, `ffn.w_down.weight`.
  - **AdamW 대상 = 나머지 전부:** `tok_emb.weight`(=tied `lm_head`), 모든 `RMSNorm.weight`, 모든 bias, 그리고 **마음 기제 헤드**(`mood_film`, `mood_read`, `latent_proj`, `feedback_proj`, `conf_head`, 그리고 C1/C2에서 추가될 `workspace_*`, `attn_schema_*`). 헤드는 항등/제로 init 동역학을 지켜야 하므로 Muon에 넣지 않는다.
  - 판별은 이름 기반 화이트리스트로(정규식 하드코딩 금지, `named_parameters()` 순회 + 접미사 매칭). 어디에도 안 걸리는 2D 파라미터가 있으면 경고 로그.
- **LR:** Muon LR과 AdamW LR을 분리. 시작값 제안 — Muon `lr≈0.02`(warmup+cosine 동일 스케줄), AdamW(embeddings 등) `lr≈cfg.learning_rate`. 둘 다 `lr_at()` 곱해서 스케줄. weight_decay는 Muon 대상엔 0.1, embedding엔 0 권장.
- **resume 호환:** 체크포인트 `optim` 저장/로드가 두 옵티마이저를 모두 담게(`{"muon":..., "adamw":...}`). **옵티마이저 종류가 바뀐 예전 체크포인트를 resume하면** state 불일치 → 모델 가중치만 로드하고 옵티마이저는 새로 시작(경고 로그). `--resume` 실패로 죽지 말 것.
**완료 기준.** (1) `python -m tests.test_kv_loop` 통과(모델 불변). (2) tiny 오버핏 테스트에서 AdamW 대비 같은 스텝 수에 loss가 같거나 낮음. (3) 동일 목표 val loss 도달 스텝 수가 AdamW baseline보다 감소(A1 재베이스라인에서 확인).

### A4. 사전학습은 loop off + compile on, 마음 기제는 SFT로
**목적.** `full`이 확률적 n_loop 때문에 `torch.compile`을 끄고 loop 1.5배 연산까지 지고 감. 제일 비싼 사전학습 단계에서 이 둘을 제거.
**대상.** `train/config.py`.
**구현.**
- `full` 프리셋: `loop_start=0, loop_end=0, n_loop=1`(loop off), `compile=True`, batch/accum은 loop 없는 메모리에 맞춰 유효배치 480 유지(예: `batch_size=24, grad_accum=20`).
- loop를 사전학습에 넣고 싶을 때를 위해 별도 프리셋 `full-loop`를 남기되(현재 `full` 설정 이관), **기본 권장 경로는 `full`(base) → SFT에서 loop/mood/latent 부여**임을 config 주석과 README에 명시.
**완료 기준.** `full`이 compile 켜진 채로 돈다(로그에 컴파일 소요 출력). ms/step이 loop 버전보다 감소.

### A5. 데이터 로더 프리페치 (더블버퍼)
**목적.** 모델이 작고 빨라 동기식 `get_batch`(파이썬 루프+`np.stack`+pin+전송)가 GPU를 놀림.
**대상.** `train/pretrain.py`.
**구현.**
- 백그라운드 스레드 1개 + `queue.Queue(maxsize=2)` 프리페처. 다음 배치를 CPU에서 미리 만들어(넘파이 스택 + `pin_memory`) 큐에 넣고, 메인 루프는 `.get()` 후 `to(device, non_blocking=True)`.
- CUDA 스트림까지는 불필요(과설계 금지). 스레드 종료·예외 전파 처리.
- grad_accum 마이크로배치도 프리페처에서 공급.
**완료 기준.** 동일 설정에서 ms/step 감소, val loss 곡선은 프리페치 전과 통계적으로 동일(무작위성만 차이). GPU 활용률(nvidia-smi) 상승 확인.

### A6. 토크나이저 샘플 편향 수정
**목적.** `train_tokenizer`가 파일 **앞부분** `sample-mb`만 읽어 사전이 앞쪽 문서에 편향됨.
**대상.** `tokenizer/train_tokenizer.py`.
**구현.**
- 파일 전체 크기를 구한 뒤, 코퍼스 전역에서 랜덤 오프셋으로 여러 청크를 읽어 합쳐 `sample-mb`를 채운다(각 청크는 `DOC_SEP` 경계로 정렬). 시드 고정.
- 순수 파이썬 BPE 학습이 느린 건 1회성이라 감수. (선택) 학습 시간 로그 유지.
**완료 기준.** 같은 `--sample-mb`로 뽑은 사전이 코퍼스 뒷부분 문장도 합리적으로 분절(데모 문장 외에 나무위키 뒷부분 표제어 한두 개로 육안 확인).

---

## 6. 워크스트림 B — 데이터 (나무위키)

### B1. 나무위키 다운로더 추가 + 위키 혼합
**목적.** 말동무/서브컬처 결. `heegyu/namuwiki-extracted`는 나무마크가 정리된 판(원본 `heegyu/namuwiki`는 `[[...]]`,`{{{...}}}`,`== ==` 마크업이 그대로라 노이즈). 라이선스 cc-by-nc-sa-2.0(비상업) — 개인 프로젝트 OK.
**대상.** `data/download.py`.
**구현.**
- `--source {wiki,namu,mix}` 인자 추가. `namu`는 `load_dataset("heegyu/namuwiki-extracted", split="train", streaming=True)`의 `text` 필드 사용.
- `mix`는 두 소스를 문서 단위로 교차 스트리밍(예: 위키:나무 = 1:1, `--mix-ratio`로 조절). 각 문서 뒤에 기존 `DOC_SEP` 유지(pack.py 계약 불변).
- 공통 정제: `text.strip()`, 200자 미만 스킵은 유지. 나무위키 잔여 잡음(연속 공백/특수문자/남은 대괄호 링크 등) 가벼운 정규화 함수 추가(과하게 지우지 말 것 — 구어체 보존).
- 라이선스 고지: 코퍼스 저장 시 `data/raw/DATASOURCES.txt`에 사용 데이터셋·라이선스·덤프일자 기록.
**완료 기준.** `python -m data.download --source mix --max-docs 5000`이 위키/나무 혼합 corpus.txt를 생성, `DATASOURCES.txt` 생성. 육안으로 나무마크 잔재가 거의 없음.

### B2. 혼합 코퍼스 재패킹 검증
**목적.** pack/tokenizer 파이프라인이 혼합 코퍼스에서 그대로 도는지 확인.
**대상.** 없음(기존 `data/pack.py`·`tokenizer/train_tokenizer.py` 재사용).
**구현.** 게이트 순서상: `download(mix)` → `train_tokenizer`(A6 반영) → `pack`. vocab은 16392 유지, 특수 토큰 불변이므로 신규 기제와 무관.
**완료 기준.** `data/bin/train.bin`,`val.bin` 생성, `tok.vocab_size <= 65536` assert 통과, val 토큰 수 > 0.

---

## 7. 워크스트림 C — 새 마음 기제

> 전부 §2 불변식 준수: `ModelConfig` 플래그 기본 off, 항등/제로 init, 체크포인트 자기설정, `test_kv_loop` 케이스 추가, 새 토큰 없음.

### C3. 확신도(conf) 상시화 + 캘리브레이션 로깅 (먼저, 가벼움)
**목적.** conf_head(HOT 지표)는 이미 있음. 학습 중 캘리브레이션을 1급 지표로 승격.
**대상.** `train/sft.py`.
**구현.** `--conf`로 학습 시, `eval_interval`마다 검증셋에서 ECE를 간단 계산해 로그(정식 리포트는 D2/`eval_conf.py`). 새 파라미터 없음.
**완료 기준.** SFT 로그에 주기적 `ECE=...` 출력.

### C1. 지속 워크스페이스 슬롯 (GWT)
**목적.** 턴 단위 latent를 넘어, **세션 내내 지속되며 블록들이 읽고(broadcast) 쓰는 소수의 워크스페이스 벡터.** GWT의 제한용량 작업공간·전역 방송에 가장 근접.
**대상.** `model/gpt.py`, `chat.py`, `tests/test_kv_loop.py`.
**구현.**
- `ModelConfig`: `workspace_slots: int = 0`(0=off), 선택 `workspace_dim`(기본 d_model).
- `GPT.__init__`: `workspace_slots>0`이면 (a) 슬롯 상태 텐서 규약 정의(세션 지속, `mood`처럼 forward 인자로 흘림 — 파라미터가 아니라 상태), (b) **쓰기 헤드** `ws_write: Linear(d_model→slots*dim)`와 **읽기 주입** `ws_read: Linear(slots*dim→d_model)` 또는 블록이 슬롯을 key/value로 크로스어텐션하는 방식 중 택1. **제로 init**로 켠 직후 항등.
- 방송 지점: 각 Block이 어텐션 입력에 워크스페이스를 크로스어텐션 또는 FiLM류로 주입(mood와 동일한 "residual 밖 조건화"라 상태가 이상해도 안 망가지게).
- 갱신: 턴 종료 시 은닉 요약으로 슬롯을 EMA 갱신(mood의 `update_mood`와 대칭). tanh 유계.
- **KV 캐시 정합성(최대 리스크):** 병렬(학습) 경로와 캐시 증분(생성) 경로에서 슬롯 주입 결과가 **비트 수준으로 동일**해야 함. 슬롯은 턴 내 고정(토큰마다 안 바뀜)으로 설계해 정합성을 단순화. 
- `chat.py`: `--workspace-file`로 세션 간 저장/복원(mood-file과 대칭), `--show-workspace`로 슬롯 노름/상위 성분 출력.
**완료 기준.** (1) `test_kv_loop`에 workspace 케이스 추가 후 캐시 증분 ≡ 일괄, 제로 init 항등 통과. (2) 켠 채 SFT 후 val loss가 off 대비 악화되지 않음. (3) `chat.py --show-workspace`로 슬롯이 턴에 따라 변함을 육안 확인.

### C2. 주의 도식 헤드 (AST)
**목적.** 시스템이 **자기 어텐션 상태의 단순 모델**을 갖게. Graziano 주의 도식 이론 대응 + 검증 가능.
**대상.** `model/gpt.py`, `tests/test_kv_loop.py`, (검증은 D6).
**구현.**
- `ModelConfig`: `attn_schema: bool = False`.
- 헤드 `attn_schema_head: Linear(d_model → K)`. K는 "어텐션 상태 요약" 차원 — 제안: **레이어별/헤드별 어텐션 엔트로피** 또는 **최근 vs 원거리 어텐션 질량 비**같은 저차원 요약(K = n_layers 또는 n_layers*n_heads의 축소).
- **타깃 문제:** SDPA(fused)는 어텐션 가중치를 반환하지 않음. 해결: 학습 시 **보조 손실 전용**으로, 서브샘플한 위치에 한해 `softmax(QK^T/√d)`를 수동 계산(no_grad)해 요약 타깃을 만들고, 헤드가 은닉에서 그 타깃을 회귀(detach된 타깃, conf_head와 동형 절연). 전체가 아니라 배치의 일부 위치만 → 비용 통제.
- loss에 `+ λ·schema_loss`(λ 작게, 예 0.05). **본체 logits 불변**(헤드는 은닉을 읽기만).
**완료 기준.** (1) `test_kv_loop`: attn_schema 켜도 logits 불변(detach 절연) 케이스 추가·통과. (2) 학습 로그에 `schema_loss` 출력. (3) D6에서 도식이 실제 어텐션 요약을 유의하게 예측.

> **문서화만:** 행위성(agency)·신체성(embodiment)은 이번 비목표. `INDICATORS.md`(§9)에 "미충족/근거"로 명시.

---

## 8. 워크스트림 D — 검증 하네스 (철학 → 실행 eval)

> 공통: 결과는 표/수치로 stdout, 옵션으로 `eval_out/`에 JSON 저장. 재현 위해 시드 고정. 각 도구는 대응 기제가 꺼진 체크포인트엔 친절히 에러.

### D2. 캘리브레이션 리포트 (기존 — 확장)
**상태.** `tools/eval_conf.py` 이미 존재(구간별 정답률 + ECE).
**대상.** `tools/eval_conf.py`.
**구현.** 신뢰도 다이어그램을 텍스트 막대 또는 `eval_out/reliability.json`으로 저장하는 옵션 `--save` 추가. 나머지 유지.
**완료 기준.** `--save` 시 JSON 생성, 기존 표 출력 불변.

### D5. 상태 지속 일관성 (mood/workspace) — 원래 문제의 정답
**목적.** "왜 갈아끼우면 유지가 안 됐나"를 수치화. 같은 입력 + 다른 상태 → **다르지만 일관된** 출력, 그리고 기준선 복귀.
**대상.** 신규 `tools/eval_state.py`.
**구현.**
- 고정 프롬프트 집합에 대해: (a) mood/workspace를 여러 초기값으로 주입해 생성 → 출력 분포 차이 측정(토큰 분포 KL, 생성 임베딩 거리). (b) 중립 대화를 N턴 흘려 상태가 기준선으로 감쇠·복귀하는지 궤적(`mood_trajectory` 재사용) 분석. (c) **재현성:** 같은 상태 → 같은(greedy) 출력.
**완료 기준.** 상태별 출력이 유의하게 갈리되(민감성) 발산하지 않고(안정성) 복귀함을 보이는 리포트. 발산/붕괴 시 경고.

### D1. 개입 인과성 (activation patching, "거미→개미")
**목적.** 내부 상태가 상관이 아니라 **인과적으로** 출력을 지배하는지 — J-space 핵심 실험의 자가 재현.
**대상.** 신규 `tools/eval_intervention.py`, 필요 시 `model/gpt.py`에 훅 유틸.
**구현.**
- 특정 레이어 L·위치의 은닉을 **캡처/덮어쓰기**하는 헬퍼(forward hook 또는 `run_from_pos` 분해).
- 개념 방향: 텍스트 집합 A(예: 거미)와 B(개미)의 평균 은닉 차 `Δ = mean_h(A) − mean_h(B)`. 프롬프트 은닉에 `α·Δ` 주입 후 목표 토큰(예: 다리 수 "8"↔"6") 로짓/출력 변화 측정.
- workspace 슬롯·latent 상태에도 같은 개입 적용(그 공간에서 사고하는지).
**완료 기준.** 개입 강도 α에 따라 출력/목표 로짓이 **단조·인과적으로** 변하는 곡선. 무작위 방향 개입(대조군) 대비 효과가 큼.

### D3. 잠재 vs pause (같은 연산량 k)
**목적.** 연속 잠재사고가 이산 필러 속말보다 나은지 — 출력으로 환원 안 되는 내부 연산의 존재 검증(프로젝트 대표 실험).
**대상.** 신규 `tools/eval_thinking.py`(또는 `eval_loop.py` 확장).
**구현.** 동일 base에서 `--latent k` SFT 모델과 `--n-pause k` SFT 모델을 같은 k로 비교: val loss + 간단 QA/산술 정확도. 공정 위해 같은 검증셋·시드.
**완료 기준.** k별 비교표(latent vs pause vs k=0). 어느 쪽이 이기든 **수치로** 결론.

### D4. 통합/재귀 프로브 (RPT/IIT 근사)
**목적.** loop·feedback이 만드는 정보 통합의 방향성 신호(진짜 Φ 아님, 명시).
**대상.** 신규 `tools/eval_integration.py`.
**구현.** 한 유닛/헤드 활성을 섭동하고, 그 변화가 후속 위치/레이어로 **얼마나 멀리 전파**되는지 측정(출력 로짓 변화의 공간적 확산). loop off/on, feedback off/on 비교. 순전파-only는 국소, 재귀는 전역이어야.
**완료 기준.** loop/feedback 켠 모델의 전파 반경이 끈 모델보다 큼을 보이는 리포트.

### D6. 주의 도식 정확도 (AST 검증)
**목적.** C2의 자기 어텐션 모델이 실제 어텐션을 예측하나.
**대상.** 신규 `tools/eval_schema.py`.
**구현.** 검증셋에서 수동 계산한 실제 어텐션 요약 vs 도식 헤드 예측의 상관/오차. 무작위/영 예측 baseline 대비.
**완료 기준.** 도식 예측이 baseline보다 유의하게 정확(상관·MAE 리포트).

---

## 9. 워크스트림 E — Butlin 지표 매핑 문서

### E1. `INDICATORS.md` 신설
**목적.** "마음 유사체에 뭐가 얼마나 필요한가 / 무엇으로 확인하나"의 개념 백본. Butlin·Long 외 「Consciousness in Artificial Intelligence」(arXiv 2308.08708; 정식판: Trends in Cognitive Sciences 2025, "Identifying indicators of consciousness in AI systems")의 지표 속성을 레포 기제·검증 테스트에 1:1 대응.
**대상.** 신규 `INDICATORS.md`.
**구현.** 표 형식:

| 이론 | 지표 속성(요지) | minillm 기제 | 검증 도구 | 상태 |
|---|---|---|---|---|
| 재귀처리(RPT) | 순환·되먹임 | loop, feedback | D4 | 구현/검증 |
| 전역작업공간(GWT) | 제한용량 작업공간·전역 방송·점화 | latent, **workspace(C1)** | D1 | 구현/검증 |
| 고차이론(HOT) | 자기 상태의 메타표상·보고 | conf_head | D2 | 구현/검증 |
| 예측처리 | 예측오차 최소화·느린 prior | 다음토큰예측, mood | D5 | 구현 |
| 주의도식(AST) | 자기 주의의 모델 | **attn_schema(C2)** | D6 | 구현/검증 |
| 행위성 | 피드백 학습·유연한 목표추구 | (미구현) | — | 미충족 |
| 신체성 | 출력-입력 수반성 모델 | feedback(부분/최소) | — | 부분 |

- 상단에 **"얼마나"의 정직한 답**: 정해진 임계치 없음. 현재 최선은 임계선이 아니라 **지표 만족도가 높을수록 신뢰도가 높아지는 확률적 체크리스트**이며, 프레임 전체가 **계산 기능주의**(맞는 계산을 구현하면 그 속성이 생긴다, 기질 독립) 위에 섬. 목표는 현상적 의식이 아니라 **기능적 접근 의식** 지표.
- 하단에 **경계**: 현상적 의식/퀄리아(하드 프로블럼·타심 문제)는 어떤 테스트로도 결판나지 않음 — 공학 실패가 아니라 인식론 경계. "모델이 느낀다고 말하는 것"은 최약 증거(ELIZA 효과) → 개입 인과성·캘리브레이션처럼 **말로 못 꾸며내는 행동**만 증거로.
**완료 기준.** 표의 모든 "구현/검증" 행이 실제 코드·도구와 링크로 연결(파일 경로 명시). 논문 서지 포함.

---

## 10. 워크스트림 F — 오케스트레이션 / 런북 / 문서

### F1. Kaggle 학습 노트북 갱신
**대상.** `notebooks/train_cloud.ipynb`.
**구현.** 게이트3 파이프라인을 셀 순서로: 설치 → `download --source mix` → `train_tokenizer`(랜덤샘플) → `pack` → `pretrain --preset full`(fp16 자동·Muon·compile·resume) → 체크포인트 저장. **resume 규율**(세션 끊김 대비 `save_interval` 조밀), **쿼터 예산 계산 셀**(target_tokens ÷ 관측 tok/s = 예상 시간) 포함.

### F2. SFT/추론 런북 갱신
**대상.** `README.md`.
**구현.** base → SFT 순서 재정리(§A4 반영): `sft --init ckpt_best.pt`에 mood/latent/workspace/attn_schema/conf 조합 예시. `chat.py` 옵션(신규 `--workspace-file`, `--show-workspace`) 문서화. 검증 도구(D1~D6) 사용법 섹션 신설.

### F3. CLAUDE.md 갱신
**대상.** `CLAUDE.md`.
**구현.** 신규 기제(workspace, attn_schema)와 검증 도구를 "프로젝트가 지향하는 것"·아키텍처 규칙에 반영. 게이트 순서·불변식(§2·§4) 요약 추가.

---

## 11. 전역 완료 기준 (Definition of Done)

1. `python -m tests.test_kv_loop` — 신규 기제(workspace, attn_schema) 케이스 포함 **전부 통과**.
2. 게이트3 재베이스라인: fp16+Muon+토큰예산1B+loop off+compile로 base 사전학습 **1회 완주**, ms/step·최종 val loss·총 소요시간을 이전과 비교한 표를 `BASELINE.md`에 기록. 목표: 총 시간이 이전 대비 한 자릿수 배 단축.
3. 데이터: 혼합 코퍼스로 `train.bin/val.bin` 생성, `DATASOURCES.txt` 존재.
4. 마음 기제: workspace·attn_schema가 켠 채 SFT되고 off 대비 val loss 비악화, 각 대응 검증 도구(D1/D6)가 유의 결과.
5. 검증 하네스 D1~D6 **전부 실행 가능**하고 리포트 산출.
6. 문서 `INDICATORS.md`·`BASELINE.md`·갱신된 `README.md`/`CLAUDE.md`/노트북 존재.
7. 모든 신규 기능 플래그 기본 off, 예전 체크포인트 `strict=False` 로드·동작 확인.

## 12. 참고 · 데이터

- **데이터셋:** `heegyu/namuwiki-extracted`(정제판), `wikimedia/wikipedia 20231101.ko`. 라이선스 각각 cc-by-nc-sa-2.0 / CC BY-SA — **비상업 개인용**. SFT: 기존 `beomi/KoAlpaca-v1.1a`, `songys/Chatbot_data` 유지.
- **최적화:** Muon(Newton–Schulz 직교화). 참조: modded-nanoGPT(KellerJordan). 단일 GPU로 단순화.
- **이론 프레임워크:** Butlin, Long, et al., "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness", arXiv:2308.08708 (2023) / Trends in Cognitive Sciences (2025). RPT·GWT·HOT·예측처리·AST에서 지표 속성 도출, 계산 기능주의 전제, "현행 AI는 의식적이지 않으나 명백한 기술적 장벽도 없다".

---

### 부록: 작업 ID 체크리스트
- [ ] A1 토큰예산  · [ ] A2 fp16자동 · [ ] A3 Muon · [ ] A4 base/SFT분리 · [ ] A5 프리페치 · [ ] A6 토크나이저샘플
- [ ] B1 나무위키 · [ ] B2 재패킹검증
- [ ] C3 conf상시 · [ ] C1 워크스페이스 · [ ] C2 주의도식
- [ ] D2 캘리브 · [ ] D5 상태지속 · [ ] D1 개입 · [ ] D3 잠재vspause · [ ] D4 통합 · [ ] D6 도식정확도
- [ ] E1 INDICATORS · [ ] F1 노트북 · [ ] F2 README · [ ] F3 CLAUDE.md
- [ ] 게이트3 재베이스라인 BASELINE.md · [ ] 전역 DoD
