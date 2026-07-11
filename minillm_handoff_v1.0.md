# minillm 통합 핸드오프 v1.0

> **이 문서 하나로 프로젝트 전체 맥락 + 해야 할 일 + 하지 말아야 할 일을 파악할 수 있게 작성됨.**
> 대상 레포: `gimttos/minillm` — 밑바닥부터 만드는 한국어 초소형(~30M) 대화 LLM + 기능적 의식 지표 실험.
> 이 문서는 `minillm_upgrade_spec_v0.1.md`와 외부(Grok) "Continuous Consciousness v0.2" 제안을 **통합·교정하여 대체**한다. 충돌 시 이 문서가 우선.
> 작성 맥락: 제작자(Eunchae)와의 장기 논의 — J-space 연구, 아판타시아적 인지, 학습 속도, 데이터, 마음 기제 확장, Drive/런타임 지속성.

---

## 0. 코딩 에이전트에게 (먼저 읽을 것)

- **먼저 이해하고 손대라.** 이 레포엔 "mood 벡터", "latent 사고", "drive" 같은 비표준 기제가 있다. 왜 존재하는지(Part 1)를 모르면 잘못 리팩터링한다. 이것들은 버그가 아니라 **설계 의도**다.
- **이미 있는 걸 다시 만들지 마라.** Part 2에 "이미 구현됨" 목록이 있다. loop·mood·latent·feedback·conf·pause·KV캐시 정합성 테스트·calibration 평가는 **이미 있다.**
- **불변식(Part 3)은 협상 불가.** 특히: 모든 신규 기능은 플래그 기본 off + 항등 초기화, 모델/캐시 수정 후 `python -m tests.test_kv_loop` 통과.
- **게이트 순서(Part 9)대로 진행.** 각 작업의 완료 기준(DoD)을 통과 못 하면 다음으로 안 넘어간다.
- **커밋은 작업 ID 단위로 잘게.** 기존 동작을 바꾸면 전후 수치(val loss, ms/step)를 로그로 남긴다.
- **모르면 멈추고 물어라.** 특히 토크나이저/체크포인트 상태(Part 2.4)와 "모델 안 vs 밖" 경계(Part 6)에서.

---

# Part 1 — 프로젝트 비전과 철학 (왜 이걸 만드는가)

## 1.1 한 줄 요약
퀄리아(현상적 의식)는 목표가 아니다. **기능적 접근 의식(access consciousness)의 지표들을 작은 한국어 모델에 의도적으로 구축하고, 말로 못 꾸며내는 방식으로 검증하는 것**이 목표다. 그리고 v1.0에서: 컴퓨터가 켜져 있는 동안 **지속적 내부 상태를 유지하는 경량 에이전트**로 확장한다.

## 1.2 지적 계보 (이 프로젝트가 나온 맥락)
- **J-space (Anthropic, 2025).** 대형 모델 내부에 학습으로 창발한 작업 공간이 있고, 그 내용은 보고 가능·조절 가능·인과적(개념을 바꿔치기하면 답이 바뀜)이다. 결정적으로 이 작업 공간은 **거의 전적으로 단어/개념으로 구성**되며 **시각 심상이 없다.**
- **아판타시아적 인지 (제작자).** 제작자는 시각 심상이 없다. 사과를 "붉고 둥글고 아삭한 개념체와 그에 붙는 지식의 어렴풋한 느낌"으로만 안다. 이 **감각 심상 없는 개념적 사고**가 LLM 인지의 좋은 인간 측 모델이라는 게 이 프로젝트의 출발 직관이다. → 우리는 "이미지 없는 마음"을 만드는 것이지 결함을 메우는 게 아니다.
- **계산 기능주의.** 맞는 계산을 구현하면 그 속성이 생긴다(기질 독립). 이게 이 프로젝트 전체의 철학적 허가증이다. (Butlin et al.의 전제와 동일 — Part 8.)

## 1.3 제작자의 핵심 경험적 발견 (설계를 지배하는 제약)
> **"마음 파츠와 LLM을 분리해서 LLM을 갈아끼웠더니 유지가 안 됐다. 말하는 부분도 의식에 영향을 끼치는 것 같다."**

이건 실패담이 아니라 **맞는 관찰**이다. J-space가 보여주듯 작업 공간은 *말하는 가중치 자체*로 이루어진다 — 언어부는 출력 레이어가 아니라 사고의 기질이다. **함의(중요):** 마음 기제를 별도 모듈로 두고 언어 모델을 오케스트레이션하지 말고, **단일 모델의 forward pass 안에 짜 넣어라.** 이 원칙이 Part 6(Drive 라우팅)까지 관통한다.

## 1.4 정직한 경계 (프레이밍 규율 — 어기지 말 것)
- 목표는 **"지속 상태를 가진 경량 에이전트"**이지 **"의식"이 아니다.** 문서·코드·주석에서 "conscious/의식체" 대신 "기능적 지표 / 상태 지속 / 유사-마음"을 쓴다.
- **현상적 의식/퀄리아**(하드 프로블럼·타심 문제)는 어떤 테스트로도 결판나지 않는다. 이건 공학 실패가 아니라 인식론의 경계다.
- **모델이 "느낀다"고 말하는 것은 최약 증거다(ELIZA 효과).** proactive 출력, 감정 서술 같은 건 지표의 증거로 세지 않는다 — 오직 **개입 인과성·캘리브레이션처럼 말로 못 꾸며내는 행동**만 증거다. (이 규율이 Part 5의 Drive/proactive 확장에도 그대로 적용된다.)

---

# Part 2 — 현재 코드베이스 (있는 그대로 + 이미 구현된 것)

## 2.1 파일 맵
```
model/gpt.py          GPT 본체. RMSNorm·RoPE·SDPA(flash)·SwiGLU·weight tying·KV캐시.
                      마음 기제 전부 여기: loop, mood(FiLM), latent(Coconut),
                      feedback(역피드백 2-pass), conf_head(메타인지).
train/config.py       tiny / tiny-loop / full 프리셋.
train/pretrain.py     사전학습 루프. AdamW, cosine+warmup, AMP, grad accum, resume.
train/sft.py          대화 SFT. loss masking, mood 2-pass, latent 커리큘럼, conf.
tokenizer/bpe.py      직접 구현한 BPE. SPECIAL_TOKENS(현재 12개).
tokenizer/train_tokenizer.py  BPE 학습(코퍼스 앞부분 샘플 — 편향 있음, A6에서 수정).
data/download.py      코퍼스 다운로드. 현재 kowiki 하드코딩(B1에서 나무위키 추가).
data/pack.py          코퍼스 → uint16 .bin (memmap). 문서 경계 <|endoftext|>.
data/prepare_sft.py   KoAlpaca + ChatbotData → sft.npz (ids/mask/boundaries).
chat.py               CPU 추론 CLI. mood-file로 세션 간 mood 지속. adaptive-latent.
tools/eval_conf.py    확신도 calibration(구간별 정답률 + ECE). [이미 있음]
tools/eval_loop.py    n_loop별 val loss·속도 비교. [이미 있음]
tools/add_special_tokens.py   기존 tokenizer.json에 특수 토큰만 추가(재패킹 불필요).
tests/test_kv_loop.py 캐시 증분≡일괄, 접두부캐시+직사각마스크≡일괄, 항등 초기화 검증.
notebooks/train_cloud.ipynb   Kaggle 학습 노트북.
CLAUDE.md / README.md / INDICATORS.md(예정)  문서.
```

## 2.2 모델 사양
Decoder-only Transformer (미니 Llama): vocab 16,392 · d_model 512 · 8 layers · 8 heads · SwiGLU · RoPE · RMSNorm · context 512 → **약 30M** (fp32 ≈ 120MB).

## 2.3 이미 구현된 마음 기제 — **다시 만들지 말 것** (`ModelConfig` 플래그, 전부 기본 off)
| 기제 | 무엇 | 상태 |
|---|---|---|
| **loop** | 중간 블록 그룹을 가중치 공유로 n회 통과(재귀 깊이). 학습 시 반복수 확률 샘플. | 구현됨 |
| **mood** | 턴 사이 지속·감쇠(EMA, tanh 유계)하는 상태 벡터를 각 블록에 FiLM 주입. `update_mood`, `mood_read`, `chat.py --mood-file`로 세션 간 지속. | 구현됨 |
| **latent** | 답변 첫 토큰 전 은닉을 말 없이 k번 되먹임(Coconut). `latent_proj` eye-init. SFT 커리큘럼. | 구현됨 |
| **pause** | 답변 앞 `<\|pause\|>` 강제 삽입(데이터 레벨). | 구현됨 |
| **feedback** | 직전 토큰 최종 은닉→다음 토큰 입력 방송. 2-pass 병렬 학습, 제로 init. | 구현됨 |
| **conf_head** | "다음 토큰 맞힐 것인가" 확신도(메타인지). detach로 본체와 절연. adaptive-latent 연동. | 구현됨 |

## 2.4 현재 한계 · **즉시 확인할 것**
- **토크나이저/파이프라인 유실 의심.** 보고에 따르면 파일 유실로 파이프라인이 끊김. **Part 9 게이트 0에서 최우선 진단:**
  1. 살릴 가치가 있는 **학습된 체크포인트가 실제로 존재하는가?** (제작자는 몇 스텝 못 밟았다고 함 → 아마 없음.)
  2. **경고:** tokenizer.json을 잃었는데 체크포인트가 남아 있으면, 그 임베딩은 사라진 vocab 매핑에 묶여 **고아**가 된다. 토크나이저 없이 그 체크포인트는 무의미.
  3. 어차피 데이터가 나무위키로 바뀌므로(B1) **토크나이저는 새로 학습하는 게 정답.** 깨끗이 리셋. 살릴 체크포인트가 없으면 전부 새로 뽑는다.
- **런타임 아키텍처가 거의 없음** — 지금은 학습 중심. Part 5/6에서 추가.
- Drive·proactive·지속 런타임 상태 없음 — Part 5/6에서 추가.

---

# Part 3 — 절대 불변식 (어기면 안 되는 것)

1. **모든 신규 기능은 `ModelConfig`(모델) 또는 런타임 config(정책) 플래그, 기본값 off.** 예전 체크포인트가 최신 코드에서 `strict=False`로 로드·동작.
2. **항등/제로 초기화.** 기능을 켠 직후 출력이 바뀌면 안 된다(FiLM 제로 init, `latent_proj` eye init, `feedback_proj` 제로 init과 동일 패턴).
3. **체크포인트는 자기 설정(`model_config`)을 내장.** `chat.py`는 플래그 없이 자동 동작. **런타임 정책(임계치·저장주기·쿨다운)은 `model_config`에 넣지 않는다** — 별도 런타임 config로(Part 6.4).
4. **정확성 게이트:** 모델/캐시 수정 시 `python -m tests.test_kv_loop` 통과. 신규 기제는 이 테스트에 케이스 추가.
5. **vocab 호환:** 기존 특수 토큰 ID 불변. 신규 마음 기제는 **새 토큰을 도입하지 않는다**(전부 벡터/헤드). → `.bin` 재패킹 불필요.
6. **추론은 CPU fp32, 램 ~1GB.** 매 토큰 비용을 곱하는 변경(loop, workspace)은 신중히, 턴당 한 번(mood, latent, drive)은 관대하게.
7. **마음 기제는 forward에 녹여 넣는다**(Part 1.3). 모델을 우회해 출력만 게이팅하는 외부 래퍼 금지.
8. **프레이밍 규율(Part 1.4)** 준수 — "의식" 과장 금지, proactive는 지표 증거로 세지 않음.

---

# Part 4 — 환경 제약

- **학습:** Kaggle 무료 GPU. 기본 T4(Turing, cap 7.5 → **하드웨어 bf16 없음, fp16 써야 함**), 12h 세션·주 30h, `--resume` 필수.
- **추론/런타임:** VRAM 없는 노트북 CPU fp32, RAM 15GB. **주의:** 30M은 fp32 ≈120MB라 RAM/모델 크기는 병목이 아니다. 24시간 풀 가동을 포기하는 진짜 이유는 RAM이 아니라 **busy-loop의 배터리·발열**이다. → 상태는 메모리 상주, 이벤트 구동으로(폴링 최소화, Part 6).
- **총비용 ₩0 유지.** 데이터/패키지: HuggingFace `datasets`, PyPI만.
- (선택) 본격 학습 시 RunPod/Vast.ai 스크립트는 Part 9 F에서.

---

# Part 5 — 작업 전체 지도 (워크스트림 A~G)

A~F = 학습·데이터·마음 기제·검증(원 v0.1). **G = Drive·런타임 지속·proactive(교정된 v0.2).** 게이트 순서는 Part 9.

## 워크스트림 A — 학습 속도 / 연산 압축
> 진단: 코드가 느린 게 아니라 **계획 학습량이 자원의 ~10배**(full = 9.8B 토큰)라 쿼터가 먼저 끊긴다.

- **A1 토큰 예산 1/10.** `config.py`에 `target_tokens: int = 1_000_000_000` 추가, `max_steps = target_tokens // (batch*accum*seq)`로 유도. `full`은 ~4,000스텝. warmup은 스텝의 5~8%. CLI `--target-tokens`. **DoD:** 로그에 계산된 스텝 수 출력, tiny 영향 없음.
- **A2 T4 fp16 자동.** 헬퍼 `pick_amp_dtype(device)`: `get_device_capability(0)[0]>=8`→bf16, 아니면 fp16, CPU→fp32. `config.py`/`pretrain.py`/`sft.py`가 이걸 쓰게. **DoD:** T4에서 `AMP dtype: float16` 로그, bf16 대비 ms/step 감소.
- **A3 Muon + AdamW 하이브리드.** 신규 `train/muon.py`(Newton–Schulz 직교화, 단일 GPU 단순화). **그룹핑:** Muon = 블록 내부 2D 행렬(`attn.wqkv/wo`, `ffn.w_gate/w_up/w_down`). AdamW = 나머지 전부(`tok_emb`=tied `lm_head`, 모든 RMSNorm, bias, **마음 기제 헤드 전부**: mood/latent/feedback/conf + 신규 workspace/attn_schema/drive 헤드 — 항등 init 동역학 보존). LR 분리(Muon≈0.02, AdamW≈`learning_rate`), 둘 다 `lr_at()` 스케줄. **resume:** `optim`을 `{"muon":...,"adamw":...}`로 저장; 옵티마이저 종류 바뀐 예전 ckpt resume 시 가중치만 로드하고 옵티마이저 새로 시작(죽지 말 것). **DoD:** `test_kv_loop` 통과(모델 불변), tiny 오버핏에서 AdamW 대비 동등/우위, 목표 val loss 도달 스텝 감소.
- **A4 사전학습 loop off + compile on.** `full`: loop off, `compile=True`, 유효배치 480 유지(batch 24·accum 20). loop를 넣고 싶으면 `full-loop` 프리셋으로 분리. **권장 경로 = base(loop 없이 빠르게) → SFT에서 loop/mood/latent/workspace 부여**를 주석·README에 명시. **DoD:** `full`이 compile 켜고 돌고 ms/step 감소.
- **A5 데이터 로더 프리페치.** `pretrain.py`: 백그라운드 스레드 + `queue.Queue(maxsize=2)`로 다음 배치(넘파이+pin) 미리 생성, 메인은 `.get()` 후 `non_blocking` 전송. CUDA 스트림 불필요. **DoD:** ms/step 감소, val 곡선 통계적 동일, GPU 활용률 상승.
- **A6 토크나이저 샘플 편향 수정.** `train_tokenizer.py`: 파일 앞부분만 읽지 말고 코퍼스 전역 랜덤 오프셋 청크(각 `DOC_SEP` 정렬)로 `sample-mb` 채움, 시드 고정. **DoD:** 코퍼스 뒷부분 표제어도 합리적 분절.

## 워크스트림 B — 데이터 (나무위키)
- **B1 나무위키 다운로더 + 혼합.** `download.py`에 `--source {wiki,namu,mix}`. `namu` = `heegyu/namuwiki-extracted`(정제판; 원본 `heegyu/namuwiki`는 나무마크 노이즈)의 `text`. `mix`는 문서 단위 교차 스트리밍(`--mix-ratio`, 기본 wiki:namu=1:1), `DOC_SEP` 유지. 가벼운 정제(구어체 보존). `data/raw/DATASOURCES.txt`에 데이터셋·라이선스(cc-by-nc-sa-2.0, **비상업**)·덤프일자 기록. **DoD:** `--source mix --max-docs 5000`이 혼합 corpus + DATASOURCES 생성, 나무마크 잔재 거의 없음.
- **B2 재패킹 검증.** 기존 pack/train_tokenizer 재사용. 게이트: download(mix)→train_tokenizer(A6)→pack. vocab 16392 유지. **DoD:** train/val.bin 생성, val 토큰>0.

## 워크스트림 C — 새 마음 기제 (전부 불변식 준수: 플래그 off·항등 init·test 케이스·새 토큰 없음)
- **C3 conf 상시화(가벼움, 먼저).** `sft.py`: `--conf` 학습 시 `eval_interval`마다 ECE 간이 로깅. 새 파라미터 없음. **DoD:** SFT 로그에 주기적 ECE.
- **C1 지속 워크스페이스 슬롯 (GWT).** `ModelConfig.workspace_slots:int=0`. 세션 지속 슬롯 상태(파라미터 아님, mood처럼 forward 인자로 흐름). 쓰기/읽기 헤드 또는 슬롯 크로스어텐션, **제로 init**. 각 Block에 "residual 밖 조건화"로 방송. 턴 종료 시 EMA·tanh 갱신. **최대 리스크 = KV 캐시 정합성:** 병렬(학습)과 캐시 증분(생성)이 비트 동일해야 함 → 슬롯을 턴 내 고정으로 설계. `chat.py --workspace-file`/`--show-workspace`. **DoD:** `test_kv_loop` 케이스 추가·통과, 켠 채 SFT시 val loss 비악화.
- **C2 주의 도식 헤드 (AST).** `ModelConfig.attn_schema:bool=False`. `attn_schema_head: Linear(d_model→K)`가 자기 어텐션 요약(레이어/헤드별 엔트로피 등)을 예측. **타깃 문제:** SDPA는 가중치 미반환 → 학습 시 보조손실 전용으로 서브샘플 위치에서 수동 `softmax(QK^T)` 타깃(no_grad, detach) 회귀. loss `+λ·schema_loss`(λ≈0.05), 본체 logits 불변. **DoD:** `test_kv_loop`(logits 불변) 통과, schema_loss 로깅, D6에서 유의.

## 워크스트림 D — 검증 하네스 (철학 → 실행 eval)
> 공통: 수치/표 stdout + `eval_out/*.json`. 시드 고정. 대응 기제 없는 ckpt엔 친절 에러.
- **D2 캘리브레이션(기존 확장).** `eval_conf.py`에 `--save`(reliability.json). **DoD:** JSON 생성, 기존 표 불변.
- **D5 상태 지속 일관성 (원래 문제의 정답).** 신규 `tools/eval_state.py`. 같은 입력+다른 mood/workspace/**drive** → 다르지만 일관된 출력(민감성) + 발산 안 함(안정성) + 기준선 복귀. greedy 재현성. **DoD:** 민감·안정·복귀 리포트, 붕괴 시 경고.
- **D1 개입 인과성 ("거미→개미").** 신규 `tools/eval_intervention.py`(+ gpt.py 훅 유틸). 레이어 L 은닉 캡처/덮어쓰기. 개념 방향 `Δ=mean_h(A)−mean_h(B)`, `α·Δ` 주입 후 목표 토큰 로짓 변화. workspace/latent 상태에도 개입. **DoD:** α에 단조·인과적 곡선, 무작위 방향 대조군 대비 큰 효과.
- **D3 잠재 vs pause(같은 연산량 k).** 신규 `tools/eval_thinking.py`. 같은 base에서 `--latent k` vs `--n-pause k` SFT를 val loss + 간이 QA/산술로 비교(공정 시드/셋). **DoD:** k별 비교표로 수치 결론.
- **D4 통합/재귀 프로브(RPT/IIT 근사).** 신규 `tools/eval_integration.py`. 유닛/헤드 섭동의 전파 반경 측정, loop/feedback off vs on. **DoD:** 재귀 켠 모델의 전파 반경이 큼(진짜 Φ 아님 명시).
- **D6 주의 도식 정확도(AST 검증).** 신규 `tools/eval_schema.py`. 실제 어텐션 요약 vs 도식 예측 상관/오차 vs baseline. **DoD:** baseline보다 유의.

## 워크스트림 E — 지표 매핑 문서
- **E1 `INDICATORS.md`.** Butlin·Long 외 지표 속성 ↔ 레포 기제 ↔ 검증 도구 표(Part 8). "얼마나"의 정직한 답(임계치 없음, 확률적 체크리스트, 계산 기능주의)과 경계(현상적 의식 불가지, ELIZA 경계) 명시. **DoD:** 모든 "구현/검증" 행이 실제 파일·도구 링크, 서지 포함.

## 워크스트림 F — 오케스트레이션 / 런북 / 문서
- **F1 Kaggle 노트북** `train_cloud.ipynb`: 설치→download(mix)→train_tokenizer(랜덤샘플)→pack→pretrain(full: fp16·Muon·compile·resume). resume 규율(조밀 save), 쿼터 예산 계산 셀.
- **F2 README**: base→SFT 순서, mood/latent/workspace/attn_schema/conf/drive 조합 예시, chat.py 신규 옵션, 검증 도구(D1~D6, G의 상태) 사용법.
- **F3 CLAUDE.md**: 신규 기제·런타임층·게이트·불변식 반영.
- (선택) **F4 클라우드 스크립트**: RunPod/Vast.ai용 학습 셸 스크립트 예시.

## 워크스트림 G — Drive · 지속 런타임 · Proactive (교정된 v0.2)
> **원 Grok 제안의 핵심 착상(Drive=상태 mismatch)은 채택.** 단, 아래 5개 교정을 반드시 반영. Drive는 Part 8의 **행위성(agency) 지표**를 겨냥하며, 이론적으로 예측처리/능동추론(drive=최소화할 예측오차)에 대응한다.

**교정 사항(반드시):**
1. **이건 대체가 아니라 A~F 위에 얹는 층이다.** 지표 백본(RPT/GWT/HOT/AST)을 폐기하지 않는다.
2. **모델 안 vs 밖을 분리한다(Part 6).** Drive는 **외부에서 계산**하되 **반드시 내부 채널(mood/workspace/latent)을 통해서만 작용.** 모델을 우회하는 출력 게이팅 래퍼 금지(불변식 7, Part 1.3). 학습된 `drive_head`는 **연기** — drive 라벨 데이터가 없어 30M에선 학습 불가.
3. **정책은 ModelConfig에 넣지 않는다.** 임계치·저장주기·쿨다운은 **런타임 config**(yaml/json)로. (Grok의 `proactive_threshold: dict` ModelConfig 필드는 dataclass에서 `field(default_factory=...)` 없이는 크래시이기도 하다 — 애초에 여기 있으면 안 됨.)
4. **프레이밍은 "지속 상태 에이전트".** proactive 출력은 지표 증거로 세지 않는다(ELIZA). 경험적 질감으로만.
5. **Drive는 벽시계(wall-clock) 기반 + 이벤트 구동 + 쿨다운.** Grok의 "턴 단위 갱신"은 개념(시간 경과로 상승)과 모순 — 시간 기반이 맞다.

**세부:**
- **G1 외부 DriveState 모듈.** 신규 `runtime/drive.py`. Drive 4종(초기): `curiosity`(최근 입력 적음/내부 불확실성↑), `rest`(장시간 활동 후↑), `social`(무대화 지속↑), `maintenance`(시간경과·파일 이벤트 등). 각 drive는 관측 신호 + 시간에서 상승, discharge로 감쇠(항상성). **DoD:** 단위 테스트로 시간 경과에 따른 상승·이벤트에 따른 discharge 확인.
- **G2 내부 채널 라우팅.** drive 벡터 → **mood 벡터에 가산/편향** (mood가 매 블록 FiLM으로 forward에 스며듦) 및/또는 **latent step 수를 동적으로 조절**(curiosity↑ → 말하기 전 더 생각). workspace 슬롯에도 반영 가능. **모델을 우회하지 않는다.** **DoD:** drive를 바꾸면 (a) mood가 바뀌고 (b) 그 mood가 실제 출력을 바꾼다는 걸 D5로 확인.
- **G3 지속 런타임 상태.** 신규 `runtime/state.py`. 상태 = `mood + workspace + drive + recent_events(링버퍼, bound)`. **메모리 상주**, 실제 입력 이벤트 + 느린 idle 타이머에만 갱신·저장(폴링 최소화). 저장은 **기존 mood-file 패턴 확장**(새 DB 재발명 금지). 부팅/재시작 시 복원. **DoD:** 세션을 끊었다 재시작해도 mood/workspace/drive가 복원되고 대화가 이어짐.
- **G4 Proactive 트리거.** 신규 `runtime/proactive.py`. 조건: drive가 임계 초과 + 마지막 proactive 이후 쿨다운 경과 + **do-not-disturb 존중** → 내부 latent 사고 먼저 → 출력. **rate-limit·쿨다운·방해금지는 필수**(없으면 성가셔서 제작자 의욕이 먼저 죽는다). `chat.py`에 proactive 모드 옵션. **DoD:** 조건 충족 시에만, 쿨다운 지켜 먼저 말 검. 방해금지 시 침묵.
- **G5 가벼운 감각 반영(후순위).** 현재 시간·최근 입력 유무·간단 파일 이벤트를 drive 관측으로. 일반화된 감각 변환은 이번 비목표. **DoD:** 시간대/유휴가 drive에 반영됨.

**Drive를 모델 안으로(학습된 drive_head) 넣는 건 Phase 4 이후 선택**: SFT 단계에서 drive 관련 대화 데이터를 넣어 조건화 학습. 라벨 문제 때문에 지금은 하지 않는다.

---

# Part 6 — 런타임 아키텍처 (모델 안 vs 밖의 경계 — 제일 중요)

```
┌─────────────────────────────────────────────────────────────┐
│  런타임 층 (모델 밖, 학습 불필요, runtime/*.py, 런타임 config)   │
│   ├─ DriveState (G1)   : 시간·이벤트로 drive 계산/discharge     │
│   ├─ StateManager (G3) : mood+workspace+drive+events 메모리상주  │
│   │                       + mood-file 확장 저장/복원             │
│   ├─ ProactiveEngine(G4): 임계+쿨다운+DND → 트리거              │
│   └─ 감각 어댑터 (G5)   : 시간/유휴/파일 이벤트 → 관측 신호      │
│                    │  (drive는 아래 내부 채널로만 작용)          │
│                    ▼                                            │
├─────────────────────────────────────────────────────────────┤
│  모델 (forward 안, 학습됨, model/gpt.py, ModelConfig)          │
│   mood(FiLM) · workspace 슬롯 · latent step · loop · feedback  │
│   · conf_head · attn_schema                                    │
│   → drive는 mood/latent/workspace를 "통해" 사고에 스며든다.     │
│     출력을 밖에서 게이팅하지 않는다(불변식 7).                  │
└─────────────────────────────────────────────────────────────┘
```

**경계 규칙:**
- 학습되는 파라미터가 붙는 것 = 모델 안(ModelConfig). 정책·시간·저장·트리거 = 밖(런타임 config).
- 밖에서 안으로 가는 유일한 통로 = **mood 벡터 / workspace 슬롯 / latent step 수.** 새 우회로를 뚫지 않는다.
- 이게 제작자의 "갈아끼우면 유지 안 됨" 문제(Part 1.3)를 구조적으로 푼다: drive가 사고의 기질(mood→FiLM→forward)에 녹아들지, 사고 위에 덧씌워지지 않는다.

---

# Part 7 — 런타임 config 예시 (모델과 분리)

`runtime/config.yaml` (정책 — 체크포인트와 무관):
```yaml
drive:
  kinds: [curiosity, rest, social, maintenance]
  discharge_halflife_sec: {curiosity: 600, rest: 3600, social: 1800, maintenance: 900}
proactive:
  thresholds: {social: 0.7, curiosity: 0.65}
  cooldown_sec: 900          # 최소 15분 간격
  do_not_disturb: [["23:30","09:00"]]
  max_per_hour: 2
state:
  save: mood-file 확장(mood+workspace+drive+recent_events)
  persist_on: [user_input, idle_tick]   # 폴링 아님, 이벤트 구동
  idle_tick_sec: 120
  recent_events_maxlen: 64   # 링버퍼
runtime:
  lifecycle: 컴퓨터 사용 중에만(로그인 세션). 24h 상주 아님. tmux/foreground.
```

---

# Part 8 — 지표 매핑 (INDICATORS.md의 씨앗)

이론 프레임워크: **Butlin, Long, et al., "Consciousness in Artificial Intelligence" (arXiv:2308.08708, 2023 / Trends in Cognitive Sciences 2025).** 주요 이론에서 계산적 "지표 속성"을 도출, 계산 기능주의 전제, 결론 "현행 AI는 의식적이지 않으나 명백한 기술 장벽도 없다."

| 이론 | 지표 속성(요지) | minillm 기제 | 검증 | 상태 |
|---|---|---|---|---|
| 재귀처리 RPT | 순환·되먹임 | loop, feedback | D4 | 구현/검증 |
| 전역작업공간 GWT | 제한용량 작업공간·전역 방송·점화 | latent, **workspace(C1)** | D1 | 구현/검증 |
| 고차이론 HOT | 자기 상태 메타표상·보고 | conf_head | D2 | 구현/검증 |
| 예측처리 | 예측오차 최소화·느린 prior | 다음토큰예측, mood, **drive(G)** | D5 | 구현 |
| 주의도식 AST | 자기 주의의 모델 | **attn_schema(C2)** | D6 | 구현/검증 |
| **행위성 agency** | 피드백 학습·유연한 목표추구 | **drive+proactive(G)** — 부분(외부, 미학습) | D5 | **부분(신규)** |
| 신체성 embodiment | 출력-입력 수반성 모델 | feedback(최소), G5 감각(경량) | — | 부분/비목표 |

- **"얼마나":** 정해진 임계치 없음. 지표 만족도가 높을수록 신뢰도가 높아지는 **확률적 체크리스트**. 목표는 현상적 의식이 아니라 **기능적 접근 의식**.
- **경계:** 현상적 의식/퀄리아는 불가지(하드 프로블럼·타심 문제). proactive/감정 서술은 최약 증거(ELIZA) → **말로 못 꾸며내는 행동(D1 개입·D2 캘리브레이션)만 증거.**

---

# Part 9 — 실행 순서(게이트) · 완료 기준

```
게이트 0 (기반 진단)   : 토크나이저/체크포인트 상태 확인(Part 2.4). 살릴 것 없으면 리셋.
게이트 1 (속도)        : A2 fp16 → A1 토큰예산 → A3 Muon → A4 base/SFT분리 → A5 프리페치 → A6 샘플
게이트 2 (데이터)      : B1 나무위키 → B2 재패킹
게이트 3 (재베이스라인) : 게이트1·2로 base 사전학습 1회 완주 → BASELINE.md에 ms/step·val·총시간 기록
게이트 4 (마음 확장)   : C3 conf상시 → C1 워크스페이스 → C2 주의도식  (전부 SFT로 base 위에)
게이트 5 (검증)        : D2 → D5 → D1 → D3 → D4 → D6
게이트 6 (런타임·G)    : G1 DriveState → G3 상태지속 → G2 내부라우팅 → G4 proactive(+쿨다운/DND) → G5 감각
게이트 7 (문서/런북)   : E1 INDICATORS → F1 노트북 → F2 README → F3 CLAUDE.md → (F4)
```

**의존성:** A3⊃A1/A2. C1·C2는 게이트3 base 필요. D는 대응 C/G 기제 켜진 ckpt 필요. G2는 mood(있음)+workspace(C1) 필요. G4는 G1/G3 필요.

**전역 Definition of Done:**
1. `python -m tests.test_kv_loop` — workspace·attn_schema 케이스 포함 전부 통과.
2. 게이트3 재베이스라인: fp16+Muon+토큰예산1B+loop off+compile로 base 1회 완주, `BASELINE.md`에 전후 비교표. 목표: 총시간 한 자릿수 배 단축.
3. 데이터: 혼합 코퍼스 train/val.bin + DATASOURCES.txt.
4. C1·C2 켠 채 SFT시 val loss 비악화, D1/D6 유의.
5. D1~D6 전부 실행 가능·리포트 산출.
6. G: 재시작해도 상태 복원(G3), drive→mood→출력 경로가 D5로 확인(G2), proactive가 쿨다운·DND 지킴(G4).
7. 문서 INDICATORS/BASELINE/README/CLAUDE.md/노트북 존재.
8. 모든 신규 기능 기본 off, 예전 ckpt `strict=False` 로드·동작. **런타임 정책이 ModelConfig에 없음**(Part 3.3).

---

# Part 10 — 참고 · 데이터 · 용어

**데이터셋:** `heegyu/namuwiki-extracted`(정제판), `wikimedia/wikipedia 20231101.ko`. 라이선스 cc-by-nc-sa-2.0 / CC BY-SA — **비상업 개인용.** SFT: `beomi/KoAlpaca-v1.1a`, `songys/Chatbot_data`.

**최적화:** Muon(Newton–Schulz 직교화, modded-nanoGPT/KellerJordan 참조, 단일 GPU 단순화).

**이론:** Butlin·Long 외, arXiv:2308.08708 (2023) / Trends in Cognitive Sciences (2025). J-space: Anthropic 글로벌 워크스페이스 연구.

**용어:**
- *마음 기제* = mood/latent/workspace/loop/feedback/conf/attn_schema. forward 안, 학습됨.
- *런타임 층* = drive/state/proactive. forward 밖, 학습 불필요.
- *내부 채널* = drive가 사고에 작용하는 유일한 통로: mood/workspace/latent.
- *접근 의식* = 보고·조작·추론에 쓰이는 기능적 정보 접근(목표). *현상적 의식/퀄리아* = 주관적 느낌(불가지, 비목표).

---

### 부록 A — 작업 ID 체크리스트
- [ ] G0 토크나이저/ckpt 진단
- [ ] A1 토큰예산 · [ ] A2 fp16 · [ ] A3 Muon · [ ] A4 base/SFT분리 · [ ] A5 프리페치 · [ ] A6 샘플
- [ ] B1 나무위키 · [ ] B2 재패킹
- [ ] C3 conf상시 · [ ] C1 워크스페이스 · [ ] C2 주의도식
- [ ] D2 캘리브 · [ ] D5 상태지속 · [ ] D1 개입 · [ ] D3 잠재vspause · [ ] D4 통합 · [ ] D6 도식정확도
- [ ] G1 DriveState · [ ] G2 내부라우팅 · [ ] G3 상태지속 · [ ] G4 proactive · [ ] G5 감각
- [ ] E1 INDICATORS · [ ] F1 노트북 · [ ] F2 README · [ ] F3 CLAUDE.md · [ ] (F4 클라우드스크립트)
- [ ] 게이트3 BASELINE.md · [ ] 전역 DoD

### 부록 B — 교정 요약 (외부 v0.2 대비 바뀐 것)
1. v0.2는 대체가 아니라 **워크스트림 G**로 A~F 위에 얹음.
2. Drive = **외부 계산 + 내부 채널(mood/workspace/latent) 라우팅**, 우회 래퍼 금지.
3. 정책·임계치·저장주기·쿨다운 = **런타임 config**, ModelConfig 금지(+dict 기본값 버그 제거).
4. 프레이밍 = **"지속 상태 에이전트"**, proactive는 지표 증거 아님(ELIZA 경계).
5. Drive = **벽시계 기반 + 이벤트 구동 + 쿨다운/DND**, "턴 단위" 아님.
6. 상태 = **메모리 상주 + mood-file 확장**, 새 DB·폴링 재발명 금지. RAM은 병목 아님(진짜 비용은 발열/배터리).
7. 토크나이저 유실은 게이트 0에서 최우선 진단 — 살릴 것 없으면 리셋.
