# minillm

밑바닥부터 직접 만드는 한국어 초소형(~34M) 대화 LLM. 성능 경쟁이 아니라
**"사람 마음처럼 작동하는 말동무"를 내 손으로 만들어 이해하는 것**이 목표다.

## 프로젝트가 지향하는 것

- Anthropic의 J-space(글로벌 워크스페이스) 연구와 제작자의 아판타시아적 인지
  ("이미지 없이 개념체로 사고하기")에서 영감받은 **마음 유사 기제**를 작은 모델에
  인위적으로 구축하는 실험. 단순 챗봇 성능 향상이 아니라 "기능적 마음 구조"가 목적.
- 핵심 실험 둘: **latent vs pause** (연속 잠재 사고가 이산 필러 속말보다 나은가 —
  같은 연산량 k에서 val loss 비교), **calibration** (`tools/eval_conf.py` —
  확신도가 실제 정답률과 맞으면 기능적 메타인지 성립).
- 새 기능을 제안할 때는 이 관점에서: "사람 마음의 어떤 측면을 흉내내는가"를
  먼저 말하고, 공학적 함정(학습 병렬성, CPU 비용)을 솔직하게 평가할 것.

## 환경 제약 (설계를 지배하는 조건)

- **학습**: Kaggle 무료 T4 (12h 세션, `--resume` 필수, 주 30h). 무거운 기능은
  사전학습보다 SFT(30분~1시간)로 학습 가능하게 설계하는 것을 우선.
- **추론**: VRAM 없는 노트북 CPU fp32, 램 여유 ~1GB. 매 토큰 비용을 곱하는
  변경(loop 등)은 신중히, 턴당 한 번인 비용(latent, mood)은 관대하게.
- 총비용 ₩0 유지.

## 아키텍처 규칙 (어기면 안 되는 것)

1. **모든 기능은 `ModelConfig` 플래그, 기본값 off.** 예전 체크포인트가 코드
   최신판에서 그대로 로드·동작해야 한다.
2. **항등/제로 초기화 원칙**: 기능을 켠 직후에는 출력이 바뀌지 않아야 한다
   (FiLM 제로 init, `latent_proj` eye init, `feedback_proj` 제로 init처럼).
   기존 체크포인트에 `strict=False`로 얹는 경로를 항상 보장.
3. **체크포인트는 자기 설정을 내장** (`model_config`). `chat.py`는 플래그 없이
   자동으로 올바르게 동작해야 하고, 오버라이드 인자는 실험용으로만.
4. **정확성 게이트**: 모델/캐시 코드를 건드리면 반드시
   `python -m tests.test_kv_loop` 통과 (캐시 증분 ≡ 일괄 처리). 새 기제를
   추가하면 여기에 케이스도 추가.
5. vocab은 16392 (특수 토큰 12개, `tokenizer/bpe.py`의 `SPECIAL_TOKENS`).
   기존 4개의 ID(16380~16383)는 절대 바꾸지 말 것 — 패킹된 .bin 호환성.

## 코드 스타일

- 이 저장소는 **배우면서 읽는 교재**다. 주석·독스트링은 한국어로, "무엇을"이
  아니라 "왜 이렇게 하는가"를 설명한다. 기존 파일들의 주석 밀도를 따를 것.
- 파일 구성은 단순하게 유지: 모델은 `model/gpt.py` 한 파일이 전부라는 원칙.
- 외부 의존성 추가 금지에 가깝게 (현재: torch, numpy, regex, datasets, tqdm).

## 구현된 마음 유사 기제 (2026-07 기준)

| 기제 | 켜는 곳 | 요약 |
|---|---|---|
| loop | `full-loop` 프리셋 또는 SFT | 중간 블록 2..6을 2회 반복, 반복 횟수 확률 샘플 학습 |
| pause | `prepare_sft --n-pause` + `sft --n-pause` | 답변 앞 `<\|pause\|>` 강제 삽입 (mask 0) |
| mood | `sft --mood-dim` | 턴 간 지속·감쇠 기분 벡터, FiLM 주입, 2-pass 학습 |
| latent | `sft --latent` | Coconut식 은닉 되먹임, 왼쪽 패딩+직사각 마스크로 병렬 학습 |
| feedback | `sft --feedback` | 직전 토큰 최종 은닉→다음 입력 방송, 2-pass 근사 (mood와 1-pass 공유) |
| conf | `sft --conf` | 확신도 헤드(detach 절연), `chat --adaptive-latent`로 적응적 사고 |
| workspace | `sft --workspace-slots` | GWT 지속 작업공간 슬롯, ws_read 전역 방송(제로init)·ws_write EMA 갱신, 턴 내 고정으로 캐시 정합, `chat --workspace-file` 세션 지속 |
| attn_schema | `sft --attn-schema` | AST 주의 도식 — 레이어별 어텐션 엔트로피를 은닉에서 회귀, 보조손실만(0.05), logits 불변 |

### 런타임 층 (모델 밖 — 핸드오프 워크스트림 G)

학습 파라미터가 붙지 않는 지속 상태·동인·proactive. **정책(임계·쿨다운·DND)은
`runtime/config.json`에만** 두고 ModelConfig에 넣지 않는다.

| 모듈 | 역할 |
|---|---|
| `runtime/drive.py` | DriveState 4종(curiosity/rest/social/maintenance), 벽시계 상승·이벤트 discharge |
| `runtime/route.py` | drive→mood 가산·curiosity→latent 스텝 (내부 채널만, 우회 금지) |
| `runtime/state.py` | mood+ws+drive+events 메모리 상주, `--state-file` 저장/복원 |
| `runtime/proactive.py` | 임계+쿨다운+DND+시간당 상한 → 먼저 말 걸기 판정 |
| `runtime/sense.py` | 시각·유휴·파일 mtime 등 경량 관측 |

`chat.py --state-file session.pt --proactive --show-drive`. proactive는 지표
증거가 아님(ELIZA). 단위 테스트: `python -m tests.test_drive`.

보류된 아이디어(다시 제안되면 이 결정을 참고): FiLM 자기 억제(학습 신호 없음),
레지스터 토큰(mood와 중복). 진짜 읽기-쓰기 메모리 슬롯은 workspace(C1)가
부분적으로 실현 — 완전한 Block-Recurrent급 read-write 메모리는 다음 세대 후보.
학습된 `drive_head`는 Phase 4 이후 선택(라벨 데이터 없음 → 지금은 외부 계산).

## 검증 하네스 (tools/) 와 지표 매핑

각 기제가 실제로 기능하는지 수치로 확인하는 도구 D1~D6 + eval_loop. 인지과학
지표(Butlin et al.)와의 1:1 대응은 `INDICATORS.md`. 도구: eval_conf(HOT 캘리브),
eval_state(상태 지속), eval_intervention(개입 인과성), eval_thinking(latent vs
pause), eval_integration(RPT 전파), eval_schema(AST 정확도). 전부 `--save`로
`eval_out/` JSON. 공용 로더는 `tools/_common.py`.

## 게이트 순서·불변식 (핸드오프 v1.0)

- 게이트: (0)토크나이저/ckpt 진단 → (1)속도 → (2)데이터 mix → (3)재베이스라인 →
  (4)마음확장(conf·workspace·attn_schema) → (5)검증 D1~D6 →
  (6)런타임 G(drive·state·proactive) → (7)문서.
- 강화된 불변식: 신규 기제는 **새 토큰을 도입하지 않는다**(전부 벡터/헤드
  수준 → `.bin` 재패킹 불필요). T4는 하드웨어 bf16 없음 → fp16 자동.
  런타임 정책은 ModelConfig 금지. 마음 기제는 forward 안, drive는 내부 채널로만.

## 현재 상태와 다음 단계

- 코드는 8개 마음 기제 + 검증 하네스 + **런타임 층(G)** 구현·단위 테스트 완료.
  옵티마이저 Muon 하이브리드, 토큰예산 기반 max_steps, 프리페치 반영.
  사전학습은 **아직 시작 전**.
- 권장 경로: `full`(base, loop off + compile on)로 빠르게 사전학습 →
  SFT에서 마음 기제 조합 부여. loop를 사전학습에 넣으려면 `full-loop`.
- 다음: Kaggle에서 `download --source mix` → tokenizer(랜덤샘플) → pack →
  `pretrain --preset full --optimizer muon` 1회 완주하고 `BASELINE.md` 채우기
  → SFT 조합 실험 → D1~D6 검증 → 로컬 `chat.py --state-file --proactive`.
- 로컬 `tokenizer/tokenizer.json`은 3MB 샘플로 만든 테스트용 (커밋 안 됨).
  Kaggle에서 200MB로 학습한 진짜로 교체 예정.

## 검증 습관

- 새 학습 경로는 tiny 프리셋 + 합성 데이터로 로컬 CPU에서 먼저 오버핏 확인 후
  Kaggle 투입.
- 실험 결과는 "같은 사전학습 체크포인트에서 가른 SFT들의 masked val loss 비교"가
  기본 형식. 체크포인트 이름에 기능을 명시 (`sft_pause4.pt`, `sft_latent2.pt`).
