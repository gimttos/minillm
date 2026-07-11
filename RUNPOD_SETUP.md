# RunPod에서 손 안 대고 자동 학습하기 (완전 초보용)

이 문서 하나만 위에서부터 따라 하면, **데이터 다운로드 → 토크나이저 →
패킹 → SFT 데이터 준비 → 사전학습 → SFT**까지 사람이 개입하지 않고 쭉 돌고,
**중간중간 Discord로 진행상황이 오고**, 끝나면 스스로 결과를 백업하고
(원하면) 파드를 꺼서 과금을 멈춘다.

> 실행 엔진은 저장소 루트의 `run_minillm_training.sh` 하나다. 이 스크립트가
> "어디서 끊겨도 이어서(멱등성 + `--resume`), 알림 주고, 끝나면 백업/정지"를
> 전부 담당한다. 아래는 그걸 **RunPod에서 어떻게 켜느냐**의 안내다.

---

## 0. 3분 개념 정리 (RunPod가 처음이라면)

| 용어 | 뜻 (이 프로젝트 기준) |
|---|---|
| **Pod** | 빌린 GPU 컴퓨터 한 대. 켜면 시간당 과금, 끄면(stop) GPU 과금은 멈춤. |
| **Container Disk** | 파드에 딸린 임시 디스크. **파드를 지우면 같이 사라진다.** |
| **Network Volume** | 파드와 별개로 사는 영구 저장소. 보통 `/workspace`에 붙는다. 네가 만든 **80GB가 이것**. 파드를 껐다 켜도, 심지어 파드를 지워도 데이터가 남는다. |
| **web terminal / SSH** | 파드 안에 들어가 명령어를 치는 창. |
| **stop vs terminate** | **stop**=GPU 과금만 멈춤(볼륨·데이터 유지, 나중에 재개 가능). **terminate**=파드 삭제. 우리는 항상 **stop**을 쓴다. |

**핵심 원칙 하나만 기억:** 모든 것을 `/workspace`(=80GB 네트워크 볼륨) 안에서
한다. 그래야 파드가 죽거나 껐다 켜도 코드·데이터·체크포인트가 안 사라지고,
스크립트를 다시 실행하면 끊긴 지점부터 이어진다.

---

## 1. 준비물 (파드 만지기 전에)

### 1-1. Discord 웹훅 URL 만들기 (알림받을 통로) — 5분

1. 디스코드에서 알림받을 **서버**를 하나 정한다(없으면 `+`로 새 서버 생성, 개인용이면 나 혼자만 있어도 됨).
2. 알림받을 **채널** 옆 톱니바퀴(**채널 편집**) → 왼쪽 **연동(Integrations)** →
   **웹후크(Webhooks)** → **새 웹후크** → 이름 아무거나(예: minillm) →
   **웹후크 URL 복사**.
3. 이 URL을 잘 보관한다. `https://discord.com/api/webhooks/....` 형태다.
   → 나중에 `DISCORD_WEBHOOK_URL` 환경변수에 넣는다.

> 웹훅 URL은 비밀번호나 마찬가지다(아는 사람은 그 채널에 글을 쓸 수 있다).
> 공개 저장소·스크린샷에 노출하지 말 것. 그래서 코드가 아니라 **환경변수**로 넣는다.

### 1-2. (선택) HuggingFace 토큰 — 결과를 파드 밖에 영구 백업하고 싶다면

- huggingface.co → Settings → **Access Tokens** → `write` 권한 토큰 생성 → 복사.
- huggingface.co에서 **New model/dataset**로 private **dataset** repo 하나 생성
  (예: `gimttos/minillm-runs`).
- → 나중에 `HF_TOKEN`, `HF_REPO`(예: `gimttos/minillm-runs`) 환경변수로 넣는다.
- 넣지 않으면 백업은 `/workspace`(네트워크 볼륨)에만 남는다. 볼륨을 안 지우면 그것도 안전.

### 1-3. RunPod 크레딧

- RunPod 계정에 크레딧을 충전해 둔다(4090은 대략 시간당 몇 백 원대, 변동). 잔액이
  0이 되면 파드가 강제로 멈추니 학습 예상 시간(아래 6장) + 여유를 두고 충전.

---

## 2. 파드에 80GB 볼륨 붙이고 실행 (이미 4090 파드가 있다면)

네트워크 볼륨은 **파드를 만들 때** 연결한다. 이미 만든 파드에 볼륨이 안 붙어
있으면, 그 파드는 stop/terminate하고 **볼륨을 선택해서 새로 파드를 만드는 게**
가장 확실하다(데이터는 볼륨에 있으니 파드를 새로 만들어도 안전).

1. RunPod 콘솔 → **Storage**에서 네가 만든 80GB 볼륨의 **리전(region)**을 확인.
2. **Pods → Deploy** → 같은 리전에서 **4090** 선택.
3. **Network Volume** 항목에서 그 80GB 볼륨을 선택 → `/workspace`에 마운트됨.
4. **템플릿(이미지)은 PyTorch 계열**을 고른다(예: "RunPod PyTorch 2.x").
   → torch가 이미 깔려 있어 설치가 빠르고 GPU 인식이 보장된다.
5. **Deploy On-Demand**로 띄운다. (Spot/Interruptible은 싸지만 중간에 강제 종료될
   수 있음 — 우리 스크립트는 재개가 되지만, 처음엔 On-Demand를 권장.)

> 이미 만든 4090 파드에 볼륨이 잘 붙어 있으면 이 장은 건너뛰고, 그 파드에
> `/workspace`가 있는지 `df -h /workspace`로 확인만 하면 된다.

---

## 3. 환경변수(Secrets) 등록

스크립트 동작을 코드 수정 없이 바꾸는 스위치들이다. 파드 배포 화면(또는
Template 편집)의 **Environment Variables**에 아래를 넣는다.

**필수급**

| 이름 | 값 예시 | 설명 |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | `https://discord.com/api/webhooks/...` | 진행 알림 통로 (1-1에서 만든 것) |

**선택 (파드 밖 백업)**

| 이름 | 값 예시 |
|---|---|
| `HF_TOKEN` | `hf_xxx` (write 토큰) |
| `HF_REPO` | `gimttos/minillm-runs` |

**선택 (완료 시 파드 자동 정지 → 과금 절약)**

| 이름 | 값 예시 | 설명 |
|---|---|---|
| `POD_AUTO_STOP` | `true` | 파이프라인 끝나면 스스로 파드 stop |
| `RUNPOD_API_KEY` | `...` | RunPod 콘솔 Settings → API Keys에서 발급 |

> `RUNPOD_POD_ID`는 RunPod가 파드마다 **자동으로** 주입하므로 직접 넣지 않아도 된다.
> 자동 정지를 쓰려면 파드 안에서 `runpodctl`이 그 API 키를 알아야 한다(4장 참고).

**선택 (파이프라인 튜닝 — 기본값으로도 잘 돔)**

| 이름 | 기본 | 설명 |
|---|---|---|
| `SOURCE` | `mix` | 데이터 소스 `wiki`/`namu`/`mix` |
| `MAX_DOCS` | `0`(전체) | 스모크 테스트 땐 `3000` 등 소량 |
| `SAMPLE_MB` | `200` | 토크나이저 학습 표본 크기 |
| `TARGET_TOKENS` | `0`(=1B) | 사전학습 토큰 예산 |
| `PRETRAIN_PRESET` | `full` | `full`/`full-loop`/`tiny` |
| `SFT_ARGS` | `--latent 2 --conf` | SFT 마음 기제 조합 |
| `HEARTBEAT_MIN` | `30` | 학습 중 진행 알림 주기(분) |

> 환경변수를 나중에 바꾸려면 파드를 편집(재시작)해야 반영된다. 귀찮으면 접속해서
> 실행 직전에 `export DISCORD_WEBHOOK_URL=...` 처럼 터미널에서 직접 넣어도 된다.

---

## 4. 파드 접속 + 최초 셋업 (딱 한 번)

파드가 뜨면 콘솔에서 **Connect → Start Web Terminal**(또는 SSH)로 들어간다.
아래를 **순서대로** 복붙한다.

```bash
# (1) 네트워크 볼륨으로 이동 — 반드시 /workspace 아래에서 작업한다
cd /workspace

# (2) 코드 받기 (이미 clone 돼 있으면 이 블록은 건너뛰고 pull만)
git clone https://github.com/gimttos/minillm.git
cd minillm

# (3) 파이썬 의존성 (PyTorch 템플릿이면 torch/numpy는 이미 있음)
pip install -U datasets regex tqdm huggingface_hub

# (4) (자동정지 쓸 때만) runpodctl에 API 키 등록
#     RUNPOD_API_KEY를 환경변수로 넣었다면:
runpodctl config --apiKey "$RUNPOD_API_KEY"
```

> `git clone`이 안 되면(사설 저장소) 토큰이 붙은 URL을 쓰거나, RunPod 콘솔의
> 파일 업로드로 코드를 올려도 된다. 공개 저장소면 위 그대로 된다.

---

## 5. 먼저 5분 스모크 테스트 (진짜 학습 전에 강력 권장)

"손 안 대고 몇 시간"을 돌리기 전에, **파이프라인 전체가 배선대로 도는지**를
아주 작은 데이터로 5분 만에 확인한다. 여기서 Discord 알림도 실제로 오는지 본다.

```bash
cd /workspace/minillm
MAX_DOCS=3000 SAMPLE_MB=20 PRETRAIN_PRESET=tiny \
  bash run_minillm_training.sh
```

- `tiny` 프리셋은 CPU/GPU에서 몇 분이면 끝나는 초소형 설정이라 "되긴 되는가"만 본다.
- 끝까지 통과하고 Discord에 `✅ ... 완료`, 마지막에 `🎉 전체 완료`가 오면 성공.
- 통과했으면 다음 실행을 **깨끗하게** 하려고 스모크 산출물을 지운다:

```bash
rm -f data/raw/.download.done tokenizer/.tokenizer.done \
      data/bin/.pack.done data/bin/.sft_prepared.done \
      checkpoints/.pretrain.done checkpoints/.sft.done
rm -rf data/raw data/bin checkpoints tokenizer/tokenizer.json
```

> 이 `.done` 파일들이 "이 단계 끝났음" 표시(마커)다. 지워야 진짜 데이터로 처음부터 다시 한다.

---

## 6. 진짜 학습 — 접속을 끊어도 계속 돌게 (핵심)

web terminal 창을 닫으면 그 안에서 돌던 명령이 죽는다. 그래서 **tmux**(화면을
떼었다 붙일 수 있는 세션)** 안에서** 돌린다. 이러면 브라우저를 닫아도, 노트북을
꺼도 파드 안에서 학습이 계속된다.

```bash
cd /workspace/minillm
tmux new -s train           # 'train'이라는 이름의 세션 생성 (안에 들어가짐)

# (tmux 안에서) 진짜 학습 시작 — 기본값(mix, 1B 토큰, full)
bash run_minillm_training.sh
```

- 이제 **`Ctrl+b` 를 누르고 손 뗀 뒤 `d`** → 세션에서 "빠져나옴(detach)". 학습은 계속 돈다.
- 브라우저/터미널을 닫아도 된다. **여기서부터 손 뗌.** Discord로 알림이 온다.
- 다시 상태를 보고 싶으면 파드에 접속해서:

```bash
tmux attach -t train        # 다시 세션 안으로 (로그가 실시간으로 보임)
# 다시 빠져나오려면 Ctrl+b 그다음 d
```

**예상 소요(대략, 4090 기준):** 다운로드/토크나이저/패킹이 데이터 양에 따라
수십 분~한두 시간, 사전학습(1B 토큰, ~4천 스텝)이 몇 시간. 데이터가 크면
다운로드가 가장 지루하다. 첫 실행이 부담되면 `TARGET_TOKENS=200000000`(0.2B)로
짧게 한 바퀴 돌려 감을 잡아도 된다.

### 파드가 중간에 죽거나 껐다 켜졌다면?

당황할 것 없다. 다시 접속해서 **똑같은 명령을 다시 실행**하면 된다:

```bash
cd /workspace/minillm
tmux new -s train
bash run_minillm_training.sh      # 끝난 단계는 건너뛰고, 사전학습은 이어서(resume) 감
```

끝난 단계는 `.done` 마커 덕에 건너뛰고, 사전학습은 `checkpoints/ckpt.pt`에서
자동으로 이어진다. 이게 "손 안 대도 되는" 핵심 장치다.

---

## 7. 모니터링 (뭘 보면 되나)

- **Discord**: 단계 시작/완료 알림 + `HEARTBEAT_MIN`(기본 30)분마다 "지금 로그
  마지막 줄 + GPU 사용률". 사전학습 중엔 여기 loss 값이 흘러간다.
- 파드에서 직접 보고 싶으면:

```bash
cd /workspace/minillm
tail -f logs/pipeline_*.log      # 최신 로그 실시간 (Ctrl+c로 빠져나옴)
nvidia-smi                       # GPU가 실제로 돌고 있는지 (사용률/메모리)
```

- GPU 사용률이 0%에서 안 올라오면 아직 데이터 다운로드/전처리 단계(CPU 작업)라
  정상일 수 있다. 사전학습에 들어가면 사용률이 확 오른다.

---

## 8. 끝난 뒤 — 결과 가져오기 & 과금 멈추기

파이프라인이 성공하면 Discord에 `🎉 전체 완료`가 오고, 산출물은:

- `checkpoints/ckpt_best.pt` — 사전학습된 "한국어의 결" 모델
- `checkpoints/sft.pt` — 대화 모델 (로컬 `chat.py`에 쓸 것)
- `tokenizer/tokenizer.json` — 진짜 토크나이저

**내 노트북으로 내려받기** (셋 중 편한 것):

1. **HuggingFace 백업을 켰다면**(`HF_TOKEN`/`HF_REPO`) — 이미 자동 업로드됨. 웹에서 받거나 `huggingface-cli download`.
2. **runpodctl send** — 파드에서 `runpodctl send checkpoints/sft.pt` 하면 코드가
   뜨고, 내 노트북에서 `runpodctl receive <코드>`로 받는다.
3. **RunPod 콘솔 파일 브라우저**로 직접 다운로드.

**과금 멈추기 (중요):**

- `POD_AUTO_STOP=true`를 넣었으면 파이프라인이 끝나며 **스스로 stop**한다.
- 아니면 RunPod 콘솔에서 파드 **Stop**을 직접 누른다. (Terminate는 파드 삭제 —
  데이터는 볼륨에 남지만, 굳이 지울 필요 없으면 Stop만.)
- **stop 후에도 80GB 볼륨 스토리지 비용은 소액 계속 나간다.** 완전히 안 쓸 거면
  결과를 내려받은 뒤 볼륨을 삭제해야 그 비용도 멈춘다.

---

## 9. (선택) 파드 재부팅 시 자동 재개까지

"접속조차 안 하고, 파드가 재시작되면 학습도 알아서 다시 시작"까지 원하면,
파드 템플릿의 **Container Start Command**(또는 Docker Command)에 아래를 넣는다.
파드가 부팅될 때마다 실행되고, 스크립트가 멱등적이라 끝난 단계는 건너뛴다.

```bash
bash -lc 'cd /workspace/minillm && git pull --ff-only; pip install -q -U datasets regex tqdm huggingface_hub; bash run_minillm_training.sh'
```

> 이건 편하지만, 처음엔 6장의 tmux 수동 실행으로 한 번 성공을 확인한 뒤에
> 켜는 걸 권한다(무엇이 도는지 눈으로 봐야 문제를 잡기 쉽다).

---

## 10. 트러블슈팅 & 자주 하는 실수

| 증상 | 원인/해결 |
|---|---|
| Discord 알림이 안 옴 | `DISCORD_WEBHOOK_URL` 오타/미설정. 터미널에서 `echo $DISCORD_WEBHOOK_URL` 확인. 스모크 테스트로 먼저 검증. |
| `No module named 'datasets'/'torch'` | 4장의 `pip install`을 안 했거나 PyTorch 아닌 이미지. `pip install -U datasets regex tqdm` 후 재실행. |
| 파드 지웠더니 다 사라짐 | Container Disk에 작업함. **반드시 `/workspace`(네트워크 볼륨) 안에서** 작업. |
| 처음부터 다시 하고 싶다 | 5장의 `.done` 마커 삭제 참고. |
| 사전학습이 너무 오래 걸림 | `TARGET_TOKENS`를 줄여 예산 축소, 또는 `MAX_DOCS`로 데이터 축소. |
| 접속 끊으니 학습이 죽음 | tmux 안에서 안 돌렸음. 6장대로 `tmux new -s train` 후 실행. |
| 크레딧 소진으로 멈춤 | 잔액 충전 후 다시 파드 켜고 `bash run_minillm_training.sh`로 재개. |
| GPU 사용률 0% | 아직 다운로드/전처리(CPU) 단계일 수 있음. 사전학습 들어가면 오름. |

---

## 한 장 요약 (체크리스트)

- [ ] Discord 웹훅 URL 만들기 (1-1)
- [ ] 80GB 볼륨 붙은 4090 파드를 PyTorch 이미지로 배포 (2)
- [ ] `DISCORD_WEBHOOK_URL`(+선택 HF/자동정지) 환경변수 등록 (3)
- [ ] 접속 → `/workspace`에서 clone + `pip install` (4)
- [ ] 스모크 테스트 5분 (5) → 마커 청소
- [ ] `tmux new -s train` → `bash run_minillm_training.sh` → `Ctrl+b d`로 떼기 (6)
- [ ] Discord로 진행 확인, 필요 시 `tmux attach` (7)
- [ ] `🎉 완료` 오면 결과 내려받고 파드 stop (8)
