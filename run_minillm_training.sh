#!/usr/bin/env bash
# =============================================================================
# minillm 자동 학습 파이프라인 — RunPod(무인) 런처
#
# "손 하나 안 대고" 굴리기 위한 세 가지:
#   (1) 어디서 끊겨도 이어서 시작 — 멱등성 마커 + pretrain --resume
#   (2) 일정 시점마다 사람에게 알림 — Discord 웹훅(단계 + N분 하트비트)
#   (3) 끝나면 스스로 백업하고 파드를 멈춤 — HF 업로드 + runpodctl stop (과금 차단)
#
# 파이프라인 (각 단계는 마커가 있으면 건너뛴다):
#   1 데이터 다운로드 -> 2 토크나이저 -> 3 패킹        (base 코퍼스)
#   4 페르소나 대화 변환 -> 5 SFT 데이터(context/workspace)
#   6 사전학습(large) -> 7 SFT (변형별로 하나씩)
#
# 설정은 전부 환경변수로 (RunPod Environment Variables / Secrets):
#   DISCORD_WEBHOOK_URL  (거의 필수) 진행 알림 웹훅
#   HF_TOKEN, HF_REPO    (선택) 파드 밖 HuggingFace 영구 백업
#   POD_AUTO_STOP=true   (선택) 완료 시 파드 자동 정지 — 과금이 여기서 멈춘다
#   RUNPOD_API_KEY       (선택) 자동 정지에 필요 (runpodctl config --apiKey)
#
#   PRETRAIN_PRESET  (기본 large) large|full|tiny
#   OPTIMIZER        (기본 muon)
#   TARGET_TOKENS    (기본 0 = 프리셋 값)
#   PERSONA_DIR      (기본 persona_data) 페르소나 대화 zip들이 있는 폴더
#   SFT_VARIANTS     (기본 "context workspace") 돌릴 SFT 변형들. 공백 구분.
#                    context=페르소나를 <|sys|> 문맥으로 / workspace=GWT 슬롯으로
#   SFT_EPOCHS       (기본 2)
#   SFT_MIND         (기본 "--latent 2 --conf --workspace-slots 4") 마음 기제
#   HEARTBEAT_MIN    (기본 30) 진행 알림 주기(분)
#
# 사용:  cd /workspace/minillm && bash run_minillm_training.sh
#   (이미 사전학습이 따로 돌고 있다면 그 프로세스를 먼저 끄고 실행할 것 —
#    이 스크립트가 --resume 으로 마지막 저장 지점부터 이어받는다.)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"                       # 저장소 루트 고정

DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"         # RunPod가 파드마다 자동 주입
POD_AUTO_STOP="${POD_AUTO_STOP:-false}"

SOURCE="${SOURCE:-mix}"
MAX_DOCS="${MAX_DOCS:-0}"
SAMPLE_MB="${SAMPLE_MB:-200}"
PRETRAIN_PRESET="${PRETRAIN_PRESET:-large}"
OPTIMIZER="${OPTIMIZER:-muon}"
TARGET_TOKENS="${TARGET_TOKENS:-0}"

PERSONA_DIR="${PERSONA_DIR:-persona_data}"
SFT_VARIANTS="${SFT_VARIANTS:-context workspace}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_MIND="${SFT_MIND:---latent 2 --conf --workspace-slots 4}"
HEARTBEAT_MIN="${HEARTBEAT_MIN:-30}"

# 프리셋마다 out_dir이 다르다 (train/config.py) — best 체크포인트를 여기서 찾는다.
if [[ "$PRETRAIN_PRESET" == "large" ]]; then CKPT_DIR="checkpoints_large"; else CKPT_DIR="checkpoints"; fi
BASE_CKPT="$CKPT_DIR/ckpt_best.pt"

LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1          # 파드가 죽어도 볼륨에 흔적이 남는다

MAX_DOCS_ARG=""; [[ "$MAX_DOCS" != "0" ]] && MAX_DOCS_ARG="--max-docs $MAX_DOCS"
TARGET_ARG="";   [[ "$TARGET_TOKENS" != "0" ]] && TARGET_ARG="--target-tokens $TARGET_TOKENS"

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 minillm 파이프라인"
echo "   PRESET=$PRETRAIN_PRESET ($CKPT_DIR)  OPTIMIZER=$OPTIMIZER"
echo "   SFT_VARIANTS=$SFT_VARIANTS  EPOCHS=$SFT_EPOCHS"
echo "   SFT_MIND=$SFT_MIND"
echo "   AUTO_STOP=$POD_AUTO_STOP   LOG=$LOG_FILE"
echo "═══════════════════════════════════════════════════════════════"

# --------------------------- 의존성 자가 복구 -------------------------------
# 파드를 껐다 켜면 pip 패키지가 사라진다(/workspace만 영구). 마커로 앞 단계를
# 건너뛰는 재개 실행일수록 import 실패가 한참 뒤에야 터지므로 미리 확인한다.
if ! python3 -c "import regex, datasets, tqdm, numpy" 2>/dev/null; then
    echo "📦 의존성 재설치 (컨테이너 디스크 초기화 감지)"
    pip install -q -U regex datasets tqdm numpy
fi
if [[ -n "$HF_TOKEN" ]] && ! python3 -c "import huggingface_hub" 2>/dev/null; then
    pip install -q -U huggingface_hub
fi

# --------------------------- Discord ----------------------------------------
_json() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }
send_discord() {
    [[ -z "$DISCORD_WEBHOOK_URL" ]] && return 0
    curl -sS -m 15 -H "Content-Type: application/json" -X POST \
         -d "{\"content\": $(printf '%s' "$1" | _json)}" \
         "$DISCORD_WEBHOOK_URL" >/dev/null || true
}

# 학습은 한 번의 긴 프로세스라 단계 알림만으로는 몇 시간 조용하다.
# N분마다 "로그 마지막 줄 + GPU"를 보내 살아 있음을 알린다.
heartbeat_loop() {
    while true; do
        sleep $(( HEARTBEAT_MIN * 60 ))
        local last gpu
        last="$(tail -n 1 "$LOG_FILE" 2>/dev/null | cut -c1-300)"
        gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
               --format=csv,noheader 2>/dev/null | head -n1 || echo n/a)"
        send_discord "⏳ 진행중 — GPU[$gpu]
\`\`\`
${last:-(로그 없음)}
\`\`\`"
    done
}
HEARTBEAT_PID=""
stop_heartbeat() { [[ -n "$HEARTBEAT_PID" ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true; }

# --------------------------- 멱등성 가드 ------------------------------------
# 마커는 성공했을 때만 만든다 — 중간에 죽으면 다음 실행에서 그 단계를 다시 한다.
run_stage() {
    local marker="$1" desc="$2"; shift 2
    if [[ -f "$marker" ]]; then echo "⏭️  [$desc] 완료됨 — 건너뜀"; return 0; fi
    echo "▶️  [$desc] 시작..."; send_discord "▶️ [$desc] 시작"
    if "$@"; then
        touch "$marker"; echo "✅ [$desc] 완료"; send_discord "✅ [$desc] 완료"
    else
        echo "❌ [$desc] 실패"; return 1
    fi
}

# --------------------------- 백업 / 종료 ------------------------------------
backup_offpod() {
    [[ -z "$HF_TOKEN" || -z "$HF_REPO" ]] && { echo "ℹ️  HF 백업 스킵"; return 0; }
    local cli=""
    command -v hf >/dev/null 2>&1 && cli="hf"
    command -v huggingface-cli >/dev/null 2>&1 && cli="huggingface-cli"
    [[ -z "$cli" ]] && { echo "⚠️  huggingface_hub 없음 → 스킵"; return 0; }
    local dst="run_$(date +%Y%m%d-%H%M%S)"
    echo "☁️  HF($HF_REPO) 업로드..."
    "$cli" upload "$HF_REPO" "$CKPT_DIR" "$dst/$CKPT_DIR" --repo-type dataset || true
    "$cli" upload "$HF_REPO" tokenizer/tokenizer.json "$dst/tokenizer.json" --repo-type dataset || true
    echo "✅ HF 백업 완료 ($dst)"
}

cleanup() {
    local ec=$?
    stop_heartbeat
    echo ""; echo "🧹 종료 처리 (exit=$ec)"
    backup_offpod || true
    if [[ $ec -eq 0 ]]; then
        send_discord "🎉 **전체 완료!** 체크포인트: \`$CKPT_DIR/\`
$(ls -1 "$CKPT_DIR"/*.pt 2>/dev/null | sed 's|^|  • |' || true)"
    else
        send_discord "❌ **중단** (exit $ec)
\`\`\`
$(tail -n 15 "$LOG_FILE" 2>/dev/null)
\`\`\`
파드는 켜져 있습니다 — \`bash run_minillm_training.sh\`로 이어서 재개하세요."
    fi
    # 성공/실패와 무관하게 요청 시 파드를 멈춘다. GPU 과금이 여기서 끊긴다.
    # (볼륨 보관료는 계속 나가므로, 다 쓴 뒤엔 볼륨도 정리할 것)
    #
    # 자동 정지가 "조용히" 실패하면 밤새 GPU 과금이 흐른다 — 그게 제일 나쁘다.
    # 그래서 실패하면 Discord로 크게 알려 사람이 직접 끄게 한다.
    if [[ "$POD_AUTO_STOP" != "true" ]]; then
        send_discord "💡 자동 정지가 꺼져 있습니다 — **콘솔에서 파드를 Stop 하세요** (GPU 과금 중)."
    elif [[ "${AUTO_STOP_READY:-no}" != "yes" || -z "$RUNPOD_POD_ID" ]]; then
        # 시작 시점의 사전 점검에서 이미 불가로 판정된 경우
        send_discord "⚠️ **자동 정지 불가** (runpodctl 인증/POD_ID 문제).
👉 **콘솔에서 직접 파드를 Stop 하세요 — 안 그러면 GPU 과금이 계속됩니다.**"
    else
        echo "🛑 파드 자동 정지 시도: $RUNPOD_POD_ID"
        if runpodctl stop pod "$RUNPOD_POD_ID"; then
            send_discord "🛑 파드를 정지했습니다 (GPU 과금 중단)."
        else
            send_discord "⚠️ **자동 정지 실패!**
👉 **콘솔에서 직접 파드를 Stop 하세요 — 안 그러면 GPU 과금이 계속됩니다.**"
        fi
    fi
}
trap cleanup EXIT INT TERM

# --------------------------- 자동 정지 사전 점검 ----------------------------
# 9시간 뒤 종료 시점에 "정지 못 함"을 알게 되면 늦다 — 시작할 때 미리 확인한다.
# RUNPOD_API_KEY가 환경에 있으면 여기서 runpodctl에 등록해 준다 (사람이 따로
# runpodctl config를 칠 필요 없이 export 하나로 끝나게).
AUTO_STOP_READY="no"
if [[ "$POD_AUTO_STOP" == "true" ]]; then
    if ! command -v runpodctl >/dev/null 2>&1; then
        echo "⚠️  runpodctl 없음 → 자동 정지 불가 (끝나면 콘솔에서 직접 Stop)"
    else
        if [[ -n "$RUNPOD_API_KEY" ]] && ! runpodctl get pod >/dev/null 2>&1; then
            echo "🔑 RUNPOD_API_KEY로 runpodctl 등록 중..."
            # SSH 키 동기화가 실패해도 config 자체는 저장되므로 실패를 무시한다
            runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true
        fi
        if runpodctl get pod >/dev/null 2>&1; then
            AUTO_STOP_READY="yes"
            echo "✅ 자동 정지 준비됨 (완료 시 파드가 스스로 꺼집니다)"
        else
            echo "⚠️  runpodctl 인증 실패 → 자동 정지 불가"
            echo "    키가 유효한지, 권한이 Read/Write 인지 확인하세요"
            echo "    (RunPod > Settings > API Keys). 완료 시 콘솔에서 직접 Stop 하면 됩니다."
        fi
    fi
fi

# ============================== 파이프라인 ==================================
send_discord "🚀 minillm 시작 (preset $PRETRAIN_PRESET, pod ${RUNPOD_POD_ID:-local})"
heartbeat_loop & HEARTBEAT_PID=$!

# --- base 코퍼스 (이미 있으면 마커로 전부 건너뛴다) ---
run_stage "data/raw/.download.done" "1·데이터 다운로드" \
    python -m data.download --source "$SOURCE" --out data/raw/corpus.txt $MAX_DOCS_ARG

run_stage "tokenizer/.tokenizer.done" "2·토크나이저" \
    python -m tokenizer.train_tokenizer --input data/raw/corpus.txt \
        --sample-mb "$SAMPLE_MB" --out tokenizer/tokenizer.json

run_stage "data/bin/.pack.done" "3·패킹" \
    python -m data.pack --input data/raw/corpus.txt \
        --tokenizer tokenizer/tokenizer.json --out-dir data/bin

# --- 페르소나 대화 -> SFT 데이터 (변형마다 npz 하나) ---
run_stage "data/raw/.persona.done" "4·페르소나 대화 변환" \
    python -m data.convert_persona --input "$PERSONA_DIR" --out data/raw/persona.jsonl

for v in $SFT_VARIANTS; do
    run_stage "data/bin/.sft_$v.done" "5·SFT 데이터($v)" \
        python -m data.prepare_sft --tokenizer tokenizer/tokenizer.json \
            --conversations data/raw/persona.jsonl --mirror \
            --persona-mode "$v" --out "data/bin/sft_$v.npz"
done

# --- 사전학습 (--resume은 ckpt.pt가 있을 때만 재개하므로 항상 안전) ---
echo ""; echo "🧠 [6·사전학습] $PRETRAIN_PRESET"
if [[ ! -f "$CKPT_DIR/.pretrain.done" ]]; then
    send_discord "🧠 [6·사전학습] 시작 ($PRETRAIN_PRESET). 가장 오래 걸리는 단계입니다."
    python -m train.pretrain --preset "$PRETRAIN_PRESET" --optimizer "$OPTIMIZER" \
        --resume $TARGET_ARG
    touch "$CKPT_DIR/.pretrain.done"
    echo "✅ [6·사전학습] 완료"
    send_discord "✅ [6·사전학습] 완료 → $BASE_CKPT"
else
    echo "⏭️  [6·사전학습] 완료됨 — 건너뜀"
fi

[[ -f "$BASE_CKPT" ]] || { echo "❌ $BASE_CKPT 가 없습니다"; exit 1; }

# --- SFT: 변형별로 (같은 base에서 갈라 학습 -> val loss 비교가 곧 실험) ---
for v in $SFT_VARIANTS; do
    run_stage "$CKPT_DIR/.sft_$v.done" "7·SFT($v)" \
        bash -c "python -m train.sft --init '$BASE_CKPT' \
            --data 'data/bin/sft_$v.npz' --out '$CKPT_DIR/sft_$v.pt' \
            --epochs $SFT_EPOCHS $SFT_MIND"
done

echo ""; echo "🏁 파이프라인 정상 종료"
exit 0
