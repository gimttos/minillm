#!/usr/bin/env bash
# =============================================================================
# minillm 자동 학습 파이프라인 — RunPod(무인) 런처
#
# 왜 이렇게 짰나:
#   - "손 하나 안 대고" 굴리려면 (1)어디서 끊겨도 이어서 시작하고(멱등성+resume),
#     (2)일정 시점마다 사람에게 상태를 알려주고(Discord), (3)끝나면 스스로
#     결과를 파드 밖으로 백업하고 파드를 멈춰야(비용) 한다. 이 세 가지가 이
#     스크립트의 전부다.
#   - 실제 저장소 CLI(python -m data.download / tokenizer.train_tokenizer /
#     data.pack / data.prepare_sft / train.pretrain / train.sft)에 정확히 맞춘다.
#     (예전 버전의 accelerate/configs/*.yaml 참조는 이 저장소에 존재하지 않아
#      전부 걷어냈다.)
#
# 설정은 전부 환경변수로 (RunPod 템플릿의 Environment Variables / Secrets):
#   DISCORD_WEBHOOK_URL   (거의 필수) 진행 알림을 받을 디스코드 웹훅 URL
#   HF_TOKEN, HF_REPO     (선택) 산출물을 파드 밖 HuggingFace로 영구 백업
#   RUNPOD_API_KEY, RUNPOD_POD_ID, POD_AUTO_STOP=true  (선택) 완료 시 파드 자동 정지
#   SOURCE                (기본 mix) 데이터 소스: wiki|namu|mix
#   MAX_DOCS              (기본 0=전체) 첫 스모크 테스트 때 3000 등 소량으로
#   SAMPLE_MB             (기본 200)  토크나이저 학습 표본 크기(MB)
#   TARGET_TOKENS         (기본 0=프리셋값 1B) 사전학습 토큰 예산 오버라이드
#   PRETRAIN_PRESET       (기본 full) full|full-loop|tiny  (tiny는 로컬 리허설용)
#   OPTIMIZER             (기본 muon) adamw|muon
#   SFT_ARGS              (기본 아래) SFT 마음 기제 조합 인자
#   HEARTBEAT_MIN         (기본 30) 학습 중 몇 분마다 진행상황을 디스코드로 보낼지
#
# 사용:  cd /workspace/minillm && bash run_minillm_training.sh
# =============================================================================

set -euo pipefail

# --------------------------- 설정(환경변수) ---------------------------------
cd "$(dirname "$0")"                       # 저장소 루트에서 실행 (경로 고정)

DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"                      # 예: gimttos/minillm-runs (미리 dataset repo 생성)
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"      # RunPod가 파드마다 자동 주입
POD_AUTO_STOP="${POD_AUTO_STOP:-false}"

SOURCE="${SOURCE:-mix}"
MAX_DOCS="${MAX_DOCS:-0}"
SAMPLE_MB="${SAMPLE_MB:-200}"
TARGET_TOKENS="${TARGET_TOKENS:-0}"
PRETRAIN_PRESET="${PRETRAIN_PRESET:-full}"
OPTIMIZER="${OPTIMIZER:-muon}"
# 기본 SFT: 대화형 + 가벼운 메타인지(conf)·잠재사고(latent). chat.py가 자동 인식.
SFT_ARGS="${SFT_ARGS:---latent 2 --conf}"
HEARTBEAT_MIN="${HEARTBEAT_MIN:-30}"

LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date +%Y%m%d-%H%M%S).log"
# 모든 stdout/stderr를 로그 파일에도 남긴다(파드가 죽어도 network volume에 흔적).
exec > >(tee -a "$LOG_FILE") 2>&1

MAX_DOCS_ARG=""; [[ "$MAX_DOCS" != "0" ]] && MAX_DOCS_ARG="--max-docs $MAX_DOCS"
TARGET_TOKENS_ARG=""; [[ "$TARGET_TOKENS" != "0" ]] && TARGET_TOKENS_ARG="--target-tokens $TARGET_TOKENS"

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 minillm 파이프라인 시작"
echo "   LOG_FILE   = $LOG_FILE"
echo "   SOURCE     = $SOURCE   MAX_DOCS=${MAX_DOCS}   SAMPLE_MB=${SAMPLE_MB}"
echo "   PRESET     = $PRETRAIN_PRESET   OPTIMIZER=$OPTIMIZER   TARGET_TOKENS=${TARGET_TOKENS}"
echo "   SFT_ARGS   = $SFT_ARGS"
echo "   AUTO_STOP  = $POD_AUTO_STOP"
echo "═══════════════════════════════════════════════════════════════"

# --------------------------- 의존성 자가 복구 -------------------------------
# RunPod 파드는 /workspace(네트워크 볼륨)만 영구다. pip 패키지는 컨테이너
# 디스크에 설치돼 파드를 껐다 켜면 사라진다 — 마커 덕에 앞 단계를 건너뛰고
# 재개하는 실행일수록 import가 처음 터지는 곳이 한참 뒤라서, 매 실행 시작에
# 확인하고 없으면 조용히 재설치한다. (torch는 PyTorch 이미지에 내장이라 생존)
if ! python3 -c "import regex, datasets, tqdm, numpy" 2>/dev/null; then
    echo "📦 의존성 재설치 (파드 재시작으로 컨테이너 디스크 초기화 감지)"
    pip install -q -U regex datasets tqdm numpy
fi
if [[ -n "$HF_TOKEN" ]] && ! python3 -c "import huggingface_hub" 2>/dev/null; then
    pip install -q -U huggingface_hub
fi

# --------------------------- Discord 유틸 -----------------------------------
# jq가 없어도 되도록 python3로 JSON 문자열을 안전하게 이스케이프한다(파이썬은 보장됨).
_json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

send_discord() {
    local msg="$1"
    [[ -z "$DISCORD_WEBHOOK_URL" ]] && return 0
    local payload
    payload="{\"content\": $(printf '%s' "$msg" | _json_escape)}"
    curl -sS -m 15 -H "Content-Type: application/json" -X POST \
         -d "$payload" "$DISCORD_WEBHOOK_URL" >/dev/null || true
}

# --------------------------- 진행 하트비트 ----------------------------------
# 학습(pretrain)은 한 번의 긴 프로세스라 단계 알림만으로는 몇 시간 조용하다.
# 그래서 백그라운드로 N분마다 "로그 마지막 줄 + GPU 사용률"을 디스코드로 보낸다.
heartbeat_loop() {
    local interval=$(( HEARTBEAT_MIN * 60 ))
    while true; do
        sleep "$interval"
        local last gpu
        last="$(tail -n 1 "$LOG_FILE" 2>/dev/null | cut -c1-300)"
        if command -v nvidia-smi >/dev/null 2>&1; then
            gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader 2>/dev/null | head -n1)"
        else
            gpu="n/a"
        fi
        send_discord "⏳ minillm 진행중 — GPU[$gpu]
\`\`\`
${last:-(로그 없음)}
\`\`\`"
    done
}

HEARTBEAT_PID=""
start_heartbeat() { heartbeat_loop & HEARTBEAT_PID=$!; }
stop_heartbeat()  { [[ -n "$HEARTBEAT_PID" ]] && kill "$HEARTBEAT_PID" 2>/dev/null || true; }

# --------------------------- 멱등성 가드 ------------------------------------
# 마커는 "성공했을 때만" 만든다. 그래야 중간에 죽어도 다음 실행에서 그 단계를
# 다시 시도한다. (부분 산출물로 오염될 수 있는 download/pack에 특히 중요)
run_stage() {
    local marker="$1"; local desc="$2"; shift 2
    if [[ -f "$marker" ]]; then
        echo "⏭️  [$desc] 이미 완료 (마커: $marker) — 건너뜀"; return 0
    fi
    echo "▶️  [$desc] 시작..."
    send_discord "▶️ [$desc] 시작"
    if "$@"; then
        touch "$marker"
        echo "✅ [$desc] 완료"
        send_discord "✅ [$desc] 완료"
    else
        echo "❌ [$desc] 실패"; return 1
    fi
}

# --------------------------- 파드 밖 백업 -----------------------------------
backup_offpod() {
    [[ -z "$HF_TOKEN" || -z "$HF_REPO" ]] && { echo "ℹ️  HF 백업 스킵(토큰/repo 없음)"; return 0; }
    echo "☁️  HuggingFace($HF_REPO)로 체크포인트/토크나이저 업로드..."
    local cli=""
    command -v hf >/dev/null 2>&1 && cli="hf"
    command -v huggingface-cli >/dev/null 2>&1 && cli="huggingface-cli"
    [[ -z "$cli" ]] && { echo "⚠️  huggingface_hub 미설치 → 스킵"; return 0; }
    local dst="run_$(date +%Y%m%d-%H%M%S)"
    if [[ "$cli" == "hf" ]]; then
        hf upload "$HF_REPO" checkpoints "$dst/checkpoints" --repo-type dataset || true
        hf upload "$HF_REPO" tokenizer/tokenizer.json "$dst/tokenizer.json" --repo-type dataset || true
    else
        huggingface-cli upload "$HF_REPO" checkpoints "$dst/checkpoints" --repo-type dataset --private || true
        huggingface-cli upload "$HF_REPO" tokenizer/tokenizer.json "$dst/tokenizer.json" --repo-type dataset --private || true
    fi
    echo "✅ HF 백업 완료 ($dst)"
}

# --------------------------- 종료 처리 --------------------------------------
cleanup() {
    local ec=$?
    stop_heartbeat
    echo ""; echo "🧹 종료 처리 (exit=$ec)"
    backup_offpod || true
    if [[ $ec -eq 0 ]]; then
        send_discord "🎉 **minillm 파이프라인 전체 완료!** 체크포인트: \`checkpoints/sft.pt\`, \`checkpoints/ckpt_best.pt\`"
    else
        send_discord "❌ **minillm 파이프라인 중단** (exit $ec)
마지막 로그:
\`\`\`
$(tail -n 15 "$LOG_FILE" 2>/dev/null)
\`\`\`
파드는 아직 켜져 있으니 접속해 \`bash run_minillm_training.sh\`로 이어서 재개하세요."
    fi
    # 성공/실패와 무관하게, 요청 시 파드 정지(볼륨 스토리지 비용만 남고 GPU 과금 중단).
    if [[ "$POD_AUTO_STOP" == "true" && -n "$RUNPOD_POD_ID" ]] && command -v runpodctl >/dev/null 2>&1; then
        echo "🛑 파드 자동 정지 요청: $RUNPOD_POD_ID"
        runpodctl stop pod "$RUNPOD_POD_ID" || true
    fi
}
trap cleanup EXIT INT TERM

# ============================== 파이프라인 ==================================
send_discord "🚀 minillm 학습 시작 (pod: ${RUNPOD_POD_ID:-local}, preset: $PRETRAIN_PRESET)"
start_heartbeat

# 1) 데이터 다운로드 → data/raw/corpus.txt
run_stage "data/raw/.download.done" "1·데이터 다운로드" \
    python -m data.download --source "$SOURCE" --out data/raw/corpus.txt $MAX_DOCS_ARG

# 2) 토크나이저 학습 → tokenizer/tokenizer.json
run_stage "tokenizer/.tokenizer.done" "2·토크나이저 학습" \
    python -m tokenizer.train_tokenizer --input data/raw/corpus.txt \
        --sample-mb "$SAMPLE_MB" --out tokenizer/tokenizer.json

# 3) 패킹 → data/bin/train.bin, val.bin
run_stage "data/bin/.pack.done" "3·데이터 패킹" \
    python -m data.pack --input data/raw/corpus.txt \
        --tokenizer tokenizer/tokenizer.json --out-dir data/bin

# 4) SFT 데이터 준비 → data/bin/sft.npz (사전학습과 병렬 무관, 미리 만들어 둠)
run_stage "data/bin/.sft_prepared.done" "4·SFT 데이터 준비" \
    python -m data.prepare_sft --tokenizer tokenizer/tokenizer.json --out data/bin/sft.npz

# 5) 사전학습 (base). --resume는 ckpt.pt가 있을 때만 실제로 재개하므로 항상 안전.
#    중간에 파드가 죽어도 이 스크립트를 다시 실행하면 여기서 이어서 학습한다.
echo ""; echo "🧠 [5·사전학습] preset=$PRETRAIN_PRESET"
if [[ ! -f "checkpoints/.pretrain.done" ]]; then
    send_discord "🧠 [5·사전학습] 시작 (preset $PRETRAIN_PRESET, opt $OPTIMIZER). 이 단계가 가장 오래 걸립니다."
    python -m train.pretrain --preset "$PRETRAIN_PRESET" --optimizer "$OPTIMIZER" \
        --resume $TARGET_TOKENS_ARG
    touch checkpoints/.pretrain.done
    echo "✅ [5·사전학습] 완료"
    send_discord "✅ [5·사전학습] 완료 → checkpoints/ckpt_best.pt"
else
    echo "⏭️  [5·사전학습] 이미 완료 — 건너뜀"
fi

# 6) SFT (base → 대화 모델). 마음 기제 조합은 SFT_ARGS로.
run_stage "checkpoints/.sft.done" "6·SFT(대화 모델)" \
    bash -c "python -m train.sft --init checkpoints/ckpt_best.pt \
        --data data/bin/sft.npz --out checkpoints/sft.pt $SFT_ARGS"

echo ""; echo "🏁 파이프라인 정상 종료"
exit 0
