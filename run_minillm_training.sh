#!/bin/bash
# =============================================================================
# minillm Training Pipeline - RunPod / Kaggle robust launcher
# v2 (피드백 5개 모두 반영)
#
# 사용법:
#   1. RunPod 템플릿에서 secret으로 HF_TOKEN, DISCORD_WEBHOOK_URL 설정
#   2. Pod 시작 시 git clone + pip install은 템플릿이나 startup script에서 미리
#   3. 이 스크립트 실행: bash run_minillm_training.sh
#   4. tiny 드라이런 먼저: max-docs 1000, pretrain max_steps 100 정도로 테스트
# =============================================================================

set -euo pipefail

# ----------------------------- CONFIG ---------------------------------------
HF_REPO="gimttos/minillm-runs"          # private dataset repo 미리 만들어 두세요
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
POD_AUTO_STOP="${POD_AUTO_STOP:-false}"
HF_TOKEN="${HF_TOKEN:-}"                # RunPod secret으로 주입 추천

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/minillm_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

BACKUP_DIR="/workspace/minillm-runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 minillm Training Pipeline 시작"
echo "   LOG_FILE     = $LOG_FILE"
echo "   BACKUP_DIR   = $BACKUP_DIR"
echo "   POD_AUTO_STOP= $POD_AUTO_STOP"
echo "═══════════════════════════════════════════════════════════════"

# --------------------------- FUNCTIONS --------------------------------------

send_discord() {
    local msg="$1"
    if [[ -n "$DISCORD_WEBHOOK_URL" ]]; then
        curl -sS -H "Content-Type: application/json" \
             -X POST \
             -d "{\"content\": $(printf '%s' "$msg" | jq -Rs .)}" \
             "$DISCORD_WEBHOOK_URL" || true
    fi
}

backup_important_files() {
    echo ""
    echo "📦 중요 산출물 백업 시작 → $BACKUP_DIR"
    
    mkdir -p "$BACKUP_DIR"/{checkpoints,logs,configs,data,scripts}
    
    # Pretrain + SFT 체크포인트 전부 잡기 (best/periodic 모두)
    # ls -t checkpoints/*.pt 방식보다 robust
    find checkpoints -name "*.pt" -exec cp --parents {} "$BACKUP_DIR/" \; 2>/dev/null || true
    find sft_checkpoints -name "*.pt" -exec cp --parents {} "$BACKUP_DIR/" \; 2>/dev/null || true
    find . -maxdepth 2 -name "ckpt_*.pt" -exec cp --parents {} "$BACKUP_DIR/" \; 2>/dev/null || true
    
    # logs, configs, tokenizer 등
    cp -r logs "$BACKUP_DIR/" 2>/dev/null || true
    cp configs/*.yaml "$BACKUP_DIR/configs/" 2>/dev/null || true
    cp -r data/processed/*.json data/tokenizer/ "$BACKUP_DIR/data/" 2>/dev/null || true
    cp tokenizer.json "$BACKUP_DIR/" 2>/dev/null || true
    
    # 소스 스크립트도 가볍게
    cp pretrain.py sft.py prepare_sft.py run_minillm_training.sh "$BACKUP_DIR/scripts/" 2>/dev/null || true
    
    echo "✅ 로컬 백업 완료"
    
    # === 핵심 수정 #1: 파드 밖(HF)으로 영구 백업 ===
    if [[ -n "$HF_TOKEN" ]]; then
        echo "☁️  HF private repo로 업로드 중... ($HF_REPO)"
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli upload "$HF_REPO" "$BACKUP_DIR" "$(basename "$BACKUP_DIR")" \
                --repo-type dataset --private || true
        elif command -v hf &>/dev/null; then
            hf upload "$HF_REPO" "$BACKUP_DIR" "$(basename "$BACKUP_DIR")" || true
        else
            echo "⚠️  huggingface-cli / hf 명령어를 찾을 수 없음. pip install huggingface_hub 해두세요."
        fi
    else
        echo "⚠️  HF_TOKEN 없음 → HF 업로드 스킵 (로컬 백업만 남음)"
    fi
}

# 멱등성 가드 (성공 시에만 마커 생성) — 핵심 수정 #3
run_if_not_exists() {
    local marker="$1"
    local desc="$2"
    shift 2
    local cmd="$*"
    
    if [[ -f "$marker" ]]; then
        echo "⏭️  $desc — 이미 완료됨 (마커 존재)"
        return 0
    fi
    
    echo "▶️  $desc 시작..."
    if eval "$cmd"; then
        touch "$marker"
        echo "✅ $desc 완료 → 마커 생성: $marker"
    else
        echo "❌ $desc 실패"
        return 1
    fi
}

cleanup() {
    local ec=$?          # 핵심 수정 #4: 종료 코드 먼저 잡기
    echo ""
    echo "🧹 Cleanup 실행 (exit code: $ec)"
    
    backup_important_files || true
    
    if [[ $ec -ne 0 ]]; then
        send_discord "❌ minillm 파이프라인 **중단** (exit $ec)\n로그: \`$LOG_FILE\`"
    else
        send_discord "✅ minillm 파이프라인 **성공 완료**!\n로그: \`$LOG_FILE\`"
    fi
    
    if [[ "$POD_AUTO_STOP" == "true" && -n "$RUNPOD_POD_ID" ]]; then
        echo "🛑 Pod auto-stop 요청..."
        runpodctl stop pod "$RUNPOD_POD_ID" || true
    fi
}

trap cleanup EXIT INT TERM

# ----------------------------- MAIN -----------------------------------------

send_discord "🚀 minillm training 시작 (pod: ${RUNPOD_POD_ID:-로컬})"

# 1. 데이터 다운로드 (부분 실패 시 재시도되도록 마커를 성공 후에만)
run_if_not_exists "data/raw/.download.done" "Data Download" \
    'python -m data.download --source mix --max-docs 50000'

# 2. 토크나이저 (atomic 파일이라 기존 방식도 OK지만 일관되게 마커 사용)
run_if_not_exists "data/processed/.tokenizer.done" "Tokenizer Training" \
    'python scripts/train_tokenizer.py --output_dir data/processed'

# 3. 데이터 패킹 (필요 시 주석 해제, bin 파일은 partial 위험하니 마커 필수)
# run_if_not_exists "data/processed/.pack.done" "Data Packing" \
#     'python -m data.pack --input data/raw/corpus.txt --output data/processed/train.bin --max-seq-len 2048'

# 4. Pretrain (핵심 수정 #2: resume 버그 해결)
echo ""
echo "🧠 Pretraining 단계"

# 어떤 체크포인트라도 있으면 resume (best가 아직 안 만들어졌어도 periodic으로 resume)
if find checkpoints -name "*.pt" 2>/dev/null | grep -q .; then
    RESUME_ARG="--resume"
    echo "   기존 체크포인트 발견 → --resume 모드 진입"
else
    RESUME_ARG=""
    echo "   fresh start"
fi

# PRETRAIN_CMD를 여기서 즉시 구성 (루프 밖에서 미리 정해두는 실수 방지)
accelerate launch --config_file configs/accelerate.yaml pretrain.py \
    --config configs/pretrain.yaml \
    $RESUME_ARG \
    --output_dir checkpoints \
    --logging_dir logs \
    --report_to tensorboard,wandb \
    2>&1 | tee -a "$LOG_FILE"

echo "✅ Pretrain 단계 종료"

# 5. SFT 단계 (prepare_sft + sft 학습)
echo ""
echo "🎯 SFT 단계"

# prepare_sft가 별도라면 여기서 실행 (멱등성 가드 추천)
# run_if_not_exists "data/processed/.sft_prepared.done" "SFT Data Prepare" \
#     'python scripts/prepare_sft.py --pretrain_dir checkpoints --output_dir data/sft'

# SFT 학습 (체크포인트 경로가 checkpoints/ 안에 들어가도록 맞춰두세요)
# accelerate launch sft.py --config configs/sft.yaml --output_dir checkpoints/sft ...

echo "✅ SFT 단계 종료 (구현에 따라 추가)"

# 최종 성공 메시지
send_discord "🎉 모든 파이프라인 완료! HF에 백업 업로드 확인하세요.\n로그: \`$LOG_FILE\`"

echo ""
echo "🏁 Pipeline 정상 종료"
exit 0
