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

Decoder-only Transformer (미니 Llama): vocab 16,384 · d_model 512 ·
8 layers · 8 heads · SwiGLU · RoPE · RMSNorm · context 512 → 약 30M 파라미터
(fp32 체크포인트 ≈ 120MB).

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
- 사전학습: T4 기준 위키 1회독 12~18시간 (여러 세션에 나눠서)
- SFT: 30분~1시간

### 3. 로컬에서 대화
클라우드에서 만든 `checkpoints/sft.pt`와 `tokenizer/tokenizer.json`을 내려받고:
```bash
python chat.py --ckpt checkpoints/sft.pt
# SFT 전에 "이어쓰기"만 시험해 보려면:
python chat.py --ckpt checkpoints/ckpt_best.pt --raw
```

> Windows 콘솔에서 한글이 깨지면 `set PYTHONUTF8=1` 후 실행.

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
