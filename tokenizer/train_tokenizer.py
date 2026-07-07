"""토크나이저 학습 스크립트.

data/raw/ 에 준비된 텍스트에서 일부를 샘플링해 BPE 사전을 학습한다.
사전은 코퍼스 전체가 아니라 대표 샘플로만 학습해도 충분하다 —
자주 나오는 패턴의 "통계"만 필요하기 때문이다.

사용법:
    python -m tokenizer.train_tokenizer --input data/raw/corpus.txt \
        --vocab-size 16384 --sample-mb 100 --out tokenizer/tokenizer.json
"""

import argparse
import time
from pathlib import Path

from tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="학습용 텍스트 파일")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-mb", type=float, default=100,
                    help="사전 학습에 사용할 텍스트 양(MB). 파일 앞부분에서 읽음")
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    n_bytes = int(args.sample_mb * 1024 * 1024)
    with open(args.input, encoding="utf-8", errors="ignore") as f:
        text = f.read(n_bytes)
    print(f"샘플 {len(text):,}자 로드. BPE 학습 시작 (vocab={args.vocab_size})")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size, verbose=True)
    print(f"학습 완료: {time.time() - t0:.0f}초")

    tok.save(args.out)
    print(f"저장: {args.out}")

    # 결과 확인: 실제 한국어 문장이 어떻게 토큰화되는지 눈으로 본다
    demo = "안녕하세요! 저는 밑바닥부터 만든 작은 언어 모델입니다."
    ids = tok.encode(demo)
    pieces = [tok.decode([i]) for i in ids]
    print(f"\n예시: {demo}")
    print(f"토큰 수: {len(ids)} (글자 수 {len(demo)})")
    print(f"조각: {pieces}")


if __name__ == "__main__":
    main()
