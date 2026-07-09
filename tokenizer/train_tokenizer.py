"""토크나이저 학습 스크립트.

data/raw/ 에 준비된 텍스트에서 일부를 샘플링해 BPE 사전을 학습한다.
사전은 코퍼스 전체가 아니라 대표 샘플로만 학습해도 충분하다 —
자주 나오는 패턴의 "통계"만 필요하기 때문이다.

단, 파일 **앞부분**만 읽으면 사전이 앞쪽 문서(예: 나무위키 표제어 순)에
편향된다(§A6). 그래서 코퍼스 전역에서 랜덤 오프셋으로 여러 청크를 뽑아
합친다. 각 청크는 DOC_SEP 경계로 정렬해 문서 중간이 잘리지 않게 한다.

사용법:
    python -m tokenizer.train_tokenizer --input data/raw/corpus.txt \
        --vocab-size 16384 --sample-mb 100 --out tokenizer/tokenizer.json
"""

import argparse
import os
import random
import time

from tokenizer.bpe import BPETokenizer

# data/download.py 의 DOC_SEP 과 동일해야 한다 (문서 경계 표식).
DOC_SEP = "\n\n<<<DOC>>>\n\n"


def read_random_sample(path: str, n_bytes: int, seed: int = 1337,
                       chunk_bytes: int = 1 << 20) -> str:
    """코퍼스 전역에서 랜덤 오프셋으로 청크들을 읽어 합쳐 대표 샘플을 만든다.

    각 청크는 임의 위치에서 시작하므로 앞쪽 문서 중간을 자를 수 있다 —
    첫 DOC_SEP 이후부터 취해 잘린 조각을 버리고 온전한 문서만 남긴다.
    시드 고정으로 재현 가능.
    """
    size = os.path.getsize(path)
    if size <= n_bytes:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    rng = random.Random(seed)
    parts, total = [], 0
    with open(path, "rb") as f:
        while total < n_bytes:
            off = rng.randint(0, max(size - chunk_bytes, 0))
            f.seek(off)
            text = f.read(chunk_bytes).decode("utf-8", "ignore")
            i = text.find(DOC_SEP)                # 문서 경계로 정렬
            if i != -1:
                text = text[i + len(DOC_SEP):]
            parts.append(text)
            total += len(text.encode("utf-8"))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="학습용 텍스트 파일")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-mb", type=float, default=100,
                    help="사전 학습에 사용할 텍스트 양(MB). 코퍼스 전역 랜덤 샘플")
    ap.add_argument("--seed", type=int, default=1337, help="랜덤 샘플 시드")
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    n_bytes = int(args.sample_mb * 1024 * 1024)
    text = read_random_sample(args.input, n_bytes, seed=args.seed)
    print(f"랜덤 샘플 {len(text):,}자 로드. BPE 학습 시작 (vocab={args.vocab_size})")

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
