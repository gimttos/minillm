"""사전학습용 한국어 코퍼스 다운로드 → data/raw/corpus.txt 로 저장.

기본: 한국어 위키피디아 (wikimedia/wikipedia, 약 1.4GB 텍스트).
문서 사이는 <|endoftext|> 경계로 학습 시 구분되도록 빈 줄 2개 + 구분자를 넣지 않고,
pack.py 단계에서 문서 단위로 endoftext 토큰을 삽입하므로 여기서는
"한 줄에 문서 하나" 형태 대신 문서를 \n\n<<<DOC>>>\n\n 로 구분해 저장한다.

사용법:
    python -m data.download                      # 위키 전체
    python -m data.download --max-docs 5000      # 로컬 검증용 소량
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DOC_SEP = "\n\n<<<DOC>>>\n\n"  # pack.py가 이 구분자로 문서를 나눈다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/corpus.txt")
    ap.add_argument("--max-docs", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # streaming=True: 전체를 메모리/디스크에 받지 않고 문서 단위로 흘려 받는다
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko",
                      split="train", streaming=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in tqdm(ds, desc="downloading"):
            text = doc["text"].strip()
            if len(text) < 200:  # 너무 짧은 문서(리다이렉트 등)는 버림
                continue
            f.write(text)
            f.write(DOC_SEP)
            n += 1
            if args.max_docs and n >= args.max_docs:
                break

    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"완료: {n:,}개 문서, {size_mb:.0f}MB -> {args.out}")


if __name__ == "__main__":
    main()
