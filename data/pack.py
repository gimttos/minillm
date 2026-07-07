"""코퍼스를 토큰화해 학습용 바이너리(.bin)로 변환.

모델은 텍스트가 아니라 토큰 ID의 긴 나열을 학습한다. 매 스텝마다
토큰화를 반복하면 낭비이므로, 한 번 토큰화해서 uint16 배열로 저장해 두고
학습 때는 np.memmap으로 원하는 위치만 잘라 읽는다 (RAM 절약).

- 문서 경계마다 <|endoftext|> 토큰을 넣는다. 모델은 이 토큰을 보고
  "여기서 문맥이 끊긴다"를 배운다.
- 200개 문서마다 1개는 검증(val)용으로 떼어 둔다 — 본 적 없는 텍스트에
  대한 loss가 진짜 실력이다.
- 코퍼스가 GB 단위여도 돌 수 있게 문서 단위로 스트리밍 처리한다.

사용법:
    python -m data.pack --input data/raw/corpus.txt --tokenizer tokenizer/tokenizer.json
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.download import DOC_SEP
from tokenizer.bpe import BPETokenizer

FLUSH_EVERY = 8_000_000  # 토큰이 이만큼 쌓이면 디스크로 내보냄 (~16MB)


def iter_docs(path: str):
    """파일 전체를 메모리에 올리지 않고 문서를 하나씩 흘려보낸다."""
    buf = ""
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            *docs, buf = buf.split(DOC_SEP)
            yield from (d for d in docs if d.strip())
    if buf.strip():
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/corpus.txt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", default="data/bin")
    ap.add_argument("--val-every", type=int, default=200, help="N개 문서당 1개를 val로")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    eot = tok.encode_special("<|endoftext|>")
    assert tok.vocab_size <= 65536, "uint16에 담으려면 vocab이 65536 이하여야 함"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    buffers: dict[str, list[int]] = {"train": [], "val": []}
    files = {k: open(out_dir / f"{k}.bin", "wb") for k in counts}

    def flush(split: str):
        np.array(buffers[split], dtype=np.uint16).tofile(files[split])
        counts[split] += len(buffers[split])
        buffers[split].clear()

    for i, doc in enumerate(tqdm(iter_docs(args.input), desc="tokenizing")):
        # 가벼운 정규화: 줄 끝 공백·빈 줄 정리
        doc = "\n".join(line.strip() for line in doc.splitlines() if line.strip())
        split = "val" if i % args.val_every == 0 else "train"
        buffers[split].extend(tok.encode(doc))
        buffers[split].append(eot)
        if len(buffers[split]) >= FLUSH_EVERY:
            flush(split)

    for split in files:
        flush(split)
        files[split].close()
    print(f"train: {counts['train']:,} 토큰, val: {counts['val']:,} 토큰 -> {out_dir}")


if __name__ == "__main__":
    main()
