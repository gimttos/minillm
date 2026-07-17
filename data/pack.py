"""코퍼스를 토큰화해 학습용 바이너리(.bin)로 변환.

모델은 텍스트가 아니라 토큰 ID의 긴 나열을 학습한다. 매 스텝마다
토큰화를 반복하면 낭비이므로, 한 번 토큰화해서 uint16 배열로 저장해 두고
학습 때는 np.memmap으로 원하는 위치만 잘라 읽는다 (RAM 절약).

- 문서 경계마다 <|endoftext|> 토큰을 넣는다. 모델은 이 토큰을 보고
  "여기서 문맥이 끊긴다"를 배운다.
- 200개 문서마다 1개는 검증(val)용으로 떼어 둔다 — 본 적 없는 텍스트에
  대한 loss가 진짜 실력이다.
- 코퍼스가 GB 단위여도 돌 수 있게 문서 단위로 스트리밍 처리한다.

## 왜 멀티프로세싱인가

우리 BPE는 순수 파이썬이고 청크마다 병합 후보를 매번 다시 훑는다(O(n²) 성향).
6GB 코퍼스에 ~2.5시간이 걸렸는데, 스케일업으로 30~80GB를 다루려면 단일 코어로는
하루를 훌쩍 넘겨 사실상 불가능하다. 토큰화는 문서끼리 완전히 독립적이라
프로세스로 쪼개면 코어 수만큼 그대로 빨라진다 (GIL을 피하려면 스레드가 아니라
프로세스여야 한다).

순서는 반드시 보존한다(imap). val 분할이 문서 인덱스 기준이라 순서가 흔들리면
train/val 구성이 실행마다 달라져 재현이 깨진다.

사용법:
    python -m data.pack --input data/raw/corpus.txt --tokenizer tokenizer/tokenizer.json
    python -m data.pack --input ... --workers 16 --out-dir data/bin_32k
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.download import DOC_SEP
from tokenizer.bpe import BPETokenizer

FLUSH_EVERY = 8_000_000  # 토큰이 이만큼 쌓이면 디스크로 내보냄 (~16MB)

# 워커 프로세스마다 한 번씩 채워지는 전역 — 토크나이저를 매 문서 피클링하는
# 비용을 피한다 (initializer에서 파일로부터 직접 로드).
_TOK: BPETokenizer | None = None
_EOT: int = 0


def _init_worker(tok_path: str) -> None:
    global _TOK, _EOT
    _TOK = BPETokenizer.load(tok_path)
    _EOT = _TOK.encode_special("<|endoftext|>")


def _encode_doc(doc: str) -> list[int]:
    """문서 하나 -> 토큰 + 말미 <|endoftext|>. (워커에서 실행)"""
    # 가벼운 정규화: 줄 끝 공백·빈 줄 정리
    doc = "\n".join(line.strip() for line in doc.splitlines() if line.strip())
    return _TOK.encode(doc) + [_EOT]


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
    ap.add_argument("--workers", type=int, default=0,
                    help="토큰화 프로세스 수 (0=CPU 코어 수 자동)")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    assert tok.vocab_size <= 65536, "uint16에 담으려면 vocab이 65536 이하여야 함"
    workers = args.workers or (os.cpu_count() or 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    buffers: dict[str, list[int]] = {"train": [], "val": []}
    files = {k: open(out_dir / f"{k}.bin", "wb") for k in counts}

    def flush(split: str):
        np.array(buffers[split], dtype=np.uint16).tofile(files[split])
        counts[split] += len(buffers[split])
        buffers[split].clear()

    print(f"토큰화 시작: vocab {tok.vocab_size:,} | 워커 {workers}개")
    with Pool(workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        # imap = 순서 보존. chunksize로 IPC 왕복을 줄인다(문서가 짧을수록 중요).
        stream = pool.imap(_encode_doc, iter_docs(args.input), chunksize=64)
        for i, ids in enumerate(tqdm(stream, desc="tokenizing")):
            split = "val" if i % args.val_every == 0 else "train"
            buffers[split].extend(ids)
            if len(buffers[split]) >= FLUSH_EVERY:
                flush(split)

    for split in files:
        flush(split)
        files[split].close()
    total = counts["train"] + counts["val"]
    print(f"train: {counts['train']:,} 토큰, val: {counts['val']:,} 토큰 "
          f"(합 {total/1e9:.2f}B) -> {out_dir}")


if __name__ == "__main__":
    main()
