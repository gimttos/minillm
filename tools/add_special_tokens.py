"""이미 학습된 tokenizer.json에 새 특수 토큰을 추가한다 (merge 재학습 없음).

토크나이저를 새로 학습하면(vocab 16392) 처음부터 새 특수 토큰이 들어가지만,
이미 학습·패킹된 토크나이저를 계속 쓰고 싶을 때는 이 스크립트로 특수 토큰만
사전 끝에 이어 붙인다. 기존 토큰 ID는 하나도 변하지 않으므로 이미 만든
train.bin/val.bin을 다시 패킹할 필요가 없다.

사용법:
    python -m tools.add_special_tokens --tokenizer tokenizer/tokenizer.json
"""

import argparse
import json
from pathlib import Path

from tokenizer.bpe import SPECIAL_TOKENS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    path = Path(args.tokenizer)
    obj = json.loads(path.read_text(encoding="utf-8"))
    specials: dict = obj["special_tokens"]
    next_id = max(max(specials.values()), 255 + len(obj["merges"])) + 1

    added = []
    for tok in SPECIAL_TOKENS:
        if tok not in specials:
            specials[tok] = next_id
            added.append((tok, next_id))
            next_id += 1

    if not added:
        print("추가할 특수 토큰이 없습니다 — 이미 최신입니다.")
        return

    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    for tok, i in added:
        print(f"  + {tok} -> id {i}")
    vocab_size = 256 + len(obj["merges"]) + len(specials)
    print(f"저장: {path} (vocab {vocab_size})")
    print(f"주의: 모델의 vocab_size도 {vocab_size} 이상이어야 합니다.")


if __name__ == "__main__":
    main()
