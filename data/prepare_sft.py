"""대화(SFT) 데이터 준비: 공개 한국어 지시·대화 데이터 -> 토큰화된 예시들.

사전학습이 "한국어의 결"을 가르친다면, SFT는 "질문에 답하는 형식"을 가르친다.
각 예시를 chat 템플릿으로 감싼다:

    <|user|> {질문} <|end|> <|assistant|> {답변} <|end|> <|endoftext|>

그리고 loss_mask를 함께 저장한다 — 답변(assistant) 토큰만 1이고,
사용자 발화·특수 토큰은 0이다. 모델은 "답변을 생성하는 법"만 배우면 되고
사용자의 질문을 외울 필요는 없기 때문이다.

출력: data/bin/sft.npz  (ids, mask 를 패딩 없이 이어붙인 형태 + 각 예시 경계)

사용법:
    python -m data.prepare_sft --tokenizer tokenizer/tokenizer.json
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer.bpe import BPETokenizer


def load_pairs(max_len_chars: int = 1500):
    """(instruction, response) 쌍들을 공개 데이터에서 모은다.
    네트워크/스키마 문제로 일부가 실패해도 나머지로 진행한다."""
    pairs = []

    # 1) KoAlpaca: 지시-응답 (instruction/output)
    try:
        ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
        for r in ds:
            q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
            if q and a and len(q) + len(a) < max_len_chars:
                pairs.append((q, a))
    except Exception as e:
        print(f"[skip] KoAlpaca: {e}")

    # 2) ChatbotData: 일상 대화 (Q/A)
    try:
        ds = load_dataset("songys/Chatbot_data", split="train")
        for r in ds:
            q, a = (r.get("Q") or "").strip(), (r.get("A") or "").strip()
            if q and a:
                pairs.append((q, a))
    except Exception as e:
        print(f"[skip] ChatbotData: {e}")

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/bin/sft.npz")
    ap.add_argument("--max-tokens", type=int, default=512, help="예시 최대 토큰(모델 문맥)")
    ap.add_argument("--n-pause", type=int, default=0,
                    help="답변 앞에 강제 삽입할 <|pause|> 수 — 말하기 전 '생각할 시간'")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    U, A, END, EOT = (tok.encode_special(t) for t in
                      ("<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"))
    pauses = [tok.encode_special("<|pause|>")] * args.n_pause

    pairs = load_pairs()
    print(f"{len(pairs):,}개 대화 쌍 로드. 토큰화 중...")

    all_ids: list[int] = []
    all_mask: list[int] = []
    boundaries: list[int] = [0]  # 각 예시의 시작 인덱스 (학습 시 예시 단위로 자름)

    for q, a in tqdm(pairs):
        q_ids = tok.encode(q)
        a_ids = tok.encode(a)
        # 템플릿 조립. mask: 답변 토큰과 그 끝의 <|end|>만 1 (그걸 생성해야 하므로)
        # pause는 mask 0 — 모델이 pause를 "출력하도록" 배우는 게 아니라,
        # 강제로 주어진 그 자리의 연산만 활용하게 한다 (chat.py도 강제 삽입)
        ids = [U] + q_ids + [END, A] + pauses + a_ids + [END, EOT]
        mask = ([0] * (1 + len(q_ids) + 2 + len(pauses))) + ([1] * (len(a_ids) + 1)) + [0]
        assert len(ids) == len(mask)
        if len(ids) > args.max_tokens:
            continue
        all_ids.extend(ids)
        all_mask.extend(mask)
        boundaries.append(len(all_ids))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        ids=np.array(all_ids, dtype=np.uint16),
        mask=np.array(all_mask, dtype=np.uint8),
        boundaries=np.array(boundaries, dtype=np.int64),
    )
    print(f"{len(boundaries) - 1:,}개 예시, {len(all_ids):,} 토큰 -> {args.out}")


if __name__ == "__main__":
    main()
