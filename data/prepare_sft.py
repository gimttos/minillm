"""대화(SFT) 데이터 준비: 공개 한국어 지시·대화 데이터 -> 토큰화된 예시들.

사전학습이 "한국어의 결"을 가르친다면, SFT는 "질문에 답하는 형식"을 가르친다.
각 예시를 chat 템플릿으로 감싼다:

    <|user|> {질문} <|end|> <|assistant|> {답변} <|end|> <|endoftext|>

멀티턴이면 이 user/assistant 블록이 여러 번 이어진다 (chat.py build_prompt와
동형이라 추론 쪽 수정 불필요). loss_mask는 assistant 발화(+말미 <|end|>)만
1이다 — 모델은 "대답할 차례에 무슨 말을 하는가"만 배우면 되고, 사용자 발화는
문맥으로만 읽는다.

출력: data/bin/sft.npz  (ids, mask 를 패딩 없이 이어붙인 형태 + 각 예시 경계)

사용법:
    # 단일턴(기존): KoAlpaca + ChatbotData
    python -m data.prepare_sft --tokenizer tokenizer/tokenizer.json
    # 멀티턴: convert_aihub가 만든 대화 JSONL
    python -m data.prepare_sft --conversations data/raw/sns_convos.jsonl --mirror
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tokenizer.bpe import BPETokenizer


def load_pairs(max_len_chars: int = 1500):
    """(instruction, response) 쌍들을 공개 데이터에서 모은다.
    네트워크/스키마 문제로 일부가 실패해도 나머지로 진행한다."""
    from datasets import load_dataset  # 멀티턴 경로에선 datasets 불필요 → 지연 임포트
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


def load_conversations(path: str, mirror: bool):
    """대화 JSONL(convert_aihub 출력) -> (turns, roles) 예시 리스트.

    turns는 speakerA로 시작하는 원래 순서다. 이를 user/assistant로 배정한다:
      - 'B_asst': user=A, assistant=B (turn0부터 user로 시작)
      - 'A_asst': user=B, assistant=A (첫 A 발화를 버리고 B부터 user로 시작)
    mirror면 한 대화에서 두 방향을 모두 만들어 학습량을 2배로, 그리고 모델이
    '먼저 말 걸기'와 '대답하기'를 모두 배우게 한다 (또래 잡담이라 대칭).
    반환 원소: assistant/user 라벨이 붙은 발화 리스트 [(role, text), ...],
    항상 user로 시작한다.
    """
    convos = []
    with open(path, encoding="utf-8-sig") as f:   # BOM 있어도 안전
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns = json.loads(line).get("turns") or []
            if len(turns) < 2:
                continue
            # B_asst: 그대로 (짝수 인덱스=user, 홀수=assistant)
            convos.append([("user" if i % 2 == 0 else "assistant", t)
                           for i, t in enumerate(turns)])
            if mirror:
                # A_asst: 첫 발화(A)를 버려 B부터 user로 시작
                rest = turns[1:]
                if len(rest) >= 2:
                    convos.append([("user" if i % 2 == 0 else "assistant", t)
                                   for i, t in enumerate(rest)])
    return convos


def build_example(labeled_turns, tok, specials, pauses, max_tokens):
    """[(role, text), ...] -> (ids, mask). 최대 토큰을 넘으면 앞(오래된) 턴부터
    통째로 버려 최근 문맥을 살린다. 항상 user로 시작하고 assistant로 끝나도록
    맞춘다. 담을 게 없으면 None."""
    U, A, END, EOT = specials
    blocks = []  # (is_assistant, [토큰...]) — 각 블록은 한 발화(헤더+본문+END)
    for role, text in labeled_turns:
        body = tok.encode(text)
        if role == "assistant":
            blocks.append((True, [A] + pauses + body + [END]))
        else:
            blocks.append((False, [U] + body + [END]))

    # 뒤에서부터 예산 안에 들어오는 만큼만 담는다 (+EOT 자리 1). 최근 대화 우선.
    picked = []
    used = 1
    for is_a, toks in reversed(blocks):
        if used + len(toks) > max_tokens:
            break
        picked.append((is_a, toks))
        used += len(toks)
    picked.reverse()

    # user로 시작하도록 앞의 매달린 assistant 블록을 떨군다 (문맥 없는 답변 방지)
    while picked and picked[0][0]:
        picked.pop(0)
    # 최소한 user 1 + assistant 1 이 있어야 학습 신호가 있다
    if len(picked) < 2 or not any(is_a for is_a, _ in picked):
        return None

    ids, mask = [], []
    for is_a, toks in picked:
        ids.extend(toks)
        if is_a:
            # 헤더 A(+pause)는 mask 0, 본문과 말미 END만 1
            head = 1 + len(pauses)
            mask.extend([0] * head + [1] * (len(toks) - head))
        else:
            mask.extend([0] * len(toks))
    ids.append(EOT)
    mask.append(0)
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/bin/sft.npz")
    ap.add_argument("--max-tokens", type=int, default=512, help="예시 최대 토큰(모델 문맥)")
    ap.add_argument("--n-pause", type=int, default=0,
                    help="답변 앞에 강제 삽입할 <|pause|> 수 — 말하기 전 '생각할 시간'")
    ap.add_argument("--conversations", default="",
                    help="멀티턴 대화 JSONL(convert_aihub 출력). 주면 멀티턴 모드")
    ap.add_argument("--mirror", action="store_true",
                    help="멀티턴에서 A/B 역할을 양방향으로 만들어 학습량 2배")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    U, A, END, EOT = (tok.encode_special(t) for t in
                      ("<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"))
    pauses = [tok.encode_special("<|pause|>")] * args.n_pause

    all_ids: list[int] = []
    all_mask: list[int] = []
    boundaries: list[int] = [0]  # 각 예시의 시작 인덱스 (학습 시 예시 단위로 자름)

    if args.conversations:
        convos = load_conversations(args.conversations, args.mirror)
        print(f"{len(convos):,}개 대화(mirror={args.mirror}) 로드. 토큰화 중...")
        dropped = 0
        for labeled in tqdm(convos):
            ex = build_example(labeled, tok, (U, A, END, EOT), pauses, args.max_tokens)
            if ex is None:
                dropped += 1
                continue
            ids, mask = ex
            all_ids.extend(ids)
            all_mask.extend(mask)
            boundaries.append(len(all_ids))
        if dropped:
            print(f"  (담지 못한 대화 {dropped:,}개 스킵)")
    else:
        pairs = load_pairs()
        print(f"{len(pairs):,}개 대화 쌍 로드. 토큰화 중...")
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
