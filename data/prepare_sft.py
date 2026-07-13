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


def load_conversations(path: str, mirror: bool, use_persona: bool = True):
    """대화 JSONL(convert_aihub/convert_persona 출력) -> 예시 리스트.

    turns는 화자0으로 시작하는 원래 순서다. 이를 user/assistant로 배정한다:
      - 'B_asst': user=화자0, assistant=화자1 (turn0부터 user로 시작)
      - 'A_asst': user=화자1, assistant=화자0 (첫 발화를 버려 user로 시작)
    mirror면 한 대화에서 두 방향을 모두 만들어 학습량을 2배로, 그리고 모델이
    '먼저 말 걸기'와 '대답하기'를 모두 배우게 한다 (또래 대화라 대칭).

    페르소나 데이터(personas: [[화자0 프로필], [화자1 프로필]])면 각 방향마다
    **그 방향의 assistant 프로필**을 함께 낸다 — 모델이 배우는 것은 "내 프로필이
    주어지면 그에 맞게 말한다"이므로, 상대(user)의 프로필이 아니라 자기 것이어야
    한다. use_persona=False면 무시한다 (문맥 없는 기준선 — workspace 실험용 대조군).

    반환 원소: (labeled_turns, persona_profiles or None). labeled_turns는 항상
    user로 시작하는 [(role, text), ...].
    """
    convos = []
    with open(path, encoding="utf-8-sig") as f:   # BOM 있어도 안전
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            turns = rec.get("turns") or []
            if len(turns) < 2:
                continue
            personas = rec.get("personas") if use_persona else None
            has_p = bool(personas) and len(personas) == 2

            # B_asst: 그대로 (짝수 인덱스=user=화자0, 홀수=assistant=화자1)
            convos.append((
                [("user" if i % 2 == 0 else "assistant", t) for i, t in enumerate(turns)],
                personas[1] if has_p else None,
            ))
            if mirror:
                # A_asst: 첫 발화(화자0)를 버려 화자1부터 user로 시작
                # → assistant는 화자0이 되므로 페르소나도 personas[0]
                rest = turns[1:]
                if len(rest) >= 2:
                    convos.append((
                        [("user" if i % 2 == 0 else "assistant", t)
                         for i, t in enumerate(rest)],
                        personas[0] if has_p else None,
                    ))
    return convos


def _emit(picked, sys_ids, pauses, EOT):
    """고른 블록들 -> (ids, mask). assistant 본문과 말미 <|end|>만 mask 1."""
    ids = list(sys_ids)                           # 페르소나 프리픽스 (mask 0)
    mask = [0] * len(sys_ids)
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


def build_examples(labeled_turns, tok, specials, pauses, max_tokens, sys_ids=None):
    """[(role, text), ...] -> [(ids, mask), ...] — 예산에 맞는 **창(window)들**.

    왜 하나가 아니라 여럿인가: 예산을 넘는 대화에서 '최근 턴만 남기기'를 하면
    모든 예시가 **대화의 끝**에서 끝난다. 그러면 마지막 assistant 턴이 언제나
    작별 인사가 되고, 대화의 도입부는 학습에서 통째로 사라진다 (페르소나 대화는
    평균 574토큰이라 68%가 예산 초과 — assistant 턴의 26%가 버려졌다). 실제로
    그 결과 무슨 말을 걸어도 "좋은 하루 되세요"만 하는 붕괴가 일어났다.

    그래서 대화를 예산 크기의 연속된 창으로 나눠 **모든 턴이 적어도 한 창에는
    학습 대상으로 등장**하게 한다. 각 창은 user로 시작하고 assistant를 포함한다.

    sys_ids(페르소나)는 **모든 창에 붙고 절대 잘리지 않는다** — 정체성은 대화의
    어느 대목에서도 유지돼야 하므로 예산에서 먼저 뺀다. mask는 0."""
    U, A, END, EOT = specials
    sys_ids = sys_ids or []
    budget = max_tokens - len(sys_ids) - 1        # -1은 EOT 자리
    if budget < 8:                                # 페르소나만으로 예산이 찬 경우
        return []

    blocks = []  # (is_assistant, [토큰...]) — 각 블록은 한 발화(헤더+본문+END)
    for role, text in labeled_turns:
        body = tok.encode(text)
        if role == "assistant":
            blocks.append((True, [A] + pauses + body + [END]))
        else:
            blocks.append((False, [U] + body + [END]))

    out, i, n = [], 0, len(blocks)
    while i < n:
        # user 블록에서 창을 시작한다 (문맥 없는 답변 방지)
        while i < n and blocks[i][0]:
            i += 1
        if i >= n:
            break
        picked, used, j = [], 0, i
        while j < n and used + len(blocks[j][1]) <= budget:
            picked.append(blocks[j])
            used += len(blocks[j][1])
            j += 1
        if j == i:            # 발화 하나가 예산보다 큼 — 이 턴은 담을 수 없다
            i += 1
            continue
        # 창은 assistant로 끝나야 학습 신호가 있다 — 꼬리의 매달린 user는 버린다
        while picked and not picked[-1][0]:
            picked.pop()
            j -= 1
        if len(picked) >= 2 and any(is_a for is_a, _ in picked):
            out.append(_emit(picked, sys_ids, pauses, EOT))
        i = max(j, i + 1)     # 다음 창으로 (무한 루프 방지)
    return out


def encode_persona(profiles, tok, sys_id, end_id):
    """프로필 문장들 -> <|sys|> ... <|end|> 토큰. chat.py도 같은 형식을 써야
    학습과 추론의 프리픽스가 일치한다 (build_persona_ids와 동일 계약)."""
    if not profiles:
        return []
    return [sys_id] + tok.encode(" ".join(profiles)) + [end_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="data/bin/sft.npz")
    ap.add_argument("--max-tokens", type=int, default=512, help="예시 최대 토큰(모델 문맥)")
    ap.add_argument("--n-pause", type=int, default=0,
                    help="답변 앞에 강제 삽입할 <|pause|> 수 — 말하기 전 '생각할 시간'")
    ap.add_argument("--conversations", default="",
                    help="멀티턴 대화 JSONL(convert_aihub/convert_persona 출력)")
    ap.add_argument("--mirror", action="store_true",
                    help="멀티턴에서 A/B 역할을 양방향으로 만들어 학습량 2배")
    ap.add_argument("--persona-mode", default="context",
                    choices=["context", "workspace", "none"],
                    help="페르소나를 어느 채널로 주는가. context=<|sys|> 프리픽스로 "
                         "토큰 문맥에 / workspace=따로 저장해 GWT 슬롯으로 압축 "
                         "(토큰 문맥엔 없음) / none=주지 않음(기준선). "
                         "세 모드가 **같은 토큰열**을 쓰므로 정보량은 동일하고 "
                         "채널만 다르다 — 그래야 깨끗한 대조 실험이 된다")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    U, A, END, EOT = (tok.encode_special(t) for t in
                      ("<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"))
    pauses = [tok.encode_special("<|pause|>")] * args.n_pause
    # <|sys|>는 이미 SPECIAL_TOKENS에 예약돼 있다 — 새 토큰을 만들지 않으므로
    # vocab(16392)도 패킹된 .bin도 그대로다.
    SYS = tok.encode_special("<|sys|>") if tok.has_special("<|sys|>") else None

    all_ids: list[int] = []
    all_mask: list[int] = []
    boundaries: list[int] = [0]  # 각 예시의 시작 인덱스 (학습 시 예시 단위로 자름)
    # workspace 모드에서만 채운다: 예시별 페르소나 토큰열 (대화 문맥에는 없다)
    p_all: list[int] = []
    p_bounds: list[int] = [0]

    if args.conversations:
        want_persona = args.persona_mode != "none" and SYS is not None
        to_ws = args.persona_mode == "workspace"
        convos = load_conversations(args.conversations, args.mirror, want_persona)
        n_with_p = sum(1 for _, p in convos if p)
        print(f"{len(convos):,}개 대화(mirror={args.mirror}) 로드, "
              f"페르소나 {n_with_p:,}개 -> {args.persona_mode}. 토큰화 중...")
        dropped = 0
        for labeled, profiles in tqdm(convos):
            p_ids = encode_persona(profiles, tok, SYS, END) if profiles else []
            # 같은 토큰열을 context면 문맥 앞에, workspace면 슬롯 압축용으로 따로.
            wins = build_examples(labeled, tok, (U, A, END, EOT), pauses,
                                  args.max_tokens,
                                  sys_ids=[] if to_ws else p_ids)
            if not wins:
                dropped += 1
                continue
            for ids, mask in wins:   # 한 대화가 여러 창을 낼 수 있다
                all_ids.extend(ids)
                all_mask.extend(mask)
                boundaries.append(len(all_ids))
                if to_ws:            # 페르소나는 모든 창에 붙는다
                    p_all.extend(p_ids)
                    p_bounds.append(len(p_all))
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
    out = dict(
        ids=np.array(all_ids, dtype=np.uint16),
        mask=np.array(all_mask, dtype=np.uint8),
        boundaries=np.array(boundaries, dtype=np.int64),
    )
    if len(p_bounds) > 1:   # workspace 모드: 예시별 페르소나 토큰열을 함께 저장
        assert len(p_bounds) == len(boundaries), "페르소나/예시 경계 개수 불일치"
        out["p_ids"] = np.array(p_all, dtype=np.uint16)
        out["p_boundaries"] = np.array(p_bounds, dtype=np.int64)
    np.savez(args.out, **out)
    extra = f", 페르소나 {len(p_all):,} 토큰(workspace)" if len(p_bounds) > 1 else ""
    print(f"{len(boundaries) - 1:,}개 예시, {len(all_ids):,} 토큰{extra} -> {args.out}")


if __name__ == "__main__":
    main()
