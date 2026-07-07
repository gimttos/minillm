"""로컬 CLI 채팅 — 완성된 모델과 대화하는 곳.

체크포인트를 로드해 CPU에서 돌린다. 대화 형식은 SFT 때 배운 템플릿과
똑같이 맞춰 준다:
    <|user|> {내 말} <|end|> <|assistant|>  ...여기서부터 모델이 생성...

간단한 멀티턴: 직전까지의 대화를 이어붙이되, 문맥 길이(max_seq_len)를
넘으면 오래된 턴부터 잘라낸다.

사용법:
    python chat.py --ckpt checkpoints/sft.pt
    python chat.py --ckpt checkpoints/ckpt_best.pt --raw   # (SFT 전) 이어쓰기 테스트
"""

import argparse
import sys

import torch

from model.gpt import GPT, ModelConfig
from tokenizer.bpe import BPETokenizer


def build_prompt(tok, history, max_ctx):
    """history: [(role, text), ...] -> 토큰 ID 리스트. 뒤에서부터 채워
    문맥을 넘지 않게 오래된 턴을 버린다. 마지막은 <|assistant|>로 끝낸다."""
    U, A, END = (tok.encode_special(t) for t in ("<|user|>", "<|assistant|>", "<|end|>"))
    turns = []
    for role, text in history:
        head = U if role == "user" else A
        turns.append([head] + tok.encode(text) + [END])
    ids = [A]  # 모델이 생성을 시작할 assistant 헤드 (맨 뒤)
    for turn in reversed(turns):
        if len(turn) + len(ids) > max_ctx - 16:
            break
        ids = turn + ids
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--raw", action="store_true",
                    help="템플릿 없이 입력을 그대로 이어쓰기 (사전학습 모델 확인용)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(torch.get_num_threads())  # CPU 코어 전부 사용

    ck = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ck["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    print(f"모델 로드: {args.ckpt} ({model.num_params() / 1e6:.1f}M, {device})")

    stop_ids = {tok.encode_special("<|end|>"), tok.encode_special("<|endoftext|>")}

    if args.raw:
        print("이어쓰기 모드. 프롬프트를 입력하면 뒤를 이어 씁니다. (Ctrl+C 종료)\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            ids = tok.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            sys.stdout.write(prompt)
            for tid in model.generate(x, args.max_new, args.temperature, args.top_p,
                                      stop_ids={tok.encode_special("<|endoftext|>")}):
                sys.stdout.write(tok.decode([tid]))
                sys.stdout.flush()
            print("\n")
        return

    print("대화를 시작하세요. (Ctrl+C 종료)\n")
    history = []
    while True:
        try:
            user = input("나  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        history.append(("user", user))
        ids = build_prompt(tok, history, cfg.max_seq_len)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        sys.stdout.write("봇 > ")
        sys.stdout.flush()
        out_ids = []
        for tid in model.generate(x, args.max_new, args.temperature, args.top_p, stop_ids):
            out_ids.append(tid)
            sys.stdout.write(tok.decode([tid]))
            sys.stdout.flush()
        print("\n")
        history.append(("assistant", tok.decode(out_ids)))


if __name__ == "__main__":
    main()
