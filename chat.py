"""로컬 CLI 채팅 — 완성된 모델과 대화하는 곳.

체크포인트를 로드해 CPU에서 돌린다. 대화 형식은 SFT 때 배운 템플릿과
똑같이 맞춰 준다:
    <|user|> {내 말} <|end|> <|assistant|>  ...여기서부터 모델이 생성...

간단한 멀티턴: 직전까지의 대화를 이어붙이되, 문맥 길이(max_seq_len)를
넘으면 오래된 턴부터 잘라낸다.

마음 유사 기제는 체크포인트의 model_config가 스스로 알려 준다:
  - n_pause > 0  : 프롬프트 끝에 <|pause|>를 붙여 "생각할 시간"을 준다
  - n_latent > 0 : 답변 전에 은닉 상태를 말 없이 되먹인다 (잠재 사고)
  - mood_dim > 0 : 턴 사이에 지속·감쇠하는 기분 벡터를 유지한다.
                   --mood-file을 주면 세션이 끝나도 기분이 저장된다 —
                   "지난 대화의 느낌을 기억하는" 셈이다.

사용법:
    python chat.py --ckpt checkpoints/sft.pt
    python chat.py --ckpt checkpoints/sft.pt --show-mood --mood-file mood.pt
    python chat.py --ckpt checkpoints/sft.pt --n-loop 1        # CPU 고속 모드
    python chat.py --ckpt checkpoints/ckpt_best.pt --raw       # (SFT 전) 이어쓰기 테스트
"""

import argparse
import sys
from pathlib import Path

import torch

from model.gpt import GPT, ModelConfig
from tokenizer.bpe import BPETokenizer


def build_prompt(tok, history, max_ctx, n_pause=0):
    """history: [(role, text), ...] -> 토큰 ID 리스트. 뒤에서부터 채워
    문맥을 넘지 않게 오래된 턴을 버린다. 마지막은 <|assistant|>로 끝내고,
    n_pause > 0이면 SFT 데이터와 똑같이 <|pause|>를 강제로 붙인다."""
    U, A, END = (tok.encode_special(t) for t in ("<|user|>", "<|assistant|>", "<|end|>"))
    turns = []
    for role, text in history:
        head = U if role == "user" else A
        turns.append([head] + tok.encode(text) + [END])
    ids = [A]  # 모델이 생성을 시작할 assistant 헤드 (맨 뒤)
    if n_pause > 0 and tok.has_special("<|pause|>"):
        ids += [tok.encode_special("<|pause|>")] * n_pause
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
    # --- 마음 유사 기제 조절 (기본값은 체크포인트가 스스로 결정) ---
    ap.add_argument("--n-loop", type=int, default=None,
                    help="loop 반복 횟수 오버라이드 (1=CPU 고속, 학습 최대치 초과는 실험용)")
    ap.add_argument("--n-latent", type=int, default=None,
                    help="잠재 사고 스텝 수 오버라이드 (0이면 끔 — 비교 실험용)")
    ap.add_argument("--no-mood", action="store_true", help="기분 벡터 끄기 (비교 실험용)")
    ap.add_argument("--mood-file", default="", help="기분 벡터를 세션 간 저장/복원할 파일")
    ap.add_argument("--mood-decay", type=float, default=0.9)
    ap.add_argument("--show-mood", action="store_true",
                    help="매 턴 기분 상태를 출력하고 종료 시 mood_trajectory.npy로 저장")
    ap.add_argument("--show-conf", action="store_true",
                    help="매 턴 모델의 평균 확신도와 잠재 스텝 수를 출력")
    ap.add_argument("--no-workspace", action="store_true", help="워크스페이스 끄기(비교 실험용)")
    ap.add_argument("--workspace-file", default="",
                    help="워크스페이스 슬롯을 세션 간 저장/복원할 파일")
    ap.add_argument("--workspace-decay", type=float, default=0.95)
    ap.add_argument("--show-workspace", action="store_true",
                    help="매 턴 워크스페이스 슬롯 노름/상위 성분을 출력")
    ap.add_argument("--adaptive-latent", type=float, default=None, metavar="T",
                    help="확신도가 T보다 낮으면 잠재 스텝을 더 밟는다 (확신도 헤드 필요)")
    ap.add_argument("--max-latent", type=int, default=6,
                    help="적응적 잠재 사고의 스텝 상한")
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

    features = []
    if cfg.n_loop > 1:
        n_loop = args.n_loop if args.n_loop is not None else cfg.n_loop
        if n_loop > cfg.n_loop:
            print(f"주의: n_loop {n_loop} > 학습 최대치 {cfg.n_loop} — 외삽 실험 모드")
        features.append(f"loop x{n_loop}")
    else:
        n_loop = args.n_loop
    if cfg.n_pause > 0:
        features.append(f"pause {cfg.n_pause}")
    n_latent_eff = args.n_latent if args.n_latent is not None else cfg.n_latent
    if n_latent_eff > 0:
        features.append(f"latent {n_latent_eff}")
    if cfg.feedback:
        features.append("feedback")
    if cfg.conf_head:
        features.append("conf")
    if args.adaptive_latent is not None:
        if cfg.conf_head and cfg.n_latent > 0:
            features.append(f"adaptive<{args.adaptive_latent}")
        else:
            print("주의: --adaptive-latent 는 확신도 헤드와 latent가 켜진 모델에서만 동작")

    # --- 기분 벡터: 세션 상태 초기화/복원 ---
    mood = None
    trajectory = []
    if cfg.mood_dim > 0 and not args.no_mood:
        if args.mood_file and Path(args.mood_file).exists():
            mood = torch.load(args.mood_file, map_location=device)
            print(f"기분 복원: {args.mood_file} (‖mood‖={mood.norm():.3f})")
        else:
            mood = torch.zeros(1, cfg.mood_dim, device=device)
        features.append(f"mood {cfg.mood_dim}d")

    # --- 워크스페이스: 세션 지속 슬롯 상태 초기화/복원 ---
    ws = None
    if cfg.workspace_slots > 0 and not args.no_workspace:
        ws_size = cfg.workspace_slots * (cfg.workspace_dim or cfg.d_model)
        if args.workspace_file and Path(args.workspace_file).exists():
            ws = torch.load(args.workspace_file, map_location=device)
            print(f"워크스페이스 복원: {args.workspace_file} (‖ws‖={ws.norm():.3f})")
        else:
            ws = torch.zeros(1, ws_size, device=device)
        features.append(f"workspace {cfg.workspace_slots}슬롯")

    if features:
        print(f"마음 유사 기제: {', '.join(features)}")

    stop_ids = {tok.encode_special("<|end|>"), tok.encode_special("<|endoftext|>")}
    skip_ids = {tok.encode_special(t) for t in ("<|pause|>", "<|user|>", "<|assistant|>")
                if tok.has_special(t)}  # 혹시 생성돼도 화면에는 찍지 않는다

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
                                      stop_ids={tok.encode_special("<|endoftext|>")},
                                      n_loop=n_loop, n_latent=0):
                sys.stdout.write(tok.decode([tid]))
                sys.stdout.flush()
            print("\n")
        return

    print("대화를 시작하세요. (Ctrl+C 종료)\n")
    history = []
    try:
        while True:
            try:
                user = input("나  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            history.append(("user", user))
            ids = build_prompt(tok, history, cfg.max_seq_len, n_pause=cfg.n_pause)
            x = torch.tensor([ids], dtype=torch.long, device=device)

            # 첫 턴처럼 기분이 아직 0이면 None으로 — SFT 때도 "기분 없음"으로
            # 학습한 경로가 있어 이쪽이 자연스럽다
            mood_arg = mood if (mood is not None and mood.abs().max() > 0) else None
            # 워크스페이스도 처음(빈 슬롯)엔 주입하지 않는다 — mood와 같은 규율
            ws_arg = ws if (ws is not None and ws.abs().max() > 0) else None

            sys.stdout.write("봇 > ")
            sys.stdout.flush()
            out_ids = []
            for tid in model.generate(x, args.max_new, args.temperature, args.top_p,
                                      stop_ids, mood=mood_arg, n_loop=n_loop,
                                      n_latent=args.n_latent,
                                      conf_threshold=args.adaptive_latent,
                                      max_latent=args.max_latent, ws=ws_arg):
                out_ids.append(tid)
                if tid not in skip_ids:
                    sys.stdout.write(tok.decode([tid]))
                    sys.stdout.flush()
            print("\n")
            history.append(("assistant", tok.decode(out_ids)))

            if args.show_conf and model._turn_conf_mean is not None:
                print(f"  (확신도 {model._turn_conf_mean:.2f}, "
                      f"잠재 스텝 {model._turn_latent_steps})\n")

            # --- 턴이 끝나면 기분을 감쇠·누적 갱신 ---
            if mood is not None:
                mood = model.update_mood(mood, decay=args.mood_decay)
                trajectory.append(mood.squeeze(0).cpu().numpy().copy())
                if args.show_mood:
                    v = mood.squeeze(0)
                    top = v.abs().topk(min(3, v.numel()))
                    dims = ", ".join(f"[{i}]={v[i]:+.2f}" for i in top.indices.tolist())
                    print(f"  (기분 ‖{v.norm():.3f}‖ {dims})\n")

            # --- 워크스페이스도 턴 종료 시 은닉 요약으로 EMA 갱신 ---
            if ws is not None:
                ws = model.update_workspace(ws, decay=args.workspace_decay)
                if args.show_workspace:
                    v = ws.squeeze(0)
                    top = v.abs().topk(min(3, v.numel()))
                    dims = ", ".join(f"[{i}]={v[i]:+.2f}" for i in top.indices.tolist())
                    print(f"  (워크스페이스 ‖{v.norm():.3f}‖ {dims})\n")
    finally:
        if mood is not None and args.mood_file:
            torch.save(mood.cpu(), args.mood_file)
            print(f"기분 저장: {args.mood_file}")
        if ws is not None and args.workspace_file:
            torch.save(ws.cpu(), args.workspace_file)
            print(f"워크스페이스 저장: {args.workspace_file}")
        if trajectory and args.show_mood:
            import numpy as np
            np.save("mood_trajectory.npy", np.stack(trajectory))
            print(f"기분 궤적 저장: mood_trajectory.npy ({len(trajectory)}턴)")


if __name__ == "__main__":
    main()
