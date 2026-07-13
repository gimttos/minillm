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
                   --mood-file / --state-file 로 세션 간 지속.

런타임 층 (모델 밖, 학습 불필요 — 핸드오프 워크스트림 G)
========================================================
  --state-file   : mood+workspace+drive+events 를 한 파일에 저장/복원
  --proactive    : drive 임계·쿨다운·DND 를 지켜 먼저 말 걸기
  --runtime-config : 정책 JSON (ModelConfig에 넣지 않음)
drive 는 mood/latent/workspace 내부 채널로만 작용한다 (우회 래퍼 금지).

사용법:
    python chat.py --ckpt checkpoints/sft.pt
    python chat.py --ckpt checkpoints/sft.pt --show-mood --mood-file mood.pt
    python chat.py --ckpt checkpoints/sft.pt --state-file session.pt --proactive
    python chat.py --ckpt checkpoints/sft.pt --n-loop 1        # CPU 고속 모드
    python chat.py --ckpt checkpoints/ckpt_best.pt --raw       # (SFT 전) 이어쓰기 테스트
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import torch

from model.gpt import GPT, ModelConfig
from tokenizer.bpe import BPETokenizer


def build_persona_ids(tok, profiles):
    """프로필 문장들 -> <|sys|> ... <|end|> 토큰.
    prepare_sft.encode_persona와 **똑같은 형식**이어야 한다 — 학습 때 본 프리픽스와
    추론 때 주는 프리픽스가 어긋나면 페르소나가 먹히지 않는다."""
    if not profiles or not tok.has_special("<|sys|>"):
        return []
    return ([tok.encode_special("<|sys|>")] + tok.encode(" ".join(profiles))
            + [tok.encode_special("<|end|>")])


def build_prompt(tok, history, max_ctx, n_pause=0, sys_ids=None):
    """history: [(role, text), ...] -> 토큰 ID 리스트. 뒤에서부터 채워
    문맥을 넘지 않게 오래된 턴을 버린다. 마지막은 <|assistant|>로 끝내고,
    n_pause > 0이면 SFT 데이터와 똑같이 <|pause|>를 강제로 붙인다.

    sys_ids(페르소나)는 절대 잘리지 않는다 — 대화가 길어져도 정체성은 유지돼야
    하므로 예산에서 먼저 빼고 남는 것으로 턴을 채운다 (prepare_sft와 같은 규칙)."""
    U, A, END = (tok.encode_special(t) for t in ("<|user|>", "<|assistant|>", "<|end|>"))
    sys_ids = sys_ids or []
    turns = []
    for role, text in history:
        head = U if role == "user" else A
        turns.append([head] + tok.encode(text) + [END])
    ids = [A]  # 모델이 생성을 시작할 assistant 헤드 (맨 뒤)
    if n_pause > 0 and tok.has_special("<|pause|>"):
        ids += [tok.encode_special("<|pause|>")] * n_pause
    for turn in reversed(turns):
        if len(sys_ids) + len(turn) + len(ids) > max_ctx - 16:
            break
        ids = turn + ids
    return sys_ids + ids


def _start_input_thread(prompt: str, q: queue.Queue) -> None:
    """stdin을 별도 스레드에서 읽어 proactive idle 틱과 공존시킨다."""
    def _reader():
        while True:
            try:
                line = input(prompt)
            except (EOFError, KeyboardInterrupt):
                q.put(None)
                return
            q.put(line)
    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--repetition-penalty", type=float, default=1.2,
                    help="등장한 토큰의 logit 벌점 (1.0=끔). 작은 모델의 "
                         "같은 말 도배 루프를 끊는다. 1.1~1.3 권장")
    ap.add_argument("--raw", action="store_true",
                    help="템플릿 없이 입력을 그대로 이어쓰기 (사전학습 모델 확인용)")
    # --- 페르소나: <|sys|> 프리픽스로 정체성을 준다 (SFT가 배운 형식과 동일) ---
    ap.add_argument("--persona", default="",
                    help="프로필 문장들. 여러 개는 | 로 구분 "
                         "(예: \"나는 이자카야 사장이다.|나는 확고한 성격이다.\")")
    ap.add_argument("--persona-file", default="",
                    help="프로필을 한 줄에 하나씩 담은 텍스트 파일")
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
    # --- 런타임 층 (G): 정책은 ModelConfig 밖 ---
    ap.add_argument("--state-file", default="",
                    help="mood+workspace+drive+events 통합 상태 파일 (mood-file 확장)")
    ap.add_argument("--runtime-config", default="runtime/config.json",
                    help="런타임 정책 JSON (임계·쿨다운·DND — 체크포인트와 무관)")
    ap.add_argument("--proactive", action="store_true",
                    help="drive 임계+쿨다운+DND를 지켜 먼저 말 걸기 (ELIZA 증거 아님)")
    ap.add_argument("--show-drive", action="store_true",
                    help="매 턴/idle 틱 drive·상태 요약 출력")
    ap.add_argument("--no-drive", action="store_true",
                    help="drive 라우팅 끄기 (비교 실험용 — 상태 파일은 유지 가능)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(torch.get_num_threads())  # CPU 코어 전부 사용

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer)
    print(f"모델 로드: {args.ckpt} ({model.num_params() / 1e6:.1f}M, {device})")

    # 페르소나 프리픽스 (학습 때와 같은 <|sys|> ... <|end|> 형식)
    profiles = []
    if args.persona_file:
        profiles = [ln.strip() for ln in
                    Path(args.persona_file).read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
    elif args.persona:
        profiles = [p.strip() for p in args.persona.split("|") if p.strip()]
    sys_ids = build_persona_ids(tok, profiles)
    if profiles and not sys_ids:
        print("주의: 이 토크나이저에 <|sys|>가 없어 페르소나를 주입하지 못했습니다")

    features = []
    if sys_ids:
        features.append(f"persona {len(profiles)}문장")
    if cfg.n_loop > 1:
        n_loop = args.n_loop if args.n_loop is not None else cfg.n_loop
        if n_loop > cfg.n_loop:
            print(f"주의: n_loop {n_loop} > 학습 최대치 {cfg.n_loop} — 외삽 실험 모드")
        features.append(f"loop x{n_loop}")
    else:
        n_loop = args.n_loop
    if cfg.n_pause > 0:
        features.append(f"pause {cfg.n_pause}")
    base_latent = args.n_latent if args.n_latent is not None else cfg.n_latent
    if base_latent > 0:
        features.append(f"latent {base_latent}")
    if cfg.feedback:
        features.append("feedback")
    if cfg.conf_head:
        features.append("conf")
    if args.adaptive_latent is not None:
        if cfg.conf_head and cfg.n_latent > 0:
            features.append(f"adaptive<{args.adaptive_latent}")
        else:
            print("주의: --adaptive-latent 는 확신도 헤드와 latent가 켜진 모델에서만 동작")

    # --- 런타임 상태 (G3): --state-file 이 있으면 통합 관리 ---
    from runtime.state import StateManager, load_runtime_config
    from runtime.proactive import ProactiveEngine

    rt_cfg = load_runtime_config(args.runtime_config if Path(args.runtime_config).exists() else None)
    use_runtime = bool(args.state_file) or args.proactive or args.show_drive
    sm: StateManager | None = None
    proactive: ProactiveEngine | None = None

    mood = None
    ws = None
    trajectory = []
    ws_size = 0
    if cfg.workspace_slots > 0 and not args.no_workspace:
        ws_size = cfg.workspace_slots * (cfg.workspace_dim or cfg.d_model)

    if use_runtime:
        sm = StateManager(
            cfg=rt_cfg,
            mood_dim=cfg.mood_dim if not args.no_mood else 0,
            workspace_size=ws_size,
            device=device,
            state_path=args.state_file,
        )
        # 구형 분리 파일이 있고 통합 파일이 없으면 한 번 흡수
        if args.mood_file and Path(args.mood_file).exists() and sm.mood is not None:
            if not args.state_file or not Path(args.state_file).exists():
                legacy = torch.load(args.mood_file, map_location=device, weights_only=False)
                if torch.is_tensor(legacy):
                    sm.mood = legacy.to(device)
        if args.workspace_file and Path(args.workspace_file).exists() and sm.ws is not None:
            if not args.state_file or not Path(args.state_file).exists():
                legacy = torch.load(args.workspace_file, map_location=device, weights_only=False)
                if torch.is_tensor(legacy):
                    sm.ws = legacy.to(device)
        mood = sm.mood
        ws = sm.ws
        if args.state_file and Path(args.state_file).exists():
            print(f"상태 복원: {args.state_file} ({sm.summary()})")
        features.append("runtime-state")
        if args.proactive:
            proactive = ProactiveEngine.from_config(rt_cfg)
            # 상태 파일에 proactive 시각이 있으면 복원
            for ev in reversed(sm.events):
                if ev.get("kind") == "proactive" and "t" in ev:
                    proactive.last_proactive_ts = float(ev["t"])
                    break
            features.append("proactive")
        if not args.no_drive:
            features.append("drive→mood/latent")
    else:
        # --- 기존 경로: mood/workspace 단독 파일 ---
        if cfg.mood_dim > 0 and not args.no_mood:
            if args.mood_file and Path(args.mood_file).exists():
                mood = torch.load(args.mood_file, map_location=device, weights_only=False)
                print(f"기분 복원: {args.mood_file} (‖mood‖={mood.norm():.3f})")
            else:
                mood = torch.zeros(1, cfg.mood_dim, device=device)
            features.append(f"mood {cfg.mood_dim}d")
        if cfg.workspace_slots > 0 and not args.no_workspace:
            if args.workspace_file and Path(args.workspace_file).exists():
                ws = torch.load(args.workspace_file, map_location=device, weights_only=False)
                print(f"워크스페이스 복원: {args.workspace_file} (‖ws‖={ws.norm():.3f})")
            else:
                ws = torch.zeros(1, ws_size, device=device)
            features.append(f"workspace {cfg.workspace_slots}슬롯")

    if cfg.mood_dim > 0 and not args.no_mood and "mood" not in " ".join(features):
        features.append(f"mood {cfg.mood_dim}d")
    if cfg.workspace_slots > 0 and not args.no_workspace and "workspace" not in " ".join(features):
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
                                      n_loop=n_loop, n_latent=0,
                                      repetition_penalty=args.repetition_penalty):
                sys.stdout.write(tok.decode([tid]))
                sys.stdout.flush()
            print("\n")
        return

    def generate_reply(history, *, is_proactive=False, drive_kind=""):
        """한 턴 생성. drive 라우팅(G2)은 sm 이 있을 때만 mood/latent에 스며든다."""
        nonlocal mood, ws
        ids = build_prompt(tok, history, cfg.max_seq_len, n_pause=cfg.n_pause,
                           sys_ids=sys_ids)
        x = torch.tensor([ids], dtype=torch.long, device=device)

        n_latent = base_latent
        if sm is not None and not args.no_drive:
            mood_arg = sm.mood_arg()
            ws_arg = sm.ws_arg()
            n_latent = sm.routed_latent(base_latent)
        else:
            mood_arg = mood if (mood is not None and mood.abs().max() > 0) else None
            ws_arg = ws if (ws is not None and ws.abs().max() > 0) else None

        tag = "봇 > " if not is_proactive else f"봇* > "
        sys.stdout.write(tag)
        sys.stdout.flush()
        out_ids = []
        for tid in model.generate(x, args.max_new, args.temperature, args.top_p,
                                  stop_ids, mood=mood_arg, n_loop=n_loop,
                                  n_latent=n_latent if args.n_latent is None else args.n_latent,
                                  conf_threshold=args.adaptive_latent,
                                  max_latent=args.max_latent, ws=ws_arg,
                                  repetition_penalty=args.repetition_penalty):
            out_ids.append(tid)
            if tid not in skip_ids:
                sys.stdout.write(tok.decode([tid]))
                sys.stdout.flush()
        print("\n")
        text = tok.decode(out_ids)
        history.append(("assistant", text))

        if args.show_conf and model._turn_conf_mean is not None:
            print(f"  (확신도 {model._turn_conf_mean:.2f}, "
                  f"잠재 스텝 {model._turn_latent_steps})\n")

        # --- 턴 종료: mood/ws EMA 갱신 (모델 안 기제) ---
        if mood is not None:
            mood = model.update_mood(mood, decay=args.mood_decay)
            trajectory.append(mood.squeeze(0).cpu().numpy().copy())
            if sm is not None:
                sm.set_mood(mood)
            if args.show_mood:
                v = mood.squeeze(0)
                top = v.abs().topk(min(3, v.numel()))
                dims = ", ".join(f"[{i}]={v[i]:+.2f}" for i in top.indices.tolist())
                print(f"  (기분 ‖{v.norm():.3f}‖ {dims})\n")

        if ws is not None:
            ws = model.update_workspace(ws, decay=args.workspace_decay)
            if sm is not None:
                sm.set_workspace(ws)
            if args.show_workspace:
                v = ws.squeeze(0)
                top = v.abs().topk(min(3, v.numel()))
                dims = ", ".join(f"[{i}]={v[i]:+.2f}" for i in top.indices.tolist())
                print(f"  (워크스페이스 ‖{v.norm():.3f}‖ {dims})\n")

        if sm is not None:
            if is_proactive:
                sm.on_proactive(drive_kind)
            else:
                sm.on_assistant_reply(n_tokens=len(out_ids))
            if args.show_drive:
                print(f"  ({sm.summary()})\n")
            if args.state_file:
                sm.save()

        return text

    print("대화를 시작하세요. (Ctrl+C 종료)")
    if args.proactive:
        idle = int(rt_cfg.get("state", {}).get("idle_tick_sec", 120))
        print(f"proactive 모드: idle {idle}s마다 drive 갱신, 임계·쿨다운·DND 준수")
        print("(proactive 출력은 지표 증거가 아님 — ELIZA 경계)\n")
    else:
        print()

    history = []
    try:
        if args.proactive and sm is not None and proactive is not None:
            # 입력 스레드 + idle 틱 (busy-loop 금지: timeout 대기만)
            idle_sec = float(rt_cfg.get("state", {}).get("idle_tick_sec", 120))
            in_q: queue.Queue = queue.Queue()
            _start_input_thread("나  > ", in_q)
            while True:
                try:
                    user = in_q.get(timeout=idle_sec)
                except queue.Empty:
                    # idle tick: drive 상승 → proactive 판정
                    sm.idle_tick()
                    if args.show_drive:
                        print(f"  [idle] {sm.summary()}")
                    ok, reason = proactive.should_speak(sm.drive)
                    if ok and history:
                        # 빈 대화에선 먼저 말하지 않음 — 맥락 없는 독백 방지
                        generate_reply(history, is_proactive=True, drive_kind=reason)
                        proactive.mark_spoke(reason)
                        if args.show_drive:
                            print(f"  [proactive:{reason}] {sm.summary()}\n")
                    elif ok and not history and args.show_drive:
                        print(f"  [proactive 보류: 대화 이력 없음, drive={reason}]")
                    continue

                if user is None:
                    print()
                    break
                user = user.strip()
                if not user:
                    # 프롬프트 재표시용 — 입력 스레드는 한 번만 뜨므로 빈 줄은 무시
                    continue
                sm.on_user_input(user)
                history.append(("user", user))
                generate_reply(history, is_proactive=False)
        else:
            while True:
                try:
                    user = input("나  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user:
                    continue
                if sm is not None:
                    sm.on_user_input(user)
                history.append(("user", user))
                generate_reply(history, is_proactive=False)
    finally:
        if sm is not None and args.state_file:
            sm.save(args.state_file)
            print(f"상태 저장: {args.state_file} ({sm.summary()})")
        # 구형 단독 파일도 유지 (state-file 없을 때)
        if sm is None:
            if mood is not None and args.mood_file:
                torch.save(mood.cpu(), args.mood_file)
                print(f"기분 저장: {args.mood_file}")
            if ws is not None and args.workspace_file:
                torch.save(ws.cpu(), args.workspace_file)
                print(f"워크스페이스 저장: {args.workspace_file}")
        else:
            # 통합 상태와 별도로 구형 파일 요청이 있으면 텐서만 덤프
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
