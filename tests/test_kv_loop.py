"""KV 캐시·마음 유사 기제의 정확성 테스트.

가장 잘 나는 조용한 사고는 "캐시로 한 토큰씩 처리한 결과가 전체를 한 번에
처리한 결과와 미묘하게 다른" 것이다 — loss는 그럴듯하게 내려가는데 생성만
이상해진다. 그래서 모든 조합(기본/loop/mood/latent)에 대해:

  1. 캐시 증분 처리 ≡ 전체 일괄 처리          (generate 경로의 정확성)
  2. 접두부 캐시 + 직사각 마스크 병렬 ≡ 일괄   (잠재 SFT 경로의 정확성)
  3. 기능을 켠 직후에는 출력이 변하지 않음      (항등 초기화·체크포인트 호환)

사용법:
    python -m tests.test_kv_loop
"""

import torch

from model.gpt import GPT, ModelConfig

TOL = dict(rtol=2e-4, atol=2e-4)  # float32에서 rope 복소 연산 오차 허용치


def small_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=64, d_model=32, n_layers=4, n_heads=2,
                ffn_hidden=64, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def full_logits(model, ids, mood=None, ws=None):
    return model.lm_head(model.hidden_states(ids, mood=mood, ws=ws))


def cached_logits(model, ids, mood=None, ws=None):
    """generate와 같은 경로: 캐시를 쓰며 한 토큰씩."""
    caches = model.new_caches()
    outs = []
    for t in range(ids.size(1)):
        h = model.run_from_pos(model.tok_emb(ids[:, t:t + 1]), t, caches, mood=mood, ws=ws)
        outs.append(model.lm_head(h))
    return torch.cat(outs, dim=1)


def chunked_logits(model, ids, split, ws=None):
    """잠재 SFT와 같은 경로: 접두부를 캐시에 쌓고 나머지를 직사각 마스크로 병렬."""
    caches = model.new_caches()
    h1 = model.run_from_pos(model.tok_emb(ids[:, :split]), 0, caches, ws=ws)
    S = ids.size(1) - split
    cols = torch.arange(split + S)
    rows = torch.arange(S)
    mask = (cols[None, :] <= (split + rows)[:, None]).view(1, 1, S, split + S)
    h2 = model.run_from_pos(model.tok_emb(ids[:, split:]), split, caches, attn_mask=mask, ws=ws)
    return model.lm_head(torch.cat([h1, h2], dim=1))


def check(name, a, b):
    torch.testing.assert_close(a, b, **TOL)
    print(f"  ok: {name}")


def main():
    torch.manual_seed(42)

    for label, cfg in [
        ("기본", small_cfg()),
        ("loop x2 (블록 1..3)", small_cfg(loop_start=1, loop_end=3, n_loop=2)),
        ("loop 전 구간 x3", small_cfg(loop_start=0, loop_end=4, n_loop=3)),
        ("mood 16d", small_cfg(mood_dim=16)),
        ("latent 2", small_cfg(n_latent=2)),
        ("workspace 3슬롯", small_cfg(workspace_slots=3)),
        ("전부 켬", small_cfg(loop_start=1, loop_end=3, n_loop=2, mood_dim=16,
                            n_latent=2, workspace_slots=2)),
    ]:
        print(f"[{label}]")
        model = GPT(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 24))
        mood = None
        if cfg.mood_dim > 0:
            # FiLM 경로가 실제로 뭔가 하도록 무작위 가중치를 심고 검사한다
            for block in model.blocks:
                torch.nn.init.normal_(block.mood_film.weight, std=0.1)
            mood = torch.randn(2, cfg.mood_dim)
        ws = None
        if cfg.workspace_slots > 0:
            # ws_read 제로 init를 무작위로 덮어써 방송이 실제로 뭔가 하게 한다
            torch.nn.init.normal_(model.ws_read.weight, std=0.05)
            ws = torch.randn(2, cfg.workspace_slots * cfg.d_model)

        with torch.no_grad():
            full = full_logits(model, ids, mood=mood, ws=ws)
            check("캐시 증분 == 일괄", cached_logits(model, ids, mood=mood, ws=ws), full)
            if mood is None:
                check("접두부+직사각 마스크 == 일괄",
                      chunked_logits(model, ids, 10, ws=ws), full)

    # --- 역피드백: 2-pass 병렬 처리와 캐시 증분 처리가 일치해야 한다 ---
    print("[역피드백]")
    torch.manual_seed(11)
    fb = GPT(small_cfg(feedback=True)).eval()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        ref = full_logits(fb, ids)
        # 제로 초기화: 피드백을 줘도 출력이 변하지 않아야 한다
        h_ref = fb.hidden_states(ids)
        check("제로 초기화 피드백 = 항등",
              fb.lm_head(fb.hidden_states(ids, feedback_h=h_ref)), ref)
        # 무작위 가중치를 심고: 같은 feedback_h에 대해 병렬 == 캐시 증분
        torch.nn.init.normal_(fb.feedback_proj.weight, std=0.05)
        full_fb = fb.lm_head(fb.hidden_states(ids, feedback_h=h_ref))
        caches = fb.new_caches()
        outs = []
        for t in range(ids.size(1)):
            emb = fb.tok_emb(ids[:, t:t + 1])
            if t > 0:
                emb = emb + fb.feedback_proj(h_ref[:, t - 1:t])
            outs.append(fb.lm_head(fb.run_from_pos(emb, t, caches)))
        check("피드백 병렬 == 캐시 증분", torch.cat(outs, dim=1), full_fb)

    # --- 항등 초기화: 기능을 켠 직후에는 출력이 변하지 않아야 한다 ---
    print("[항등 초기화 / 체크포인트 호환]")
    torch.manual_seed(7)
    base = GPT(small_cfg()).eval()
    ids = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        ref = full_logits(base, ids)

        # 기존 체크포인트를 새 기능을 전부 켠 모델에 strict=False로 로드
        on = GPT(small_cfg(mood_dim=16, n_latent=2, feedback=True,
                           conf_head=True, workspace_slots=3, attn_schema=True)).eval()
        missing, unexpected = on.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        check("strict=False 로드 후 출력 동일", full_logits(on, ids), ref)
        check("mood=0 벡터도 출력 동일 (FiLM 제로 초기화)",
              full_logits(on, ids, mood=torch.zeros(2, 16)), ref)
        # 워크스페이스: ws_read 제로 init라 슬롯 상태가 무엇이든 출력 불변
        ws_rand = torch.randn(2, 3 * 32)
        check("ws_read 제로 초기화 = 항등", full_logits(on, ids, ws=ws_rand), ref)
        v = torch.randn(3, 32)
        check("latent_proj는 항등 초기화", on.latent_proj(v), v)

    # 확신도·도식 헤드: loss에 conf/schema 항이 붙어도 logits는 불변 (detach 절연)
    y = torch.randint(0, 64, (2, 20))
    logits_on, loss_on = on(ids, y)
    logits_base, _ = base(ids, y)
    check("conf_head/attn_schema 켜도 logits 불변", logits_on, logits_base)
    assert on._last_schema_loss is not None, "attn_schema 손실이 계산되지 않음"
    print(f"  ok: attn_schema 손실 계산됨 ({on._last_schema_loss.item():.4f})")

    print("\n모든 테스트 통과.")


if __name__ == "__main__":
    main()
