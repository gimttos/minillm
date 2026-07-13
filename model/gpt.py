"""Decoder-only Transformer (미니 Llama) — 직접 구현.

전체 흐름
=========
토큰 ID 열 → [임베딩] → [Transformer 블록 × N] → [RMSNorm] → [출력층]
→ "다음 토큰이 무엇일지"에 대한 vocab 크기의 확률 분포

각 Transformer 블록:
    x = x + Attention(RMSNorm(x))   # 다른 위치의 정보를 끌어온다
    x = x + SwiGLU(RMSNorm(x))      # 끌어온 정보를 가공한다

이 파일이 곧 모델의 전부다. 학습(pretrain.py)도 대화(chat.py)도
전부 이 클래스의 forward를 부르는 것뿐이다.

마음 유사 기제 (전부 ModelConfig 플래그, 기본 off)
=================================================
- loop  : 중간 블록들을 같은 가중치로 여러 번 통과 — 파라미터를 늘리지
          않고 "한 번 더 생각할 시간"을 준다. (재귀 깊이)
- mood  : 턴 사이에 지속·감쇠하는 작은 상태 벡터를 각 블록에 FiLM
          (scale/shift)으로 주입 — "객관적인 기분 상태"의 구현.
- latent: 답변 첫 토큰을 뱉기 전에 은닉 상태를 말 없이 k번 자기 입력으로
          되먹인다(Coconut) — "속말 없는 개념적 사고".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512        # 토큰 하나를 나타내는 벡터의 차원
    n_layers: int = 8         # Transformer 블록 수
    n_heads: int = 8          # 어텐션 헤드 수 (head_dim = 512/8 = 64)
    ffn_hidden: int = 1408    # SwiGLU 내부 차원 (~ 8/3 * d_model)
    max_seq_len: int = 512    # 한 번에 볼 수 있는 최대 토큰 수 (문맥 길이)
    rope_theta: float = 10000.0
    dropout: float = 0.0      # 데이터가 많으면 0이 보통 (과적합 걱정 없음)

    # --- 마음 유사 기제 (기본값 off — 예전 체크포인트는 그대로 로드된다) ---
    loop_start: int = 0       # blocks[loop_start:loop_end]를 하나의 그룹으로
    loop_end: int = 0         # n_loop번 반복 통과시킨다 (가중치 공유)
    n_loop: int = 1
    mood_dim: int = 0         # 기분 벡터 차원 (0 = 없음)
    n_latent: int = 0         # 답변 전 잠재 사고(Coconut) 스텝 수
    n_pause: int = 0          # 답변 앞 강제 <|pause|> 수 (데이터 레벨 — chat.py가 참조)
    feedback: bool = False    # 직전 토큰의 최종 은닉 -> 다음 토큰 입력 (역피드백)
    conf_head: bool = False   # "다음 토큰을 맞힐 것인가" 확신도 헤드 (메타인지)
    # workspace(GWT): 세션 내내 지속되며 모든 블록이 읽는 소수의 작업공간 슬롯.
    # 턴 단위 latent를 넘어, 제한용량 작업공간 + 전역 방송을 흉내낸다.
    workspace_slots: int = 0  # 0 = off
    workspace_dim: int = 0    # 슬롯 하나의 차원 (0 = d_model)
    # attn_schema(AST): 자기 어텐션 상태(레이어별 엔트로피)를 은닉에서 예측하는
    # 보조 헤드. 자기 주의의 단순 모델 — 본체 logits는 불변(detach 절연).
    attn_schema: bool = False


class RMSNorm(nn.Module):
    """LayerNorm의 단순화판: 평균을 빼지 않고 크기(RMS)만 1로 정규화.

    깊은 네트워크에서 값이 폭주/소멸하지 않게 각 블록 입구에서 벡터
    크기를 일정하게 맞춰 준다. 학습 파라미터는 스케일(weight) 하나뿐.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    """RoPE(회전 위치 인코딩)용 회전 각도를 미리 계산한다.

    어텐션은 그 자체로는 토큰 순서를 모른다. RoPE는 q, k 벡터를
    2차원씩 짝지어 "위치 × 주파수"만큼 회전시켜 순서 정보를 심는다.
    두 벡터의 내적(=어텐션 점수)이 절대 위치가 아니라 상대 거리에만
    의존하게 되는 것이 핵심 성질이다.

    반환: (max_seq_len, head_dim/2) 크기의 복소수 텐서 e^{i·pos·freq}
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq_len).float()
    angles = torch.outer(pos, freqs)                      # (seq, head_dim/2)
    return torch.polar(torch.ones_like(angles), angles)   # 복소수로 표현


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim), rope: (T, head_dim/2) 복소수."""
    # (실수부, 허수부) 쌍으로 묶어 복소수 곱 = 2D 회전
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * rope)                   # 회전 적용
    return out.reshape(*x.shape).type_as(x)


class Attention(nn.Module):
    """멀티헤드 셀프 어텐션 (causal).

    각 토큰이 질문(q)을 던지고, 앞선 토큰들의 키(k)와 맞춰본 뒤,
    잘 맞는 토큰의 값(v)을 가중 평균해 가져온다.
    "causal" = 미래 토큰은 절대 보지 못하게 마스킹 (다음 토큰 예측이니까).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # q, k, v를 한 번에 계산하는 projection (bias 없음 — Llama 방식)
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.last_entropy = None   # attn_schema: 직전 collect 패스의 위치별 엔트로피

    def forward(self, x, rope, kv_cache=None, attn_mask=None, collect=False):
        B, T, C = x.shape
        q, k, v = self.wqkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_dim) : 헤드별로 독립 어텐션
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        if kv_cache is not None:
            # 생성 시: 지난 토큰들의 k, v는 다시 계산하지 않고 캐시에서 이어붙임
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            kv_cache[0], kv_cache[1] = k, v

        if collect:
            # attn_schema 타깃 수집 전용(no_grad 별도 패스): 어텐션 가중치를
            # 직접 계산해 위치별 엔트로피(집중/분산의 요약)를 얻는다. SDPA는
            # 가중치를 반환하지 않으므로 여기서만 수동 계산한다 (본체는 SDPA 유지).
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale        # (B, H, T, T)
            causal = torch.tril(torch.ones(q.size(2), k.size(2),
                                           dtype=torch.bool, device=q.device))
            scores = scores.masked_fill(~causal, float("-inf"))
            attn = scores.softmax(dim=-1)
            ent = -(attn * (attn + 1e-9).log()).sum(-1)       # (B, H, T)
            self.last_entropy = ent.mean(1)                   # (B, T) 헤드 평균
            y = attn @ v
        else:
            # 어텐션 본체: softmax(q·k / sqrt(d)) · v  (PyTorch 내장 고속 구현 사용)
            # is_causal은 q와 k 길이가 같을 때(학습)만. 생성 시(q 길이 1)는
            # 캐시의 과거 토큰 전부를 봐도 되므로 마스크가 필요 없다.
            # attn_mask(True=참조 허용)는 잠재 사고 SFT처럼 접두부를 캐시에 두고
            # 접미부를 병렬 처리할 때(q 길이 ≠ k 길이) 직사각 causal을 직접 준다.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=(attn_mask is None and q.size(2) == k.size(2)),
                dropout_p=self.dropout if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 헤드 합치기
        return self.wo(y)


class SwiGLU(nn.Module):
    """FFN(피드포워드): 어텐션이 모아 온 정보를 토큰별로 가공한다.

    일반 MLP(Linear→ReLU→Linear) 대신, 게이트가 있는 SwiGLU를 쓴다:
        out = W2( silu(W_gate·x) * (W_up·x) )
    silu(W_gate·x)가 "이 정보를 얼마나 통과시킬지"를 조절하는 문 역할.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        if cfg.mood_dim > 0:
            # 기분 벡터 -> scale/shift (FiLM). 제로 초기화(GPT.__init__에서)라
            # 학습 전에는 완전한 항등이어서 기존 체크포인트와 호환된다.
            self.mood_film = nn.Linear(cfg.mood_dim, 2 * cfg.d_model)

    def forward(self, x, rope, kv_cache=None, mood=None, attn_mask=None,
                ws=None, collect=False):
        # residual 연결(x + ...) 덕분에 그래디언트가 깊은 층까지 잘 흐른다
        h = self.attn_norm(x)
        if mood is not None:
            # 기분이 어텐션 입력의 방향/크기를 살짝 비튼다 — residual 자체는
            # 건드리지 않으므로 상태가 이상해도 모델이 망가지지는 않는다
            scale, shift = self.mood_film(mood).chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        if ws is not None:
            # 워크스페이스 전역 방송: 같은 벡터를 모든 위치에 더한다(GWT의
            # "전역 방송"). 턴 내 고정이라 캐시 증분/일괄 결과가 동일하다.
            h = h + ws.unsqueeze(1)
        x = x + self.attn(h, rope, kv_cache, attn_mask, collect=collect)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying: 입력 임베딩과 출력층이 같은 행렬을 공유.
        # "토큰 -> 벡터"와 "벡터 -> 토큰"이 같은 표현을 쓰는 셈이고,
        # 파라미터를 vocab*d_model(약 8M)만큼 아낀다.
        self.lm_head.weight = self.tok_emb.weight

        if cfg.mood_dim > 0:
            # 읽기 헤드: 턴의 은닉 평균 -> 기분 관측 (update_mood에서 사용)
            self.mood_read = nn.Linear(cfg.d_model, cfg.mood_dim)
        if cfg.n_latent > 0:
            # 은닉 상태 -> 다음 잠재 스텝의 입력. "생각을 다시 입력으로 쓸 때
            # 어떤 모양이어야 하는지"를 학습할 수 있게 둔 projection.
            self.latent_proj = nn.Linear(cfg.d_model, cfg.d_model)
        if cfg.feedback:
            # 역피드백: 최상층의 추상 상태(직전 토큰 최종 은닉)를 다음 토큰의
            # 입력단에 방송한다 — "내가 방금 무엇을 생각했는지"를 알고 시작.
            self.feedback_proj = nn.Linear(cfg.d_model, cfg.d_model)
        if cfg.conf_head:
            # 확신도 헤드: 은닉 상태를 읽고 "다음 예측이 맞을 확률"을 출력.
            # 상태를 읽기만 하는 관찰자 — 학습 시 입력을 detach해 본체와 절연.
            self.conf_head = nn.Linear(cfg.d_model, 1)
        if cfg.workspace_slots > 0:
            # 워크스페이스: 슬롯 상태(세션 지속)를 읽어 전역 방송 벡터로 압축(ws_read),
            # 턴 은닉 요약으로 슬롯을 갱신(ws_write). ws_read 제로 init라 켠 직후 항등.
            ws_size = cfg.workspace_slots * (cfg.workspace_dim or cfg.d_model)
            self.ws_read = nn.Linear(ws_size, cfg.d_model)
            self.ws_write = nn.Linear(cfg.d_model, ws_size)
        if cfg.attn_schema:
            # 주의 도식 헤드: 은닉 -> 레이어별 어텐션 엔트로피 요약(K=n_layers).
            # 은닉을 읽기만 하고(detach) 본체에 되먹이지 않아 logits 불변.
            self.attn_schema_head = nn.Linear(cfg.d_model, cfg.n_layers)

        rope = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope", rope, persistent=False)

        self.apply(self._init_weights)
        # residual에 더해지는 projection은 층 수만큼 분산이 커지므로 축소 초기화 (GPT-2 방식)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
        # 항등 초기화: 기능을 켠 직후에도 모델의 출력이 변하지 않게 한다
        for block in self.blocks:
            if hasattr(block, "mood_film"):
                nn.init.zeros_(block.mood_film.weight)
                nn.init.zeros_(block.mood_film.bias)
        if cfg.n_latent > 0:
            with torch.no_grad():
                self.latent_proj.weight.copy_(torch.eye(cfg.d_model))
                nn.init.zeros_(self.latent_proj.bias)
        if cfg.feedback:
            # 제로 초기화: 켠 직후에는 피드백이 0 — 출력이 변하지 않는다
            nn.init.zeros_(self.feedback_proj.weight)
            nn.init.zeros_(self.feedback_proj.bias)
        if cfg.workspace_slots > 0:
            # 제로 init: 켠 직후 방송이 0이라 슬롯 상태가 무엇이든 출력 불변
            nn.init.zeros_(self.ws_read.weight)
            nn.init.zeros_(self.ws_read.bias)

        # 학습/평가에서 loop 반복 횟수를 임시로 고정하는 오버라이드 (None = 기본)
        self._loop_override: int | None = None
        # 직전 generate 턴에서 모은 은닉 평균 (update_mood가 읽는다)
        self._turn_hidden_mean: torch.Tensor | None = None
        self._turn_conf_mean: float | None = None   # 직전 턴 평균 확신도
        self._turn_latent_steps: int = 0             # 직전 턴 잠재 스텝 수
        self._last_schema_loss: torch.Tensor | None = None  # 직전 attn_schema 손실

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # tied weight는 한 번만 센다 (lm_head.weight == tok_emb.weight)
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # 블록 실행 (loop 포함) — forward/generate/SFT가 전부 이 경로를 쓴다
    # ------------------------------------------------------------------
    def _block_order(self, n_loop: int | None = None) -> list[Block]:
        """실행 순서의 블록 리스트. loop 구간은 그룹째로 n번 등장한다.
        예: 8층, loop 2..6, n=2 -> b0 b1 [b2 b3 b4 b5] [b2 b3 b4 b5] b6 b7"""
        s, e = self.cfg.loop_start, self.cfg.loop_end
        n = self.cfg.n_loop if n_loop is None else n_loop
        blocks = list(self.blocks)
        if e > s and n > 1:
            return blocks[:s] + blocks[s:e] * n + blocks[e:]
        return blocks

    def new_caches(self, n_loop: int | None = None) -> list[list]:
        """실행 슬롯(블록×회차)마다 KV 캐시 하나 — 2회차의 k, v는 1회차와
        다르므로 같은 블록이라도 회차 간에 캐시를 공유할 수 없다."""
        return [[None, None] for _ in self._block_order(n_loop)]

    def _run_blocks(self, x, rope, caches=None, mood=None, attn_mask=None,
                    n_loop: int | None = None, ws=None):
        # 워크스페이스 슬롯 상태(B, slots*dim) -> 전역 방송 벡터(B, d_model).
        # 턴 내 한 번만 계산해 모든 블록·모든 회차에 같은 벡터를 방송한다.
        ws_b = self.ws_read(ws) if (ws is not None and self.cfg.workspace_slots > 0) else None
        for i, block in enumerate(self._block_order(n_loop)):
            cache = caches[i] if caches is not None else None
            x = block(x, rope, cache, mood, attn_mask, ws=ws_b)
        return x

    def _embed(self, idx: torch.Tensor, feedback_h: torch.Tensor | None = None):
        """토큰 임베딩 + (켜져 있으면) 역피드백: 위치 t의 입력에 위치 t-1의
        최종 은닉을 projection해 더한다. feedback_h는 1-pass에서 미리 계산한
        은닉 (B, T, C) — detach해서 1-pass로는 그래디언트가 흐르지 않는다."""
        x = self.tok_emb(idx)
        if feedback_h is not None:
            fb = self.feedback_proj(feedback_h[:, :-1].detach())
            x = torch.cat([x[:, :1], x[:, 1:] + fb], dim=1)
        return x

    def hidden_states(self, idx: torch.Tensor, mood=None, feedback_h=None,
                      n_loop: int | None = None, ws=None) -> torch.Tensor:
        """final_norm까지 통과한 은닉 상태 (B, T, C). 로짓이 아니라 내부
        표현이 필요할 때(기분 벡터 학습, 역피드백 1-pass 등) 쓴다."""
        x = self._embed(idx, feedback_h)
        x = self._run_blocks(x, self.rope[:idx.size(1)], mood=mood, ws=ws,
                             n_loop=n_loop if n_loop is not None else self._loop_override)
        return self.final_norm(x)

    def run_from_pos(self, x_emb, pos: int, caches, mood=None, attn_mask=None, ws=None):
        """위치 pos부터의 임베딩을 KV 캐시를 이어 쓰며 통과시킨다.
        잠재 사고 SFT처럼 시퀀스를 (접두부 / 잠재 스텝 / 답변부)로 쪼개
        여러 번에 걸쳐 처리할 때 쓴다. 반환: final_norm 은닉 (B, T, C)."""
        rope = self.rope[pos:pos + x_emb.size(1)]
        x = self._run_blocks(x_emb, rope, caches, mood, attn_mask,
                             n_loop=self._loop_override, ws=ws)
        return self.final_norm(x)

    @torch.no_grad()
    def attention_entropy(self, idx: torch.Tensor) -> torch.Tensor:
        """attn_schema 타깃: 레이어별 어텐션 엔트로피 (B, T, n_layers).

        본체(SDPA)와 별개인 no_grad 패스로, 각 레이어의 위치별 엔트로피를
        수동 계산해 모은다. loop는 무시하고 기본 블록 순서(n_layers개)만 —
        도식 차원 K=n_layers를 일정하게 유지한다."""
        x = self.tok_emb(idx)
        rope = self.rope[:idx.size(1)]
        ents = []
        for block in self.blocks:
            x = block(x, rope, collect=True)
            ents.append(block.attn.last_entropy)   # (B, T)
        return torch.stack(ents, dim=-1)           # (B, T, n_layers)

    # ------------------------------------------------------------------
    # 학습 forward
    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None, mood: torch.Tensor | None = None,
                feedback_h: torch.Tensor | None = None, ws: torch.Tensor | None = None):
        """idx: (B, T) 토큰 ID. targets가 있으면 loss도 함께 반환.

        학습의 전부: 위치 t까지 보고 위치 t+1의 토큰을 맞히는 것.
        targets는 idx를 한 칸 밀어 둔 것이고, loss는 cross entropy다.
        feedback_h: 역피드백 2-pass 학습용 — 1-pass(피드백 없이 병렬)로 구한
        은닉을 주면, 각 위치의 입력에 직전 위치의 은닉이 더해진다.
        ws: 워크스페이스 슬롯 상태 (B, slots*dim) — 전역 방송으로 주입된다.
        """
        B, T = idx.shape
        x = self._embed(idx, feedback_h)
        rope = self.rope[:T]

        n_loop = self._loop_override
        if n_loop is None and self.training and self.cfg.n_loop > 1:
            # 확률적 반복 횟수: {1..n_loop} 균등 샘플. 적은 반복으로도 동작하는
            # 해를 함께 유지해, CPU에서 n_loop=1 고속 모드가 유효해지고
            # "반복 횟수별 성능" 평가도 한 가중치로 할 수 있게 된다.
            n_loop = random.randint(1, self.cfg.n_loop)

        x = self._run_blocks(x, rope, mood=mood, n_loop=n_loop, ws=ws)
        x = self.final_norm(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None  # 생성 시엔 마지막 위치만

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-1, reduction="none",
        )
        if loss_mask is not None:
            # SFT: 사용자 발화는 맞히지 않아도 됨 — 답변 토큰의 loss만 학습
            mask = loss_mask.reshape(-1).float()
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        if self.cfg.conf_head:
            # 메타인지 학습: "방금 그 예측, 맞혔는가?"를 은닉에서 읽어 맞힌다.
            # 라벨은 학습 데이터에 공짜로 들어 있다 (argmax == 정답 여부).
            # x.detach() — 확신도는 상태를 읽기만 하고 본체를 바꾸지 못한다.
            conf_logit = self.conf_head(x.detach()).squeeze(-1)     # (B, T)
            with torch.no_grad():
                correct = (logits.argmax(-1) == targets).float()
            valid = (targets != -1).float()
            if loss_mask is not None:
                valid = valid * loss_mask.float()
            bce = F.binary_cross_entropy_with_logits(conf_logit, correct,
                                                     reduction="none")
            conf_loss = (bce * valid).sum() / valid.sum().clamp(min=1)
            self._last_conf_loss = conf_loss.detach()
            loss = loss + 0.1 * conf_loss

        if self.cfg.attn_schema:
            # 주의 도식 학습: 은닉에서 "레이어별 어텐션 엔트로피"를 회귀한다.
            # 타깃은 별도 no_grad 패스로 수동 계산(detach), 입력도 detach —
            # conf_head와 동형 절연이라 본체 logits는 변하지 않는다.
            tgt = self.attention_entropy(idx) / math.log(max(T, 2))  # (B,T,K) 유계화
            pred = self.attn_schema_head(x.detach())                 # (B,T,K)
            valid = (targets != -1).float().unsqueeze(-1)
            se = ((pred - tgt) ** 2 * valid).sum() / valid.sum().clamp(min=1) / tgt.size(-1)
            self._last_schema_loss = se.detach()
            loss = loss + 0.05 * se

        return logits, loss

    # ------------------------------------------------------------------
    # 기분 벡터
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_mood(self, mood: torch.Tensor, decay: float = 0.9) -> torch.Tensor:
        """직전 generate 턴의 은닉 평균을 기분 관측으로 압축해 EMA로 갱신한다.
            mood <- decay * mood + (1 - decay) * tanh(read(은닉 평균))
        tanh로 유계라 긴 세션에서도 상태가 폭주하지 않는다."""
        if self._turn_hidden_mean is None:
            return mood
        obs = torch.tanh(self.mood_read(self._turn_hidden_mean))
        return decay * mood + (1 - decay) * obs

    @torch.no_grad()
    def update_workspace(self, ws: torch.Tensor, decay: float = 0.95) -> torch.Tensor:
        """턴 종료 시 은닉 요약으로 워크스페이스 슬롯을 EMA 갱신한다(mood와 대칭).
            ws <- decay*ws + (1-decay)*tanh(ws_write(은닉 평균))
        tanh 유계라 긴 세션에서도 슬롯이 폭주하지 않는다. mood보다 decay를
        크게(느리게 변함) 둬 "세션 내내 지속되는 작업공간" 성격을 준다."""
        if self._turn_hidden_mean is None:
            return ws
        obs = torch.tanh(self.ws_write(self._turn_hidden_mean))
        return decay * ws + (1 - decay) * obs

    # ------------------------------------------------------------------
    # 생성
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_p: float = 0.95,
                 stop_ids: set[int] | None = None,
                 mood: torch.Tensor | None = None,
                 n_loop: int | None = None,
                 n_latent: int | None = None,
                 conf_threshold: float | None = None,
                 max_latent: int | None = None,
                 ws: torch.Tensor | None = None,
                 repetition_penalty: float = 1.0):
        """토큰을 하나씩 뽑아 이어 나간다. KV 캐시로 매 스텝 O(전체)가 아닌
        O(새 토큰 1개)만 계산한다. 생성된 토큰 ID를 하나씩 yield.

        n_latent > 0이면 프롬프트를 소화한 직후, 말하기 전에 은닉 상태를
        자기 입력으로 k번 되먹이는 잠재 사고 스텝을 밟는다 (출력 없음).
        mood가 주어지면 매 블록에 FiLM으로 주입된다.
        conf_threshold가 주어지면(확신도 헤드 필요) 확신이 그 밑인 동안
        잠재 스텝을 max_latent까지 추가로 밟는다 — 적응적 사고 시간.

        repetition_penalty > 1이면 이미 등장한 토큰의 logit을 불리하게
        만든다(CTRL 방식: 양수 logit은 나누고 음수는 곱한다). 작은 모델은
        같은 말을 한 번 뱉으면 그것이 다시 다음 토큰의 근거가 되는
        자기강화 루프에 잘 빠지는데, 그 고리를 확률 수준에서 끊는다.
        1.0이면 완전 무변화 — 기존 체크포인트의 동작 보존."""
        self.eval()
        if n_latent is None:
            n_latent = self.cfg.n_latent
        if idx.size(1) > self.cfg.max_seq_len:
            idx = idx[:, -self.cfg.max_seq_len:]  # 문맥 초과분은 앞에서 자름
        caches = self.new_caches(n_loop)
        pos = 0
        hid_sum, hid_cnt = None, 0
        conf_sum = 0.0
        # 반복 페널티 대상: **이 턴에 내가 생성한 토큰만**. 프롬프트는 벌주지
        # 않는다 — 대화에서 프롬프트를 벌주면 사용자가 방금 쓴 낱말을 피하게 되어
        # ("야구 좋아해?" -> "스포츠로 즐겨보자") 화제에 호응하는 것 자체가 막힌다.
        # 막으려는 것은 자기가 뱉은 말이 다시 근거가 되는 자기강화 루프("안녕~"
        # 도배)이고, 그건 생성분만 벌줘도 끊긴다.
        # (generate는 B=1 전제 — next_id.item() 등 기존 코드와 같은 가정)
        seen_ids: set[int] = set()

        def step(x_emb):
            """임베딩 (B, T, C)를 통과시키고 마지막 위치의 final_norm 은닉을 반환."""
            nonlocal pos
            rope = self.rope[pos:pos + x_emb.size(1)]
            h = self._run_blocks(x_emb, rope, caches, mood, n_loop=n_loop, ws=ws)
            pos += x_emb.size(1)
            return self.final_norm(h[:, -1, :])

        if self.cfg.feedback:
            # 역피드백 2-pass (학습과 동일한 방식): 먼저 피드백 없이 병렬로
            # 전 위치의 은닉을 구하고, 그것을 입력에 더한 채 캐시를 채운다
            h_all = self.hidden_states(idx, n_loop=n_loop, ws=ws)
            h_last = step(self._embed(idx, h_all))
        else:
            h_last = step(self.tok_emb(idx))

        # --- 잠재 사고: 은닉 상태를 말 없이 자기 자신에게 되먹인다 ---
        k_thought = 0
        for _ in range(n_latent):
            if pos + 1 > self.cfg.max_seq_len:
                break
            h_last = step(self.latent_proj(h_last).unsqueeze(1))
            k_thought += 1

        # --- 적응적 사고: 확신이 없으면 말하기 전에 더 생각한다 ---
        if conf_threshold is not None and self.cfg.conf_head and self.cfg.n_latent > 0:
            limit = max_latent if max_latent is not None else n_latent + 4
            while (k_thought < limit and pos + 1 <= self.cfg.max_seq_len
                   and torch.sigmoid(self.conf_head(h_last)).mean().item() < conf_threshold):
                h_last = step(self.latent_proj(h_last).unsqueeze(1))
                k_thought += 1
        self._turn_latent_steps = k_thought

        for _ in range(max_new_tokens):
            # 기분 갱신용: 이 턴에서 샘플링에 쓴 은닉들의 평균을 모아 둔다
            hid_sum = h_last.clone() if hid_sum is None else hid_sum + h_last
            hid_cnt += 1
            if self.cfg.conf_head:
                conf_sum += torch.sigmoid(self.conf_head(h_last)).mean().item()

            logits = self.lm_head(h_last)

            # --- 샘플링 ---
            # 반복 페널티: 등장했던 토큰의 logit을 항상 불리한 쪽으로만 민다
            if repetition_penalty != 1.0 and seen_ids:
                ids_t = torch.tensor(sorted(seen_ids), device=logits.device)
                row = logits[0, ids_t]
                logits[0, ids_t] = torch.where(
                    row > 0, row / repetition_penalty, row * repetition_penalty)
            # temperature: 낮으면 확률 높은 토큰에 집중(안전), 높으면 다양(모험)
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            # top-p(nucleus): 누적 확률 p 안에 드는 상위 토큰만 남기고 샘플링
            # — 꼬리의 말도 안 되는 토큰이 뽑히는 사고를 막는다
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            cut = (cum - sorted_probs) > top_p  # 자기 자신 이전까지의 누적으로 판단
            sorted_probs[cut] = 0.0
            choice = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, choice)

            if stop_ids and next_id.item() in stop_ids:
                break
            seen_ids.add(next_id.item())
            yield next_id.item()
            if pos + 1 > self.cfg.max_seq_len:
                break  # 문맥 길이 초과 — 이 미니 모델의 한계선
            x_emb = self.tok_emb(next_id)
            if self.cfg.feedback:
                # 생성은 원래 순차라 진짜 피드백이 공짜다: 방금 그 생각(h_last)을
                # 알고서 다음 토큰의 처리를 시작한다
                x_emb = x_emb + self.feedback_proj(h_last)
            h_last = step(x_emb)

        if hid_cnt:
            self._turn_hidden_mean = hid_sum / hid_cnt
            self._turn_conf_mean = conf_sum / hid_cnt if self.cfg.conf_head else None
