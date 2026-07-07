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
"""

from __future__ import annotations

import math
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

    def forward(self, x, rope, kv_cache=None):
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

        # 어텐션 본체: softmax(q·k / sqrt(d)) · v  (PyTorch 내장 고속 구현 사용)
        # is_causal은 q와 k 길이가 같을 때(학습)만. 생성 시(q 길이 1)는
        # 캐시의 과거 토큰 전부를 봐도 되므로 마스크가 필요 없다.
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(q.size(2) == k.size(2)),
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

    def forward(self, x, rope, kv_cache=None):
        # residual 연결(x + ...) 덕분에 그래디언트가 깊은 층까지 잘 흐른다
        x = x + self.attn(self.attn_norm(x), rope, kv_cache)
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

        rope = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope", rope, persistent=False)

        self.apply(self._init_weights)
        # residual에 더해지는 projection은 층 수만큼 분산이 커지므로 축소 초기화 (GPT-2 방식)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # tied weight는 한 번만 센다 (lm_head.weight == tok_emb.weight)
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                loss_mask: torch.Tensor | None = None):
        """idx: (B, T) 토큰 ID. targets가 있으면 loss도 함께 반환.

        학습의 전부: 위치 t까지 보고 위치 t+1의 토큰을 맞히는 것.
        targets는 idx를 한 칸 밀어 둔 것이고, loss는 cross entropy다.
        """
        B, T = idx.shape
        x = self.tok_emb(idx)
        rope = self.rope[:T]
        for block in self.blocks:
            x = block(x, rope)
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
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_p: float = 0.95,
                 stop_ids: set[int] | None = None):
        """토큰을 하나씩 뽑아 이어 나간다. KV 캐시로 매 스텝 O(전체)가 아닌
        O(새 토큰 1개)만 계산한다. 생성된 토큰 ID를 하나씩 yield."""
        self.eval()
        caches = [[None, None] for _ in self.blocks]
        pos = 0
        x_in = idx  # 첫 스텝: 프롬프트 전체, 이후: 직전 생성 토큰 1개

        for _ in range(max_new_tokens):
            if pos + x_in.size(1) > self.cfg.max_seq_len:
                break  # 문맥 길이 초과 — 이 미니 모델의 한계선
            x = self.tok_emb(x_in)
            rope = self.rope[pos:pos + x_in.size(1)]
            for block, cache in zip(self.blocks, caches):
                x = block(x, rope, cache)
            pos += x_in.size(1)
            logits = self.lm_head(self.final_norm(x[:, -1, :]))

            # --- 샘플링 ---
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
            yield next_id.item()
            x_in = next_id
