"""검증 하네스(D1~D6) 공용 로더.

각 eval 도구가 체크포인트/데이터를 똑같이 여는 코드를 반복하지 않도록
최소한의 헬퍼만 모았다. 도구 본체의 "무엇을 재는가"는 각 파일에 남긴다.
"""

from pathlib import Path

import numpy as np
import torch

from model.gpt import GPT, ModelConfig


def load_model(ckpt: str, device: str):
    """체크포인트 -> (model, cfg). 체크포인트가 자기 설정을 내장하므로
    플래그 없이 올바른 모델이 복원된다. loop는 SFT처럼 최대치로 고정."""
    ck = torch.load(ckpt, map_location=device)
    cfg = ModelConfig(**ck["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    model._loop_override = cfg.n_loop
    return model, cfg


def load_sft_val(data: str, val_frac: float = 0.02):
    """SFT .npz 를 열어 학습에 안 쓴 검증 예시 인덱스를 돌려준다.
    반환: (val_idx, boundaries, ids, mask, pad_id)."""
    d = np.load(data)
    ids, mask, boundaries = d["ids"], d["mask"], d["boundaries"]
    n = len(boundaries) - 1
    n_val = max(int(n * val_frac), 1)
    val_idx = np.arange(n - n_val, n)
    return val_idx, boundaries, ids, mask, int(ids[0])


def require(cond, msg):
    """대응 기제가 꺼진 체크포인트엔 친절히 알려 준다."""
    if not cond:
        raise SystemExit(f"[eval] {msg}")
