"""바이트 수준 BPE(Byte-Pair Encoding) 토크나이저 — 직접 구현.

핵심 아이디어
============
LLM은 문자를 직접 다루지 않고 "토큰"이라는 정수 ID의 나열을 다룬다.
BPE는 그 토큰 사전을 데이터로부터 만드는 알고리즘이다:

  1. 모든 텍스트를 UTF-8 바이트(0~255)로 본다. → 기본 토큰 256개
  2. 코퍼스에서 가장 자주 붙어 나오는 토큰 쌍을 찾는다. (예: '하'+'다')
  3. 그 쌍을 새 토큰 하나로 합친다(merge). → 사전 크기 +1
  4. 원하는 사전 크기(vocab_size)가 될 때까지 2~3을 반복한다.

한국어 글자 하나는 UTF-8로 3바이트이므로, 학습 초반의 merge는
바이트 조각을 글자로 복원하는 일을 하고, 그다음부터 '입니다', '그리고'
같은 의미 단위가 토큰으로 자라난다. train() 후 vocab을 출력해 보면
한국어가 어떻게 토큰으로 묶이는지 직접 볼 수 있다.

속도를 위한 두 가지 장치 (원리는 그대로)
- 학습: 코퍼스 전체가 아니라 "고유한 청크(단어 조각) + 등장 횟수" 테이블
  위에서 merge를 수행하고, merge 때마다 영향을 받은 청크의 쌍 카운트만
  증분 갱신한다. (전체를 매번 다시 세면 vocab 16k 학습이 불가능할 만큼 느림)
- 인코딩: 같은 청크는 항상 같은 토큰열이 되므로 청크 단위로 캐싱한다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import regex as re  # 표준 re와 달리 \p{L}(모든 문자), \p{N}(모든 숫자)을 지원

# ---------------------------------------------------------------------------
# 사전 분할(pre-tokenization) 패턴
# ---------------------------------------------------------------------------
# BPE를 텍스트 전체에 그냥 돌리면 "다.그리고" 처럼 단어 경계를 넘는 토큰이
# 생겨 품질이 나빠진다. 그래서 먼저 텍스트를 "청크"로 자르고,
# merge는 청크 내부에서만 일어나게 한다. (GPT-2/4가 쓰는 방식의 단순화판)
#   " ?\p{L}+"            : 앞 공백 하나를 포함한 연속된 문자(한글·영문 등)
#   " ?\p{N}+"            : 숫자
#   " ?[^\s\p{L}\p{N}]+"  : 구두점·기호
#   "\s+"                 : 남은 공백(개행 등)
SPLIT_PATTERN = re.compile(r" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")

# 대화 형식을 표현하기 위한 특수 토큰. 일반 텍스트에서는 절대 만들어지지
# 않도록 merge로 생성하지 않고 사전 끝에 별도 ID로 붙인다.
# 뒤의 8개는 "마음 유사 기제"용 + 예약분 — vocab을 나중에 또 수술하지 않도록
# 미리 자리를 잡아 둔다. (vocab 16392 = 256 바이트 + 16124 merge + 특수 12개.
# 예전 16384/특수 4개 설정과 merge 수가 같아, 같은 코퍼스면 같은 merge를 배운다)
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>",
                  "<|pause|>", "<|thought|>", "<|mood|>", "<|sys|>",
                  "<|res0|>", "<|res1|>", "<|res2|>", "<|res3|>"]


class BPETokenizer:
    def __init__(self):
        # merges[(a, b)] = c : "토큰 a와 b가 붙어 있으면 새 토큰 c로 합쳐라"
        # 등록된 순서(= c가 작은 순서)가 곧 우선순위다.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab[id] = 해당 토큰이 나타내는 바이트열 (디코딩에 사용)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}  # 문자열 -> id
        self._encode_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """text로부터 vocab_size 크기의 토큰 사전을 학습한다."""
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges > 0, "vocab_size는 256 + 특수토큰 수보다 커야 한다"

        # 1) 사전 분할 후, 고유 청크와 등장 횟수를 센다.
        #    "안녕하세요"가 십만 번 나와도 merge 계산은 한 번만 하면 된다.
        chunk_freq = Counter(SPLIT_PATTERN.findall(text))

        # 각 고유 청크를 바이트 ID 리스트로 변환
        words: list[list[int]] = []   # words[i] = i번째 고유 청크의 현재 토큰열
        freqs: list[int] = []         # freqs[i] = 그 청크의 등장 횟수
        for chunk, f in chunk_freq.items():
            words.append(list(chunk.encode("utf-8")))
            freqs.append(f)

        # 2) 초기 쌍 카운트. pair_where[pair]는 그 쌍을 포함한 청크 인덱스 집합
        #    — merge 후 이 청크들만 다시 세면 되므로 전체 재계산이 필요 없다.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: dict[tuple[int, int], set[int]] = {}
        for i, w in enumerate(words):
            f = freqs[i]
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                pair_where.setdefault(pair, set()).add(i)

        # 3) 가장 흔한 쌍을 골라 합치기를 n_merges번 반복
        for step in range(n_merges):
            if not pair_counts:
                break  # 더 합칠 쌍이 없음 (코퍼스가 너무 작을 때)
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + step
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose and (step % 500 == 0 or step == n_merges - 1):
                print(f"merge {step + 1}/{n_merges}: {best} -> {new_id}"
                      f" ({self.vocab[new_id]!r}, count={pair_counts[best]})")

            # best 쌍을 포함했던 청크들만 갱신한다 (증분 업데이트의 핵심)
            for i in list(pair_where.get(best, ())):
                w, f = words[i], freqs[i]
                # 갱신 전 이 청크가 기여하던 쌍 카운트를 빼고
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    s = pair_where.get(pair)
                    if s is not None:
                        s.discard(i)
                # 청크 안의 best 쌍을 new_id로 치환한 뒤
                words[i] = w = self._merge_word(w, best, new_id)
                # 갱신 후의 쌍 카운트를 다시 더한다
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    pair_where.setdefault(pair, set()).add(i)

        # 4) 특수 토큰을 사전 맨 뒤에 등록
        next_id = 256 + len(self.merges)
        for tok in SPECIAL_TOKENS:
            self.special_tokens[tok] = next_id
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1

    @staticmethod
    def _merge_word(w: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """토큰열 w 안의 모든 pair를 new_id 하나로 치환한다."""
        out = []
        i = 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(w[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # 인코딩 / 디코딩
    # ------------------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> list[int]:
        """청크 하나를 토큰열로. 학습 때 배운 merge를 배운 순서대로 적용한다."""
        cached = self._encode_cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # 현재 토큰열에 존재하는 쌍 중, 가장 먼저 배운(=우선순위 높은) 쌍
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break  # 적용할 merge가 더 없음
            ids = self._merge_word(ids, best, self.merges[best])
        if len(self._encode_cache) < 500_000:  # 메모리 폭주 방지
            self._encode_cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """일반 텍스트 -> 토큰 ID 리스트. (특수 토큰 문자열은 해석하지 않음)"""
        ids: list[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode_special(self, token: str) -> int:
        """특수 토큰 문자열의 ID를 돌려준다. (예: '<|user|>')"""
        return self.special_tokens[token]

    def has_special(self, token: str) -> bool:
        """이 토크나이저가 해당 특수 토큰을 갖고 있는지 (구버전 사전 호환용)."""
        return token in self.special_tokens

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 리스트 -> 텍스트. 토큰별 바이트를 이어붙인 뒤 UTF-8 해석."""
        data = b"".join(self.vocab[i] for i in ids)
        # 생성 도중 잘린 한글(3바이트 중 일부)이 있어도 죽지 않도록 replace
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # 저장 / 불러오기
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        obj = {
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        for a, b, c in obj["merges"]:
            tok.merges[(a, b)] = c
            tok.vocab[c] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = obj["special_tokens"]
        for s, i in tok.special_tokens.items():
            tok.vocab[i] = s.encode("utf-8")
        return tok
