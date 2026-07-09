"""사전학습용 한국어 코퍼스 다운로드 → data/raw/corpus.txt 로 저장.

소스 (§B1):
  - wiki : 한국어 위키피디아 (wikimedia/wikipedia 20231101.ko). 백과적·정제됨.
  - namu : 나무위키 정제판 (heegyu/namuwiki-extracted). 말동무/서브컬처 결.
           원본 heegyu/namuwiki는 [[...]]·{{{...}}}·== == 마크업이 그대로라
           노이즈가 많다 — 정제판을 쓴다. 라이선스 cc-by-nc-sa-2.0(비상업).
  - mix  : 위 둘을 문서 단위로 교차 스트리밍 (--mix-ratio 로 비율 조절).

문서 사이는 pack.py 단계에서 <|endoftext|> 토큰으로 구분되므로, 여기서는
문서를 \n\n<<<DOC>>>\n\n(DOC_SEP)로 구분해 저장한다 (pack.py 계약).

사용법:
    python -m data.download                              # 위키 전체
    python -m data.download --source mix --max-docs 5000 # 위키+나무 혼합(검증)
    python -m data.download --source namu                # 나무위키만
"""

import argparse
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DOC_SEP = "\n\n<<<DOC>>>\n\n"  # pack.py가 이 구분자로 문서를 나눈다

# 나무위키 정제판에 드물게 남는 잡음. 과하게 지우면 구어체가 상하므로 최소한만.
_RE_FILE_LINK = re.compile(r"\[\[(?:파일|분류|틀):[^\]]*\]\]")  # [[파일:...]] 류
_RE_TRIPLE_BRACE = re.compile(r"\{\{\{[^}]*\}\}\}")            # {{{...}}} 잔재
_RE_MULTISPACE = re.compile(r"[ \t]{2,}")
_RE_MULTINL = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    """가벼운 정제 — 남은 마크업/과잉 공백만 정리하고 문장은 보존한다."""
    text = _RE_FILE_LINK.sub("", text)
    text = _RE_TRIPLE_BRACE.sub("", text)
    # [[문서명|표시text]] -> 표시text, [[문서명]] -> 문서명 (링크는 풀되 내용 유지)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = _RE_MULTISPACE.sub(" ", text)
    text = _RE_MULTINL.sub("\n\n", text)
    return text.strip()


def iter_docs(source: str):
    """소스별 텍스트 문서를 (source_tag, text)로 흘려 준다 (streaming)."""
    if source in ("wiki", "mix"):
        ds = load_dataset("wikimedia/wikipedia", "20231101.ko",
                          split="train", streaming=True)
        if source == "wiki":
            for d in ds:
                yield "wiki", d["text"]
            return
        wiki_it = ((("wiki", d["text"]) for d in ds))
    if source in ("namu", "mix"):
        nds = load_dataset("heegyu/namuwiki-extracted", split="train",
                           streaming=True)
        if source == "namu":
            for d in nds:
                yield "namu", d["text"]
            return
        namu_it = ((("namu", d["text"]) for d in nds))
    if source == "mix":
        yield from _interleave(wiki_it, namu_it)
        return
    raise ValueError(f"알 수 없는 source: {source}")


def _interleave(wiki_it, namu_it, ratio=1.0):
    """위키:나무 = 1:ratio 로 문서 단위 교차. 한쪽이 소진되면 나머지를 흘린다."""
    import itertools
    wiki_done = namu_done = False
    while not (wiki_done and namu_done):
        if not wiki_done:
            try:
                yield next(wiki_it)
            except StopIteration:
                wiki_done = True
        for _ in range(max(1, round(ratio))):
            if namu_done:
                break
            try:
                yield next(namu_it)
            except StopIteration:
                namu_done = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="wiki", choices=["wiki", "namu", "mix"])
    ap.add_argument("--out", default="data/raw/corpus.txt")
    ap.add_argument("--max-docs", type=int, default=0, help="0이면 전체")
    ap.add_argument("--mix-ratio", type=float, default=1.0,
                    help="mix에서 위키 1개당 나무위키 문서 수 (기본 1:1)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # mix-ratio를 반영해 이터레이터를 다시 구성 (iter_docs는 기본 1:1)
    if args.source == "mix" and args.mix_ratio != 1.0:
        wds = load_dataset("wikimedia/wikipedia", "20231101.ko",
                           split="train", streaming=True)
        nds = load_dataset("heegyu/namuwiki-extracted", split="train",
                           streaming=True)
        docs = _interleave((("wiki", d["text"]) for d in wds),
                           (("namu", d["text"]) for d in nds), ratio=args.mix_ratio)
    else:
        docs = iter_docs(args.source)

    counts = {"wiki": 0, "namu": 0}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for tag, raw in tqdm(docs, desc=f"downloading[{args.source}]"):
            text = clean(raw)
            if len(text) < 200:  # 너무 짧은 문서(리다이렉트 등)는 버림
                continue
            f.write(text)
            f.write(DOC_SEP)
            counts[tag] += 1
            n += 1
            if args.max_docs and n >= args.max_docs:
                break

    # 라이선스 고지 (§B1)
    ds_path = out.parent / "DATASOURCES.txt"
    with open(ds_path, "w", encoding="utf-8") as f:
        f.write(f"source={args.source}\n")
        f.write(f"docs: wiki={counts['wiki']:,}, namu={counts['namu']:,}\n\n")
        if counts["wiki"]:
            f.write("wikimedia/wikipedia 20231101.ko — CC BY-SA\n")
        if counts["namu"]:
            f.write("heegyu/namuwiki-extracted — cc-by-nc-sa-2.0 (비상업 개인용)\n")

    size_mb = out.stat().st_size / 1e6
    print(f"완료: {n:,}개 문서(wiki {counts['wiki']:,}/namu {counts['namu']:,}), "
          f"{size_mb:.0f}MB -> {out}")
    print(f"라이선스 고지: {ds_path}")


if __name__ == "__main__":
    main()
