"""HuggingFace 데이터셋 -> corpus.txt (스트리밍, 한국어 소스 레시피 포함).

download.py는 위키/나무위키만 받는다(합쳐 ~6GB). 스케일업(xl=8B토큰≈30GB,
xxl=20B≈80GB)에는 턱없이 모자라서 여기서 더 긁어온다.

**스트리밍**이 핵심이다. c4-ko는 전체가 수십 GB인데 통째로 받으면 디스크와
시간을 다 잡아먹는다. streaming=True로 흘려 읽으면서 필요한 만큼(--max-gb)만
쓰고 끊는다.

품질 필터(data/quality.py)를 통과한 문서만 쓴다 — 웹 크롤은 그대로 넣으면
스팸 말투를 배운다.

사용법:
    # 무엇이 있는지 보기
    python -m data.download_hf --list
    # 교과서(고품질)부터 5GB
    python -m data.download_hf --sources textbooks --out data/raw/corpus_32k.txt --append
    # 웹 크롤로 부피 채우기 (필터 통과분 기준 20GB)
    python -m data.download_hf --sources c4-ko --max-gb 20 --out data/raw/corpus_32k.txt --append
    # 임의의 저장소도 가능
    python -m data.download_hf --repo someone/ko-corpus --out ... --append
"""

import argparse
import os
import re
import time

from data.ingest import DOC_SEP, clean, _doc_from_record
from data import quality

# 각 소스가 무엇이고 왜 쓰는지. all_configs=True면 그 저장소의 모든 서브셋을 돈다.
RECIPES: dict[str, dict] = {
    "textbooks": dict(
        repo="maywell/korean_textbooks", all_configs=True,
        note="교과서체 합성 한국어(42개 서브셋). 토큰당 품질이 가장 높다 — 작은 모델일수록 유리",
    ),
    "personas": dict(
        repo="nvidia/Nemotron-Personas-Korea",
        note="합성 페르소나 서술. 깨끗한 한국어 산문 + 페르소나 실험과 직결",
    ),
    "c4-ko": dict(
        repo="allenai/c4", data_files="multilingual/c4-ko.*.json.gz",
        note="웹 크롤. 부피를 채우는 유일한 현실적 소스지만 잡음이 많다(필터 필수)",
    ),
    "instructions": dict(
        repo="heegyu/open-korean-instructions", strip_markup=True,
        note="지시-응답 모음. <usr>/<bot> 마크업을 대화체로 풀어 넣는다",
    ),
    "dialog": dict(
        repo="jungsungmoon/Korean_dialog",
        note="과제형 대화(주문 등). 양은 적지만 대화 결에 도움",
    ),
}

_MARKUP = re.compile(r"<usr>|<bot>|<sys>|<user>|<assistant>")


def _iter_dataset(repo: str, config: str | None, data_files, split: str):
    from datasets import load_dataset
    kw = dict(split=split, streaming=True)
    if data_files:
        kw["data_files"] = {split: data_files}
    if config:
        return load_dataset(repo, config, **kw)
    return load_dataset(repo, **kw)


def _configs_for(repo: str, all_configs: bool) -> list[str | None]:
    if not all_configs:
        return [None]
    from datasets import get_dataset_config_names
    try:
        return list(get_dataset_config_names(repo))
    except Exception:
        return [None]


def harvest(spec: dict, writer, budget_bytes: int, min_chars: int,
            no_filter: bool) -> tuple[int, int]:
    """한 소스를 흘려 읽으며 corpus에 쓴다. -> (채택 문서 수, 쓴 바이트)."""
    repo = spec["repo"]
    written = kept = seen = 0
    t0 = time.time()
    for config in _configs_for(repo, spec.get("all_configs", False)):
        if written >= budget_bytes:
            break
        label = f"{repo}" + (f":{config}" if config else "")
        try:
            ds = _iter_dataset(repo, config, spec.get("data_files"),
                               spec.get("split", "train"))
        except Exception as e:
            print(f"  [skip] {label} — {type(e).__name__}: {str(e)[:90]}")
            continue
        try:
            for rec in ds:
                seen += 1
                text = _doc_from_record(rec, spec.get("field", ""))
                if spec.get("strip_markup"):
                    text = _MARKUP.sub("\n", text)
                text = clean(text)
                ok = (len(text) >= min_chars if no_filter
                      else quality.keep(text, min_chars=min_chars))
                if not ok:
                    continue
                writer.write(text)
                writer.write(DOC_SEP)
                kept += 1
                written += len(text.encode("utf-8"))
                if kept % 20_000 == 0:
                    print(f"    ... {label}: {kept:,}편 / {written/1e9:.2f}GB "
                          f"({time.time()-t0:.0f}s)", flush=True)
                if written >= budget_bytes:
                    break
        except Exception as e:                      # 스트림 중간 오류는 치명적이지 않다
            print(f"  [중단] {label} — {type(e).__name__}: {str(e)[:90]}")
    rate = 100 * kept / seen if seen else 0
    print(f"  {repo}: {seen:,}편 중 {kept:,}편 채택({rate:.0f}%) / {written/1e9:.2f}GB")
    return kept, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="",
                    help="쉼표 구분 (예: textbooks,personas,c4-ko). --list로 목록 확인")
    ap.add_argument("--repo", default="", help="레시피에 없는 임의의 HF 저장소")
    ap.add_argument("--config", default="")
    ap.add_argument("--field", default="", help="텍스트 필드 못박기 (비우면 자동 탐색)")
    ap.add_argument("--out", default="data/raw/corpus_32k.txt")
    ap.add_argument("--append", action="store_true", help="기존 파일 뒤에 이어붙인다")
    ap.add_argument("--max-gb", type=float, default=0,
                    help="소스마다 이만큼(필터 통과분 기준)만 받고 끊는다. 0=제한 없음")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--no-filter", action="store_true",
                    help="품질 필터 끄기 (이미 깨끗한 소스일 때만)")
    ap.add_argument("--list", action="store_true", help="레시피 목록만 출력")
    args = ap.parse_args()

    if args.list:
        print("사용 가능한 소스:\n")
        for k, v in RECIPES.items():
            print(f"  {k:14s} {v['repo']}\n                 {v['note']}\n")
        return

    specs: list[dict] = []
    if args.repo:
        specs.append(dict(repo=args.repo, config=args.config or None,
                          field=args.field))
    for name in filter(None, (s.strip() for s in args.sources.split(","))):
        if name not in RECIPES:
            raise SystemExit(f"모르는 소스: {name} (--list 로 확인)")
        specs.append(RECIPES[name])
    if not specs:
        raise SystemExit("--sources 또는 --repo 를 주세요 (--list 로 목록 확인)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    budget = int(args.max_gb * 1e9) if args.max_gb else 10 ** 18
    mode = "a" if args.append else "w"

    total_kept = total_bytes = 0
    with open(args.out, mode, encoding="utf-8") as w:
        for spec in specs:
            print(f"\n▶ {spec['repo']} 수집 시작 "
                  f"(예산 {args.max_gb or '무제한'}GB, 필터 {'off' if args.no_filter else 'on'})")
            k, b = harvest(spec, w, budget, args.min_chars, args.no_filter)
            total_kept += k
            total_bytes += b

    size = os.path.getsize(args.out) / 1e9
    print(f"\n총 {total_kept:,}편 / {total_bytes/1e9:.2f}GB 추가 "
          f"-> {args.out} (파일 전체 {size:.2f}GB)")


if __name__ == "__main__":
    main()
