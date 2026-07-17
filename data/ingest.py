"""임의의 한국어 텍스트 더미 -> corpus.txt (pack.py가 먹는 형식).

download.py는 HuggingFace의 위키/나무위키만 받는다. 스케일업하려면 그걸로는
턱없이 모자라서(둘 다 합쳐 ~6GB) 모두의 말뭉치·AI Hub·OSCAR 같은 걸 직접
구해와야 하는데, 형식이 제각각이다(.txt/.json/.jsonl/.zip, 필드명도 다름).
이 스크립트가 그 잡다한 것들을 하나의 corpus.txt 로 흘려 넣는다.

지원:
  - .txt            : 파일 하나를 문서 하나로 (--split-lines면 줄 하나가 문서)
  - .jsonl / .ndjson: 줄마다 JSON. 텍스트 필드를 자동 탐색
  - .json           : 객체 하나 또는 배열. 중첩도 재귀 탐색
  - .zip            : 위 확장자를 zip 안에서 바로 읽음 (풀지 않는다)
  - 폴더            : 재귀적으로 전부

텍스트 필드는 흔한 이름(text/content/body/sentence/paragraph/발화/문장...)을
순서대로 찾고, 없으면 가장 긴 문자열 값을 쓴다. --field 로 못박을 수도 있다.

품질 필터는 download.py와 같은 계약을 지킨다: 200자 미만 문서는 버리고,
문서 사이에 DOC_SEP을 넣는다 (pack.py가 이 구분자로 문서를 나눈다).

**이어붙이기**: --append 면 기존 corpus.txt 뒤에 붙인다. 여러 출처를 차례로
넣어 하나의 큰 코퍼스를 만드는 것이 스케일업의 정석이다.

사용법:
    python -m data.ingest --input ~/modu_corpus --out data/raw/corpus.txt --append
    python -m data.ingest --input ~/aihub_web.zip --out data/raw/corpus.txt --append
"""

import argparse
import glob
import json
import os
import re
import zipfile

# download.py를 import하면 datasets까지 끌려오므로(무거움/불필요) 계약만 복제한다.
DOC_SEP = "\n\n<<<DOC>>>\n\n"

# 흔한 텍스트 필드 이름 (앞쪽 우선)
TEXT_KEYS = ("text", "content", "body", "paragraph", "sentence", "document",
             "contents", "raw_text", "plain_text", "발화", "문장", "내용", "원문")

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def clean(text: str) -> str:
    """download.clean과 같은 결의 최소 정리 — 과하게 손대지 않는다."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    return _NL.sub("\n\n", text).strip()


def _collect(obj, field: str, out: list) -> None:
    """레코드 안의 텍스트 조각을 재귀로 모은다 (순서 보존)."""
    if isinstance(obj, dict):
        if field:
            v = obj.get(field)
            if isinstance(v, str) and v.strip():
                out.append(v)
        else:
            for k in TEXT_KEYS:            # 알려진 텍스트 키가 있으면 그것만
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    out.append(v)
                    break
        for v in obj.values():             # 중첩 구조는 계속 파고든다
            if isinstance(v, (dict, list)):
                _collect(v, field, out)
    elif isinstance(obj, list):
        for o in obj:
            if isinstance(o, str):
                if not field and o.strip():
                    out.append(o)
            else:
                _collect(o, field, out)


def _loose(obj, out: list) -> None:
    """폴백: 알려진 키를 하나도 못 찾았을 때 모든 문자열을 긁는다."""
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _loose(v, out)
    elif isinstance(obj, list):
        for o in obj:
            _loose(o, out)


def _doc_from_record(rec, field: str) -> str:
    """레코드 하나 -> 문서 하나.

    조각을 따로 내보내지 않고 **합쳐서** 하나의 문서로 만드는 것이 핵심이다.
    대화 데이터는 발화 하나가 수십 자뿐이라 따로 내보내면 길이 필터에 전부
    걸러진다 (실제로 AI Hub 대화 zip에서 66,049개 발화가 전부 버려졌다).
    한 파일/레코드 = 한 대화 = 한 문서로 합쳐야 문맥도 보존된다."""
    if isinstance(rec, str):
        return rec
    texts: list = []
    _collect(rec, field, texts)
    if not texts:
        _loose(rec, texts)
    return "\n".join(t.strip() for t in texts if t and t.strip())


def _records(obj):
    """최상위가 배열이면 원소마다 문서, 아니면 통째로 문서 하나."""
    if isinstance(obj, list):
        yield from obj
    else:
        yield obj


def _decode(raw: bytes) -> str:
    """AI Hub·모두의말뭉치는 UTF-8이 아닌 경우가 흔하다(BOM/CP949)."""
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def _docs_from_bytes(name: str, raw: bytes, field: str, split_lines: bool):
    """파일 하나(바이트) -> 문서들."""
    low = name.lower()
    text = _decode(raw)
    if low.endswith((".jsonl", ".ndjson")):
        for line in text.splitlines():          # 줄 하나 = 레코드 하나 = 문서 하나
            line = line.strip()
            if not line:
                continue
            try:
                yield _doc_from_record(json.loads(line), field)
            except json.JSONDecodeError:
                continue
    elif low.endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        for rec in _records(data):
            yield _doc_from_record(rec, field)
    else:                                   # .txt 등 평문
        if split_lines:
            yield from (ln for ln in text.splitlines() if ln.strip())
        else:
            yield text


def iter_docs(input_path: str, field: str, split_lines: bool):
    """폴더/파일/zip -> 문서 스트림."""
    input_path = os.path.expanduser(input_path)
    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted(glob.glob(os.path.join(input_path, "**", "*"), recursive=True))
        paths = [p for p in paths if os.path.isfile(p)]

    for p in paths:
        if p.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if n.endswith("/"):
                            continue
                        try:
                            yield from _docs_from_bytes(n, z.read(n), field, split_lines)
                        except Exception:
                            continue
            except Exception:
                continue
        elif p.lower().endswith((".txt", ".json", ".jsonl", ".ndjson")):
            try:
                with open(p, "rb") as f:
                    yield from _docs_from_bytes(p, f.read(), field, split_lines)
            except Exception:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="폴더 / 파일 / zip")
    ap.add_argument("--out", default="data/raw/corpus.txt")
    ap.add_argument("--append", action="store_true",
                    help="기존 corpus.txt 뒤에 이어붙인다 (여러 출처를 모을 때)")
    ap.add_argument("--field", default="",
                    help="텍스트 필드명 못박기 (비우면 자동 탐색)")
    ap.add_argument("--split-lines", action="store_true",
                    help=".txt에서 줄 하나를 문서 하나로 취급")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="이보다 짧은 문서는 버린다 (download.py와 같은 기준)")
    args = ap.parse_args()

    in_path = os.path.expanduser(args.input)
    if not os.path.exists(in_path):
        raise SystemExit(f"입력이 없습니다: {in_path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    n_in = n_out = n_chars = 0
    with open(args.out, mode, encoding="utf-8") as w:
        for raw in iter_docs(args.input, args.field, args.split_lines):
            n_in += 1
            text = clean(raw)
            if len(text) < args.min_chars:
                continue
            w.write(text)
            w.write(DOC_SEP)
            n_out += 1
            n_chars += len(text)
            if n_out % 50_000 == 0:
                print(f"  ... {n_out:,}개 문서 ({n_chars/1e9:.2f}GB)", flush=True)

    size = os.path.getsize(args.out) / 1e9
    print(f"{n_in:,}개 원본 -> {n_out:,}개 문서 채택 ({n_chars/1e9:.2f}GB) "
          f"-> {args.out} (파일 총 {size:.2f}GB, mode={mode})")
    if n_out == 0:
        print("⚠️  채택된 문서가 0개입니다. --field 로 텍스트 필드를 지정하거나 "
              "--min-chars 를 낮춰 보세요.")


if __name__ == "__main__":
    main()
