"""AI Hub '한국어 SNS 멀티턴 대화'(dataSetSn=71694) -> 범용 대화 JSONL.

AI Hub 원본은 대화 하나당 JSON 파일 하나이고, 구조는:
    {"info": {...}, "utterances": [{"speaker": "speakerA", "text": "...", ...}, ...]}

이 스크립트는 그 잡다한 메타데이터를 걷어내고, 우리가 쓰기 좋은 최소 형식으로
줄인다 (대화 한 줄):
    {"id": "000337", "turns": ["첫 발화", "둘째 발화", ...]}

turns는 speakerA로 시작하는 원래 순서 그대로다 (역할 배정은 prepare_sft가
--mirror로 결정한다 — 여기서는 화자 중립적으로 순서만 보존).

설계 결정 (근거는 표본 분석):
  - 3인(speakerC) 대화는 버린다 (~2.8%). 1:1 말동무에는 A/B만.
  - 연속 동일 화자 발화(희귀)는 공백으로 합쳐 엄격한 교대를 유지한다 —
    prepare_sft의 user/assistant 교대 조립이 어긋나지 않게.

입력은 .json 파일들이 있는 폴더, 또는 .zip 파일(들)이 있는 폴더 모두 받는다
(zip은 풀지 않고 안에서 바로 읽는다 — AI Hub는 zip이 여러 개라 편의상).

사용법:
    python -m data.convert_aihub --input ~/aihub_multiturn --out data/raw/sns_convos.jsonl
"""

import argparse
import glob
import json
import os
import zipfile


def _clean_conversation(obj):
    """원본 대화 dict -> turns 리스트 (또는 버릴 대화면 None)."""
    us = obj.get("utterances") or []
    if len(us) < 2:
        return None
    speakers = {u.get("speaker") for u in us}
    if len(speakers) != 2:
        return None  # 3인+ 또는 화자 정보 이상 → 버림

    turns, roles = [], []
    for u in us:
        text = (u.get("text") or "").strip()
        spk = u.get("speaker")
        if not text:
            continue
        if roles and roles[-1] == spk:
            turns[-1] = turns[-1] + " " + text     # 연속 동일화자는 합침
        else:
            turns.append(text)
            roles.append(spk)
    if len(turns) < 2:
        return None
    return turns


def _iter_raw(input_dir):
    """input_dir 아래의 모든 대화 JSON을 (id, obj)로 흘려보낸다.
    .json 파일과 .zip 안의 .json 멤버를 모두 다룬다."""
    jsons = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    for fp in jsons:
        try:
            with open(fp, encoding="utf-8") as f:
                yield os.path.basename(fp)[:-5], json.load(f)
        except Exception:
            continue
    zips = glob.glob(os.path.join(input_dir, "**", "*.zip"), recursive=True)
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if not name.endswith(".json"):
                        continue
                    try:
                        with z.open(name) as f:
                            yield os.path.basename(name)[:-5], json.load(f)
                    except Exception:
                        continue
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="AI Hub JSON들(또는 zip들)이 있는 폴더")
    ap.add_argument("--out", default="data/raw/sns_convos.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_in = n_out = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for cid, obj in _iter_raw(args.input):
            n_in += 1
            turns = _clean_conversation(obj)
            if turns is None:
                continue
            w.write(json.dumps({"id": cid, "turns": turns}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"{n_in:,}개 원본 -> {n_out:,}개 대화 (3인·이상 제외) -> {args.out}")


if __name__ == "__main__":
    main()
