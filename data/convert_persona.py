"""AI Hub '페르소나 대화'(044) -> 페르소나가 붙은 대화 JSONL.

SNS 멀티턴(convert_aihub)과 원본 구조가 다르다. 여기는 대화마다 화자 2명의
**프로필 문장 5개씩**이 딸려 있고, 대화가 실제로 그 프로필에 근거해 흘러간다
("나는 이자카야 사장이다" -> 실제로 가게 얘기를 한다). 이 "프로필 -> 그에
맞는 발화"가 곧 페르소나 학습 신호다.

원본:
    {"info": {"personas": [{"persona_id": 288, "persona": [{"profile": "..."}, ...]}, ...],
              "evaluation": {"grade": "우수"}, "topic": "가족"},
     "utterances": [{"persona_id": 288, "text": "..."}, ...]}

출력 (대화 한 줄):
    {"id": ..., "topic": ..., "grade": ...,
     "personas": [[화자0 프로필 5문장], [화자1 프로필 5문장]],
     "turns": ["화자0 발화", "화자1 발화", ...]}

turns[i]는 personas[i % 2]가 말한 것이다 (역할 배정·미러는 prepare_sft가 결정).
표본 1,489개 전수 확인: persona_id가 항상 일치하고, 엄격히 교대하며, 첫 발화는
언제나 personas[0]의 것이다. 이 불변식이 깨진 대화는 버린다.

라벨링데이터(02.라벨링데이터)만 쓰면 된다 — 원천데이터(01.원천데이터)는 같은
대화의 TSV판이라 중복이다.

사용법:
    python -m data.convert_persona --input "044.페르소나 대화" --out data/raw/persona.jsonl
"""

import argparse
import glob
import json
import os
import zipfile

from data.convert_aihub import _load_json_bytes   # 인코딩 폴백(BOM/CP949) 재사용


def _to_record(cid, obj):
    """원본 대화 dict -> 출력 레코드 (또는 버릴 대화면 None)."""
    info = obj.get("info") or {}
    personas = info.get("personas") or []
    utts = obj.get("utterances") or []
    if len(personas) != 2 or len(utts) < 2:
        return None

    pids = [p.get("persona_id") for p in personas]
    profiles = [[pr.get("profile", "").strip() for pr in (p.get("persona") or [])
                 if pr.get("profile")] for p in personas]
    if not all(profiles) or pids[0] == pids[1]:
        return None

    seq = [u.get("persona_id") for u in utts]
    if any(s not in pids for s in seq):
        return None
    # 첫 발화가 personas[0], 이후 엄격 교대 — 이 정렬이 깨지면 turns[i]와
    # personas[i%2]의 대응이 어긋나 페르소나가 엉뚱한 화자에게 붙는다.
    if seq[0] != pids[0]:
        return None
    if any(seq[i] == seq[i - 1] for i in range(1, len(seq))):
        return None

    turns = [(u.get("text") or "").strip() for u in utts]
    if not all(turns):
        return None

    return {
        "id": cid,
        "topic": info.get("topic", ""),
        "grade": (info.get("evaluation") or {}).get("grade", ""),
        "personas": profiles,
        "turns": turns,
    }


def _iter_raw(input_dir):
    """input_dir 아래 라벨링 JSON들을 (id, obj)로 흘려보낸다 (zip 안도 그대로 읽음).
    원천데이터(.tsv)는 무시된다 — .json만 본다."""
    input_dir = os.path.expanduser(input_dir)
    for fp in glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True):
        try:
            with open(fp, "rb") as f:
                obj = _load_json_bytes(f.read())
        except Exception:
            obj = None
        if obj is not None:
            yield os.path.basename(fp)[:-5], obj
    for zp in glob.glob(os.path.join(input_dir, "**", "*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if not name.endswith(".json"):
                        continue
                    try:
                        obj = _load_json_bytes(z.read(name))
                    except Exception:
                        obj = None
                    if obj is not None:
                        yield os.path.basename(name)[:-5], obj
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="044.페르소나 대화 폴더 (하위 zip을 재귀 탐색)")
    ap.add_argument("--out", default="data/raw/persona.jsonl")
    ap.add_argument("--grade", default="",
                    help="이 등급만 남긴다 (예: 우수). 비우면 전부")
    args = ap.parse_args()

    in_dir = os.path.expanduser(args.input)
    if not os.path.isdir(in_dir):
        raise SystemExit(f"입력 폴더가 없습니다: {in_dir}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_in = n_out = 0
    with open(args.out, "w", encoding="utf-8") as w:
        for cid, obj in _iter_raw(in_dir):
            n_in += 1
            rec = _to_record(cid, obj)
            if rec is None:
                continue
            if args.grade and rec["grade"] != args.grade:
                continue
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    if n_in == 0:
        print(f"⚠️  {in_dir} 안에서 .json/.zip을 못 찾았습니다.")
    print(f"{n_in:,}개 원본 -> {n_out:,}개 대화 -> {args.out}")


if __name__ == "__main__":
    main()
