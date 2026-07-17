"""코퍼스 품질 필터 — 웹 크롤을 사전학습에 넣기 전에 거르는 최소 방어선.

왜 필요한가: mC4/C4 한국어는 영어판과 달리 정제가 거의 없다. 실제로
c4-ko 첫 문서가 성인물 스팸이었다. 사전학습은 "본 것을 닮는" 과정이라,
스팸이 섞이면 모델이 스팸 말투를 배운다 — 말동무를 만들려는 목적과 정면으로
충돌한다. 34M~1B의 작은 모델일수록 데이터 품질에 더 민감하다(볼 수 있는
토큰이 적으니 한 토큰의 질이 그만큼 무겁다).

거창한 분류기를 쓰지 않는다. 값싸고 설명 가능한 규칙 몇 개로 최악을 걷어내는
것이 목적이다 (Gopher/C4 계열이 쓰는 것과 같은 결):
  - 한글 비율   : 한국어 코퍼스인데 한글이 적으면 메뉴·코드·영문 스팸이다
  - 스팸 키워드 : 성인·도박 스팸은 소수의 낱말로 대부분 잡힌다
  - 줄 반복     : 같은 줄이 반복되면 네비게이션 바·자동생성 페이지다
  - 길이        : 너무 짧으면 문맥을 배울 게 없다

과하게 거르면 멀쩡한 데이터까지 날아가므로 임계는 넉넉하게 잡았다.
"""

import re

_HANGUL = re.compile(r"[가-힣]")
_ALNUM = re.compile(r"[0-9A-Za-z가-힣]")

# c4-ko 실측(1,500편)에 근거한 목록이다. 짐작이 아니라 실제 빈도를 세어 잡았다:
#   출장 9.1% · 안마 7.5% · 마사지 7.4% · 카지노 7.3% · 바카라 5.3% · 토토 5.0%
# 대략 문서의 10~15%가 성인·도박 스팸이었다.
#
# 두 단계로 나눈 이유: '출장'(출장 = 업무 출장)·'성인'(성인교육)·'마사지'(건강
# 기사)·'노출'(위험 노출)은 멀쩡한 한국어 낱말이다. 1회 등장으로 자르면 정상
# 문서가 대량으로 날아간다. 그래서 스팸에만 쓰이는 복합어는 1회로 자르고,
# 양쪽에 쓰이는 낱말은 2회 이상 반복될 때만 스팸으로 본다.

# 1회만 나와도 스팸으로 본다 — 정상 문서에 나올 이유가 없는 복합어
_SPAM_HARD = re.compile(
    r"출장안마|출장샵|출장마사지|안마방|립카페|조건만남|성인용품|성인방송"
    r"|야동|무료야동|야설|벗방|급딸|폰팅|음란물"
    r"|먹튀검증|사설토토|토토사이트|바카라사이트|카지노사이트|먹튀사이트"
    r"|비아그라|시알리스|발기부전"
)

# 2회 이상 반복될 때만 스팸 — 정상 문맥에도 쓰이는 낱말들
_SPAM_SOFT = re.compile(
    r"카지노|바카라|토토|배팅|슬롯|홀덤|먹튀|룸살롱|유흥"
    r"|출장|안마|마사지|섹시|19금|자위|섹스"
)


def hangul_ratio(text: str) -> float:
    """글자(숫자·영문·한글) 중 한글의 비율. 한국어 문서면 보통 0.5+."""
    n = len(_ALNUM.findall(text))
    if n == 0:
        return 0.0
    return len(_HANGUL.findall(text)) / n


def line_repetition(text: str) -> float:
    """중복된 줄의 비율. 네비게이션·자동생성 페이지는 이 값이 높다."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def is_spam(text: str) -> bool:
    """성인·도박 광고인가. 복합어는 1회, 일반 낱말은 2회 이상으로 판정."""
    if _SPAM_HARD.search(text):
        return True
    return len(_SPAM_SOFT.findall(text)) >= 2


def keep(text: str, min_chars: int = 200, min_hangul: float = 0.3,
         max_repetition: float = 0.5) -> bool:
    """이 문서를 사전학습 코퍼스에 넣을 것인가."""
    if len(text) < min_chars:
        return False
    if hangul_ratio(text) < min_hangul:
        return False
    if is_spam(text):
        return False
    if line_repetition(text) > max_repetition:
        return False
    return True
