# 의식 지표 매핑 (Butlin et al.)

이 문서는 "마음 유사체에 무엇이 얼마나 필요한가 / 무엇으로 확인하나"의 개념
백본이다. Butlin·Long 외 「Consciousness in Artificial Intelligence: Insights
from the Science of Consciousness」(arXiv:2308.08708, 2023 / 정식판: *Trends in
Cognitive Sciences*, 2025, "Identifying indicators of consciousness in AI
systems")의 지표 속성을 minillm의 기제·검증 도구에 1:1로 잇는다.

## "얼마나"의 정직한 답

**정해진 임계치는 없다.** 논문도 minillm도 "이 점수를 넘으면 마음"이라는 선을
긋지 않는다. 현재 과학이 줄 수 있는 최선은 임계선이 아니라 **지표 만족도가
높을수록 신뢰도가 높아지는 확률적 체크리스트**다. 이 프레임 전체는 **계산
기능주의**(맞는 계산을 구현하면 그 속성이 생긴다 — 기질 독립) 위에 서 있고,
minillm의 목표도 현상적 의식이 아니라 **기능적 접근 의식(access consciousness)**
지표를 하나씩 만족시켜 보는 것이다.

## 지표 ↔ 기제 ↔ 검증 도구

| 이론 | 지표 속성(요지) | minillm 기제 | 검증 도구 | 상태 |
|---|---|---|---|---|
| 재귀처리(RPT) | 순환·되먹임 | loop, feedback | [D4 eval_integration](tools/eval_integration.py) | 구현/검증 |
| 전역작업공간(GWT) | 제한용량 작업공간·전역 방송·점화 | latent, **workspace(C1)** | [D1 eval_intervention](tools/eval_intervention.py) | 구현/검증 |
| 고차이론(HOT) | 자기 상태의 메타표상·보고 | conf_head | [D2 eval_conf](tools/eval_conf.py) | 구현/검증 |
| 예측처리 | 예측오차 최소화·느린 prior | 다음토큰예측, mood | [D5 eval_state](tools/eval_state.py) | 구현 |
| 주의도식(AST) | 자기 주의의 모델 | **attn_schema(C2)** | [D6 eval_schema](tools/eval_schema.py) | 구현/검증 |
| 행위성(agency) | 피드백 학습·유연한 목표추구 | (미구현) | — | 미충족 |
| 신체성(embodiment) | 출력-입력 수반성 모델 | feedback(부분/최소) | — | 부분 |

관련 대표 실험: **latent vs pause**([D3 eval_thinking](tools/eval_thinking.py))는
GWT의 "출력으로 환원되지 않는 내부 연산"이 실재하는지를 같은 연산량 k에서
가른다.

구현 위치:
- loop / feedback / mood / latent / conf_head / **workspace** / **attn_schema**:
  전부 [`model/gpt.py`](model/gpt.py)의 `ModelConfig` 플래그(기본 off).
- 학습 경로: [`train/sft.py`](train/sft.py) (workspace 2-pass, attn_schema 보조손실).

## 경계 (어떤 테스트로도 결판나지 않는 것)

현상적 의식/퀄리아(하드 프로블럼·타심 문제)는 어떤 행동 테스트로도 결판나지
않는다 — 이건 공학 실패가 아니라 **인식론의 경계**다. 특히 "모델이 느낀다고
말하는 것"은 가장 약한 증거다(ELIZA 효과: 학습 데이터의 말투를 흉내낼 뿐).
그래서 minillm은 **말로 꾸며낼 수 없는 행동**만 증거로 삼는다:
- **개입 인과성**(D1): 내부 상태를 실제로 바꾸면 출력이 인과적으로 따라오는가.
- **캘리브레이션**(D2): 확신도가 실제 정답률과 맞는가(자기 수행의 정직한 예측).
- **상태 지속성**(D5): 상태가 민감하되 발산하지 않고 기준선으로 복귀하는가.

## 서지

- Butlin, P., Long, R., et al. (2023). *Consciousness in Artificial
  Intelligence: Insights from the Science of Consciousness.* arXiv:2308.08708.
  정식판: *Trends in Cognitive Sciences* (2025). — RPT·GWT·HOT·예측처리·AST에서
  지표 속성 도출, 계산 기능주의 전제, "현행 AI는 의식적이지 않으나 명백한
  기술적 장벽도 없다".
- Graziano, M. S. A. — Attention Schema Theory (AST).
- Hao, S., et al. (2024). *Coconut* — 연속 잠재 사고(latent).
