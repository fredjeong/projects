# DQN Agent

## Structure
- `environment.py`
- `model.py`
- `train.py`
- `test.py`
- `utility.py`

## Memo
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
https://pylessons.com/RL-BTC-BOT-NN

13 Aug 22:25 window와 replay buffer가 무슨 차이가 있는지 확인해볼 것. 차이가 없으면 굳이 필요하지 않으니 지우기. memory가 하는 일, 저장하는 step의 개수 파악

13 Aug 23:38 에이전트는 매번 다른 방식으로 행동하기 때문에 여러 번 테스트를 해서 그 평균을 구하는 것이 이상적이다. 아니면 평균 +- 표준편차로 그래프를 그리거나.