# DQN

## Input
- state: fiat, asset, price

## Structure
- `environment.py`: OpenAI Gym을 이용해 강화학습에 필요한 환경 세팅
- `train.py`: 모델 훈련
- `run.py`: 모델 테스트
- `network.py`: state를 입력으로 받아 action을 출력하는 인공신경망

DiscreteEnv reward를 수익금으로 변경, 파산시 페널티 -100에서 -1000으로 변경
DiscreteEnv reward를 비율로 변경
- reward = (current_value - previous_value) / current_value

리워드는 기본적으로 가격이 내려갈 것 같을 때 팔고, 오를 것 같을 때 사야 한다.
이걸 어떻게 반영해야 하지?

지금 문제는.. 매수와 매도 시 보상 함수가 똑같다는 것.
가격이 오를 것 같으면 사도록 하고, 가격이 내릴 것 같으면 팔도록 유도해야 하는데 이게 쉽지가 않다.

