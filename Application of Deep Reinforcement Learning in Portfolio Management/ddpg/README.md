# DDPG

## Structure
- `__init__.py`
- `environment.py`
- `model.py`
- `train.py`
- `test.py`
- `utilities.py`

## Notes
- DQN에서의 성공의 기반이 되었던 주요 아이디어들을 채용해 continuous action 도메인으로 옮긴 것
    - replay buffer
    - soft target network: 복제하는 것
- 추가적인 특징
    - batch normalisation
    - noise process


### Replay Buffer

기존 학습 방식에는 학습하는 순서가 어느 정도 정해져 있기 때문에 Exploration의 랜덤성과 균일성이 떨어지는데, replay buffer에 모든 내용을 step에 저장하고 실제 학습은 replay buffer에서 random하게 뽑아쓰는 방식을 택하여 이와 같은 문제점을 해결하여 agent 학습 성능을 높여주었다. (단, buffer는 clear되지 않으며 buffer를 키우고 오래된 데이터에 덮어쓰는 방식으로 학습이 진행됩니다. )

=> Exploration이 정해진 방식으로 탐색되는 고질적인 문제를 벗어나는 핵심 기법!

### Target network
1. Actor Critic을 사용
2. 고정된 정답지인 Actor-critic을 또 같고 있음 (총 2+2 = 4 개의 신경망 사용)
3. 고정된 정답지를 서서히 업데이트

Loss function의 경우
-> Critic은 정답지와 자신의 예측치를 줄이는 방식으로
-> Actor는 Critic의 감정값을 극대화하는 방식으로 학습

### Action noise

Exploration을 잘하기 위해 Action noise를 추가해주는데, 이는 action에다가 random값을 더 해주는 것이다. OU process를 이용하여 ε-Greedy 방식으로 값이 나올 수 있도록 하는 것이다.


state_action_values = 이건 Critic을 써야하지 않나?
그리고 네트워크는 총 4개임 actor의 target, policy 그리고 critic의 target, policy