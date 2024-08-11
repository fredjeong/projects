# DDPG

- Link: [baseline](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)

## `ddpg-1`
- 정의된 리워드: $\text{action}[0] * \text{price_diff} - \text{transaction_cost} * |\text{action[0]}| * \text{previous_price}$
    - 이 방식으로 하면 살 때와 팔 때의 reward가 다르잖아?
    - 따라서 reward는 자산가치의 변화 액수로 설정해야 한다.
- transaction cost 반영
- step 수정: 팔 때와 살 때 구분
- replay buffer는 자동으로 반영되는 것 확인했음