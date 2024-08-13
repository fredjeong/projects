import numpy as np
import pandas as pd

data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
prices = df['close'].values
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]
test_prices = prices[train_size:]


def valuation(fiat, asset, price):
    return fiat + asset * price

class Environment():
    # agent alance state:
    state_dim = 3 # fiat, asset, price

    # 포지션 정보
    sell = 0 # 매수
    hold = 1 # 매도
    buy = 2

    # 수수료 (0.015%)
    transaction_fee = 0.00015

    # 행동 종류
    action_sell = 0
    action_hold = 1
    action_buy = 2
    
    # action space
    actions = [action_sell, action_hold, action_buy]
    num_actions = len(actions)

    def __init__(self, initial_fiat = 20000, prices):
        # 초기 자본금 설정
        self.initial_fiat = initial_fiat
        
        # 차트 정보
        self.prices = prices
        # self.training_data
        self.observation = None
        self.idx = -1

        # 잔고 내역 및 거래 정보
        self.fiat = initial_fiat # 현재 한금 잔고
        self.asset = 0 # 보유 코인 수
        self.portfolio_value = valuation(self.fiat, self.asset, self.prices[0]) # 현재 포트폴리오 가치
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 홀드 횟수

        # balance: agent의 state 정보
        self.hold_ratio = 0 # 주식 보유 비율
        self.profitloss = 0 # 현재 손익
        self.avg_position_price = 0 # 주당 매수 단가
        self.position = 1 # 현재 포지션

    def reset(self):
        self.observation = None
        self.idx = -1
        self.portfolio_value = self.initial_fiat
        self.fiat = self.initial_fiat
        self.asset = 0
        self.hold_ratio = 0
        self.position = 1
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0

    def observe(self):
        if len(self.prices) > self.idx + 1:
            self.prices[self.idx]
            self.observation = self.prices[self.idx]
            return self.observation
        else:
            return None

    # 결정된 action(매수, 매도)를 수행할 수 있는 최소 조건을 확인
    def validate_action(self, action):
        if action == Environment.action_sell: # 매도
            if self.asset <= 0:
                return False
        elif action == Environment.action_hold: # 홀드
            pass
        elif action == Environment.action_buy: # 매도
            if self.fiat <= 0:
                return False

    def act(self, action):
        '''
        action을 수행하고 환경 정보를 업데이트
        input: policy에 출력된 action
        output: 현재까지 수익률
        '''
        if not self.validate_action(action):
            action = Environment.action_hold
        current_price = self.prices[self.idx]

        if action == Environment.action_sell:
            pass
        elif action == Environment.action_hold:
            pass
        elif action == Environment.action_buy:
            pass

    # input: agent's action space
    # output: observed_price, observed_fiat, reward, done, info
    def step(self, action=None, policy=None):
        previous_value = valuation(self.fiat, self.asset, self.prices[self.idx])

        self.idx += 1
        if self.idx >= len(self.prices) - 1:
            done = True
            return None, None, 0, done, None

        current_price = self.prices[self.idx]

        if action == None:
            pass
        else:

            reward = valuation(self.fiat, self.asset, self.prices[self.idx]) - previous_value
            return 

        observation = self.observe()
        if (np.array(observation) == None).all():
            done = True
            return None, None, 0, done, None:
        # 훈련 시작 전 초기 데이터 반환
        if action == None:
            return np.array([self.fiat, self.asset, prices[self.idx]])