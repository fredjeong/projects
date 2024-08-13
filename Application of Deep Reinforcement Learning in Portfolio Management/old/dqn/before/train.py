import numpy as np
import pandas as pd
import torch
from dqn.before.agent import Agent
from testenv2 import TradingEnv
from utils import History, Portfolio, TargetPortfolio
import matplotlib.pyplot as plt
from collections import deque

def train_dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window)>=200.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def plot_scores(scores):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(scores)), scores)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('DQN Training Scores')
    plt.show()

if __name__ == "__main__":
    # 데이터를 로드하고 전처리하는 코드가 여기에 들어가야 합니다.
    df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
    
    env = TradingEnv(df)
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)
    
    scores = train_dqn(agent, env)
    
    # 훈련 결과를 시각화하는 코드
    plot_scores(scores)