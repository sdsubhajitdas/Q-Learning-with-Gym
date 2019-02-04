import gym
import os
from QTableLearner import QTableLearner


def main():
    env_list = [  # 'Roulette-v0',
        # 'FrozenLake-v0',
        'Taxi-v2',
        # 'FrozenLake8x8-v0'
    ]

    for env in env_list:
        agent = QTableLearner(gym.make(env), env)
        agent.load_model(os.path.join('saved_models', env+'.npy'))
        agent.set_refresh_time(0.1, True)
        agent.test(20, render=True)


if __name__ == "__main__":
    main()
