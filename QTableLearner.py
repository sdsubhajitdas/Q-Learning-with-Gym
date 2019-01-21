import os
import gym
import platform
from time import sleep, time
import numpy as np


class QTableLearner:

    def __init__(self, env, model_name):
        self.env = env
        self.model_name = model_name
        action_size = self.env.action_space.n
        state_size = self.env.observation_space.n
        self.qtable = np.zeros((state_size, action_size))
        self.avg_time = 0
        self.steps_per_episode = 100

    def _render_logs(self, episode, total_episodes, epsilon, step, action, reward, done, done_count):
        # Clear Screen
        os.system('cls') if platform.system() == \
            'Windows' else os.system('clear')

        # Printing Logs
        print(f'Model Name     :\t{self.model_name}')
        print(f'Q - Table Shape:\t{self.qtable.shape}')
        print(f'Episode Number :\t{episode}/{total_episodes}')
        print(f'Episode Epsilon:\t{epsilon}')
        print(f'Episode Step   :\t{step+1}/100')
        print(f'Episode Action :\t{action}')
        print(f'Episode Reward :\t{reward}')
        print(f'Episode Done ? :\t{"Yes" if done else "No"}')
        print(f'Done Count     :\t{done_count}')

    def _render_env(self, ):
        print()
        self.env.render()

    def _render_time(self, episode_left, episode_t,  step_t, done, step_end, render, wait=0.02):
        if self.avg_time == 0:
            self.avg_time = episode_t
        elif done or step_end:
            self.avg_time = (self.avg_time+episode_t)/2

        time_left = int(self.avg_time*episode_left)
        time_left = (time_left//60, time_left % 60)
        print()
        print(
            f'Time Left            :\t{time_left[0]} mins  {time_left[1]} secs')
        print(f'Average Episode Time :\t{np.round(self.avg_time,4)} secs')
        print(f'Current Episode Time :\t{np.round(episode_t,4)} secs')
        print(f'Current Step Time    :\t{np.round(step_t,4)} secs')
        if render:
            sleep(wait)

    def train(self, train_episodes=10000, test_episodes=1000, lr=0.7, gamma=0.6, render=False):
        (epsilon, max_epsilon, min_epsilon, decay_rate) = (1.0, 1.0, 0.01, 0.01)
        done_count = 0

        t_episode = 0
        for episode in range(train_episodes):
            t_s_episode = time()

            curr_state = self.env.reset()
            curr_step = 0
            episode_done = False
            for curr_step in range(self.steps_per_episode):
                t_s_step = time()
                # Exploration Exploitation Tradeoff for the current step.
                ee_tradeoff = np.random.random()
                # Choosing action based on tradeoff. Random action or action from QTable.
                curr_action = np.argmax(
                    self.qtable[curr_state, :]) if ee_tradeoff > epsilon else self.env.action_space.sample()
                # Taking an action
                new_state, reward, episode_done, info = self.env.step(
                    curr_action)
                # Keeping track of done count
                done_count += 1 if episode_done else 0
                # Rendering Logs
                self._render_logs(episode, train_episodes, epsilon, curr_step, curr_action,
                                  reward, episode_done, done_count)
                # Rendering environment
                if render:
                    self._render_env()
                # Updating QTable using Bellman Equation
                self.qtable[curr_state, curr_action] = \
                    self.qtable[curr_state, curr_action] + lr*(reward + gamma * max(self.qtable[new_state, :]) -
                                                               self.qtable[curr_state, curr_action])
                # Environment state change
                curr_state = new_state

                # Step Time Calculation
                t_step = time() - t_s_step
                self._render_time(train_episodes-episode, t_episode,
                                  t_step, episode_done, self.steps_per_episode-1 == curr_step, render)

                if episode_done:
                    break

            # Updating Epsilon for Exploration Exploitation Tradeoff
            epsilon = min_epsilon + \
                (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

            # Episode Time Calculation
            t_episode = time()-t_s_episode
        self.env.close()

    def test(self, test_episodes=200, render=False):
        self.env.reset()
        rewards = list()
        for episode in range(test_episodes):
            state = self.env.reset()
            step = 0
            done = False
            total_rewards = 0
            for step in range(self.steps_per_episode):
                if render:
                    # Clear Screen
                    os.system('cls') if platform.system() == \
                        'Windows' else os.system('clear')
                    self.env.render()
                    sleep(0.02)
                action = np.argmax(self.qtable[state, :])

                new_state, reward, done, info = self.env.step(action)

                total_rewards += reward
                print ("Score", total_rewards)
                if done:
                    rewards.append(total_rewards)
                    break
                state = new_state
        self.env.close()
        print("Score over time: " + str(sum(rewards)/test_episodes))


if __name__ == "__main__":
    model = 'Taxi-v2'
    #model = 'CartPole-v1'
    #model = 'Blackjack-v0'
    #model = 'FrozenLake-v0'
    #model = 'FrozenLake8x8-v0'
    #model = 'GuessingGame-v0'
    #model = 'HotterColder-v0'
    #model = 'NChain-v0'
    #model = 'Roulette-v0'
    taxi_driver = QTableLearner(gym.make(model), model)
    taxi_driver.train(train_episodes=100,render=False)
    taxi_driver.test(test_episodes=10,render=True)
