import gym
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')  # Updated from v0 to v1
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env,
                  n_actions=env.action_space.shape[0])

    n_games = 250
    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation, _ = env.reset()  # Unpack observation from reset()
            action = env.action_space.sample()
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, new_observation, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, evaluate)
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.remember(observation, action, reward, new_observation, done)
            if not load_checkpoint:
                agent.learn()
            observation = new_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'episode {i} score {score:.1f} avg_score {avg_score:.2f}')

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
