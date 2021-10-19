# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 03:01:18 2020

@author: RezaKakooee
"""

#%%
import numpy as np

from agents import QAgent
from gridworld_environment import Environment

#%%
def test(env, agent, n_episodes=2, render=True):
    total_reward = []
    n_episodes = 2
    for ep in range(n_episodes):
        print('========================================')
        print('---------- The episode number is: ', ep)
        episode_reward = []
        state = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            action = np.argmax(agent.q_table[state])
            next_state, reward, done, info = env.step(action)
            experience = (state, action, next_state, reward, done)
            # print('Experience: ', experience)
            episode_reward.append(reward)
            
            if render:
                env.render()
                
            state = next_state
            
            if done:
                print("Episode finished after {} timesteps".format(t))
                break
        total_reward.append(np.sum(episode_reward))
        print('Episode Reward: {}'.format(np.sum(episode_reward)))

#%%
if __name__ == '__main__':
    env = Environment(default=5)
    agent = QAgent(env)
    agent.q_table = np.load('q_table.npy')
    test(env, agent, n_episodes=2, render=True)