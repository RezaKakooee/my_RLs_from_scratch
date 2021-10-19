# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:21:42 2020

@author: RezaKakooee
"""

#%%
import random
import numpy as np

#%%
class Random_Agent():
    def __init__(self, env):
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        
    def get_random_action(self, state):
        action = random.choice(range(self.action_size))
        return action
    
class QAgent(Random_Agent):
    def __init__(self, env, gamma=0.97, alpha=0.01, epsilon=1.0):
        super().__init__(env)
    
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        self.build_model()
    
    def build_model(self):
        self.q_table = 1e-4 * np.random.random([self.state_size, self.action_size])
        self.q_table[self.env.goal_state, :] = 0
        for w in self.env.inner_wall:
            self.q_table[w, :] = np.nan
        
    def get_action(self, state, policy='e_greedy'):
        q_state = self.q_table[state]
        if policy == 'e_greedy':
            action_greedy = np.argmax(q_state)
            action_random = random.choice(range(self.action_size))
            action = action_random if random.random() < self.epsilon else action_greedy
        return action
    
    def train(self, experience):
        state, action, next_state, reward, done = experience
        q_next = self.q_table[next_state] * (1 - done)
        q_max = np.max(q_next)
        q_target = reward + self.gamma * q_max
        q_update = q_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * q_update
        
        if done:
            self.epsilon = max(0.1, 0.9998 * self.epsilon)


            