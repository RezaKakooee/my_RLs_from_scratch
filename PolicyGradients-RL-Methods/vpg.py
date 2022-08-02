# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:20:44 2021

@author: Reza Kakooee
"""
# %%
import time
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# %%
class SimpleMLP(nn.Module):
    def __init__(self, model_config):
        super(SimpleMLP, self).__init__()
        
        layer_sizes_stack = [model_config['obs_dim']] + model_config['hiden_sizes']+ [model_config['n_actions']]
        num_layers = len(layer_sizes_stack)
        
        input_activation = model_config['input_activation']
        hiden_activation = model_config['hiden_activation']
        output_activation = model_config['output_activation']
        
        def input_layer():
            return [nn.Linear(layer_sizes_stack[0],
                              layer_sizes_stack[1], 
                              input_activation()
                              )
                    ]
    
        def hiden_layers():
            h_layers = []
            for i in range(1, num_layers-2):
                h_layers += [nn.Linear(layer_sizes_stack[i], 
                                       layer_sizes_stack[i+1], 
                                       hiden_activation()
                                       )
                            ]
            return h_layers

        def output_layer():
            return [nn.Linear(layer_sizes_stack[-2], 
                              layer_sizes_stack[-1], 
                              output_activation()
                              )
                    ]
                          
        layers_stack = input_layer() + hiden_layers() + output_layer()
        self.mlp = nn.Sequential(*layers_stack)
        
    def forward(self, X):
        return self.mlp(X)

# %%
class VPG:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def build_model(self):
        model = SimpleMLP(self.config['model_config'])
        return model
        
    def get_policy(self, obs):
        logits = self.model(torch.as_tensor(obs, dtype=torch.float32))
        dist = Categorical(logits=logits)
        return dist
        
    def select_action(self, obs):
        pi = self.get_policy(obs)
        action =  pi.sample().item()
        return action
    
    def compute_loss(self, obs, act, rew):
        dist = self.get_policy(obs)
        logp = dist.log_prob(act)
        loss = -(logp * rew).mean()     
        return loss
    
    def train_one_epoch(self, batch_obss, batch_acts, batch_gains):
        self.model.train()
        batch_loss = self.compute_loss(obs=torch.as_tensor(batch_obss, dtype=torch.float32),
                                       act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                       rew=torch.as_tensor(batch_gains, dtype=torch.float32))
        
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()


# %%            
class PGLearner:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.batch_size = config['batch_size']
        self.discount_factor = config['discount_factor']
        
        self.agent = VPG(config)
    
    
    def learn(self, n_epochs):
         for i in range(n_epochs):
             print(f"- - - - - - - - - - - - - - - - - - - - - - - - - Epoch {i}")
             batch_obss, batch_acts, batch_gains = self._interact()
             self.agent.train_one_epoch(batch_obss, batch_acts, batch_gains)
    
             
    def _interact(self):
        batch_obss = []
        batch_acts = []
        batch_rets = []
        batch_lens = []
        batch_rews = []
        batch_mean_rews = []
        batch_gains = []
        
        count_ep = 0
        while True: # for batch_size            
            done = False
            obs = self.env.reset()
            episode_reward = []
            i = 1
            while not done:
                
                action = self.agent.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                
                batch_obss.append(obs)
                batch_acts.append(action)
                batch_rews.append(reward)
                episode_reward.append(reward)
                
                obs = next_obs.copy()
                i += 1
                
            
            count_ep += 1
            
            batch_lens.append(len(episode_reward))
            batch_rets.append( sum([ self.discount_factor**(i-1) * r for i, r in enumerate(episode_reward) ]) )
            episode_mean_rews = np.mean(episode_reward)
            batch_mean_rews.append(episode_mean_rews)
            rew2go = self._calculate_reward_to_go(episode_reward)
            batch_gains += rew2go
            
            print(f"----- Episode: {count_ep}, EpisodeLen: {i}, EpisodeMeanReward: {episode_mean_rews}")
            
            if len(batch_acts) >= self.batch_size:
                break
        
        print(f"********** This batch contains {count_ep} episodes.")
        return batch_obss, batch_acts, batch_gains
    

    def _calculate_reward_to_go(self, rewards):
        sum_reward = 0
        discounted_rewards = []
        rewards.reverse()
        for r in rewards:
          sum_reward = r + self.discount_factor*sum_reward
          discounted_rewards.append(sum_reward)
        discounted_rewards.reverse()
        return discounted_rewards   

            
    def inference(self):
        done = False
        obs = self.env.reset()
        reward_list = []
        episode_reward = []
        i = 0
        while not done:
            action = self.agent.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            
            episode_reward.append(reward)
            env.render()
            obs = next_obs.copy()
            time.sleep(0.5)
            i += 1
        reward_list.append(sum(episode_reward))
        
        print(f",,,,,,,, Inference Episode Len: {i}")        
        print(f".................... Inference Reward: {reward_list}")
        
# %%
class VPGConfig:
    def __init__(self, env):
        self.env = env
        
        self.model_config = {
            'obs_dim': env.observation_space.shape[0],
            'n_actions': env.action_space.n,
            'hiden_sizes': [32, 32],
            'input_activation': nn.Tanh,
            'hiden_activation': nn.ReLU,
            'output_activation': nn.Identity, #nn.Softmax,
            }
        
        self.batch_size = 200
        self.discount_factor = 0.99
        
    
    def get_config(self):
        return self.__dict__           
        
    
# %%        
if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v0')
    
    config = VPGConfig(env).get_config()
    
    learner = PGLearner(env, config)
    learner.learn(n_epochs=1000)
    learner.inference()
    env.close()
    
        
        
        
            
        
        
    
    
