# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:20:44 2021

@author: Reza Kakooee
"""
# %%
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
        return pi.sample().item()
    
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
        self.batch_size = config['model_config']['batch_size']
        
    def calculate_return(self, rewards):
        sum_reward = 0
        discounted_rewards = []
        rewards.reverse()
        for r in rewards:
          sum_reward = r + self.dicount_factor*sum_reward
          discounted_rewards.append(sum_reward)
        discounted_rewards.reverse()
        return discounted_rewards   
    
    def interact(self):
        batch_obss = []
        batch_acts = []
        batch_rets = []
        batch_lens = []
        batch_rews = []
        batch_gains = []
        
        while True:
            
            done = False
            obs = self.env.reset()
            episode_reward = []
            while not done:
                action = self.agent.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                
                batch_obss.append(obs)
                batch_acts.append(action)
                batch_rews.append(reward)
                episode_reward.append(reward)
                
                obs = next_obs.copy()
            
            batch_lens.append(len(reward))
            batch_rets.append(sum(episode_reward))            
            if len(batch_acts) >= self.batch_size:
                break
        
        return batch_obss, batch_acts, batch_rews
                
                
    def learn(self, n_epochs):
         for i in range(n_epochs):
             batch_obss, batch_acts, batch_rews = self.interact()
             self.train_one_epoch(batch_obss, batch_acts, batch_rews)
             
# %%
class VPGConfig:
    def __init__(self, env):
        self.env = env
        
        self.model_config = {
            'obs_dim': env.observation_space.shape,
            'n_actions': env.action_space.n,
            'hiden_sizes': [32 , 32],
            'input_activation': nn.Tanh,
            'hiden_activation': nn.ReLU,
            'output_activation': nn.Softmax,
            }
        
        self.discount_factor = 0.99
        
    
    def get_config(self):
        return self.__dict__           
        
    
# %%        
if __name__ == "__main__":
    import gym
    env = gym.make('CartPole-v0')
    
    config = VPGConfig(env).get_config()
    
    learner = PGLearner(env, config)
    learner.learn(n_epochs=1)
    
    
    
        
        
        
            
        
        
    
    
