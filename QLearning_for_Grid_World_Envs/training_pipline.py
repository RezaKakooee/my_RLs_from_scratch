# -*- coding: utf-8 -*-
"""
Created on Mon May  4 06:31:30 2020

@author: RezaKakooee
"""
#%%
from gridworld_environment import Environment
from plot_environment import PlotEnvironment
from agents import QAgent
from train_QAgent import train
from test_QAgent import test
import visutils

#%% Instantiate the Environment class
## You can use deafual 5*5 or 10*10 mazez, or instantiate the Environment class with an arbitrary maze
### for deafutl
env = Environment(default=5)


### for an arbitrary maze
# maze_dict = {'n_rows':8, 'n_cols':8, 
#             'inner_wall_coords':[[1,2],[1,3],[1,4],[1,5],[2,2],[3,2],[3,3],[3,4],[3,5],[5,2],[5,3],[5,4],[6,4], [7,2],[7,3],[7,4]], 
#             'startRow':6, 'startCol':3, 
#             'goalRow':2, 'goalCol':3}
# env = Environment(maze_dict, default=None)
# PlotEnvironment(env.env_dict).show_image()

#%% Instantiate the agent
agent = QAgent(env) 

#%% Train Q-Learning Agent
q_table, total_reward, obs_history = train(env, agent, n_episodes=1000, render=False)

#%% Plot the total reward per episode
visutils.plot_reward(total_reward)

#%% Check the agent performance in some episodes
visutils.plot_obs_history(env, obs_history)

#%% Test the trained Q-Learning Agent
agent.q_table = q_table
test(env, agent, n_episodes=2, render=False)