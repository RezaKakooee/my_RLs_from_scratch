# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:25:59 2020

@author: RezaKakooee
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plot_environment import PlotEnvironment

#%%
def plot_obs_history(env, obs_history):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    view = PlotEnvironment(env.env_dict)
    n_episodes = len(obs_history.keys())
    view_episodes = np.sort(np.random.choice(np.arange(0,n_episodes), size=24, replace=False))
    obs_history_x = {ep:[] for ep in view_episodes}
    obs_history_y = {ep:[] for ep in view_episodes}
    obs_history_yx = {ep:[] for ep in view_episodes}
    for ep in obs_history.keys():
        if ep in view_episodes:
            obs_history_yx[ep] = [view.tp2cl(s) for s in obs_history[ep]]
            obs_history_y[ep] = [coords[0] for coords in obs_history_yx[ep]]
            obs_history_x[ep] = [coords[0] for coords in obs_history_yx[ep]]
    bsize = env.n_rows * env.n_cols
    chessboard = np.zeros((env.n_rows, env.n_cols))
    counter = 0
    fig, ax = plt.subplots(figsize=(15,10))
    fig.suptitle(f"Agent performance in some episodes of the training phase")
    rect_w = 1
    rect_h = 1
    rect_shift_x = 0
    rect_shift_y = 0
    for ep in view_episodes:
        counter += 1
        ax = plt.subplot(4,6,counter)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, env.n_rows)
        plt.ylim(0, env.n_cols)
        plt.title(f"Episode: {ep}")
        c = ax.pcolor(chessboard, edgecolors='k', linewidths=3, cmap='binary')
        
        for t in range(len(env.inner_wall)):    
            rect_xy = view.wall_cell_coords[t]
            COLOR = '#FF4500'
            rect = patches.Rectangle((rect_xy[1]+rect_shift_x, rect_xy[0]+rect_shift_y), rect_w, rect_h, linewidth=1, edgecolor='r', facecolor=COLOR)
            ax.add_patch(rect)
        # for ep in view_episodes:
        for s in range(len(obs_history_yx[ep])-1):
            arrow_start = obs_history_yx[ep][s]
            arrow_stop = obs_history_yx[ep][s+1]
            diff = (arrow_stop[0] - arrow_start[0], arrow_stop[1] - arrow_start[1])
            COLOR = (np.random.rand(), np.random.rand(), np.random.rand())
            if not np.array_equal(arrow_start, arrow_stop):
                plt.arrow(arrow_start[1]+0.5, arrow_start[0]+0.5, diff[1], diff[0], head_width=0.2, head_length=0.2, fc=COLOR, ec=COLOR)
                
        arrow_start = obs_history_yx[ep][-1]
        arrow_stop = view.tp2cl(env.goal_state)
        diff = (arrow_stop[0] - arrow_start[0], arrow_stop[1] - arrow_start[1])
        COLOR = (np.random.rand(), np.random.rand(), np.random.rand())
        if not np.array_equal(arrow_start, arrow_stop):
            plt.arrow(arrow_start[1]+0.5, arrow_start[0]+0.5, diff[1], diff[0], head_width=0.2, head_length=0.2, fc=COLOR, ec=COLOR)
            
    plt.draw()

def plot_reward(total_reward):
    n_episodes = len(total_reward)
    # moving_avg_episode_reward = [np.mean(total_reward[0:i]) for i in range(1,n_episodes)]
    fig = plt.figure(figsize=(12,8))
    plt.plot(np.arange(n_episodes), total_reward)
    plt.xlabel('Episode')
    # plt.ylabel('Moving Average Reward')
    plt.ylabel('Total Reward per Episode')
    plt.title(f"Agent performance the training phase")
    plt.show()