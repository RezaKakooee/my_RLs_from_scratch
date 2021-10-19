# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:55:12 2020

@author: RezaKakooee
"""

#%%
import numpy as np
import pandas as pd
from env_map_maker import define_environment_map

#%%
class MdpMetaData:
    def __init__(self, maze_dict, default=5):
    
        self.default = default
        if default == 5:
            self.n_rows = 5
            self.n_cols = 5
            self.start_state = 19
            self.goal_state = 8
            self.actions = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
            self.up_forbidden = [17, 18, 19]
            self.right_forbidden = [6, 11]
            self.down_forbidden = [2, 8, 9]
            self.left_forbidden = [8]
            self.inner_wall = [7, 12, 13, 14]
            
            self.env_dict = {'n_rows': self.n_rows,
                             'n_cols': self.n_cols,
                             'inner_wall': self.inner_wall,
                             'up_forbidden': self.up_forbidden,
                             'down_forbidden': self.down_forbidden,
                             'left_forbidden': self.left_forbidden,
                             'right_forbidden': self.right_forbidden,
                             'start_state': self.start_state,
                             'goal_state': self.goal_state}
            
        elif default == 10:
            self.n_rows = 10
            self.n_cols = 10
            self.start_state = 11
            self.goal_state = 78
            self.actions = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
            self.up_forbidden = [35, 36, 53, 54, 57, 58, 72, 81]
            self.right_forbidden = [23, 33, 41, 46, 51, 61, 64, 70, 74, 84, 94]
            self.down_forbidden = [14, 15, 16, 32, 33, 37, 38, 55, 61]
            self.left_forbidden = [27, 35, 45, 49, 53, 63, 66, 72, 76, 86, 96]
            self.inner_wall = [24, 25, 26, 34, 42, 43, 44, 47, 48, 52, 62, 65, 71, 75, 85, 95]
            
            self.env_dict = {'n_rows': self.n_rows,
                             'n_cols': self.n_cols,
                             'inner_wall': self.inner_wall,
                             'up_forbidden': self.up_forbidden,
                             'down_forbidden': self.down_forbidden,
                             'left_forbidden': self.left_forbidden,
                             'right_forbidden': self.right_forbidden,
                             'start_state': self.start_state,
                             'goal_state': self.goal_state}
        
        else:
            self.n_rows = maze_dict['n_rows']
            self.n_cols = maze_dict['n_cols']
            inner_wall_coords = []
            inner_wall_coords_ = maze_dict['inner_wall_coords']
            for x,y in inner_wall_coords_:
                inner_wall_coords.append([y,x])
            maze_dict['inner_wall_coords'] = inner_wall_coords
            self.env_dict = define_environment_map(maze_dict)
            
            self.env_dict['n_rows'] = maze_dict['n_rows']
            self.env_dict['n_cols'] = maze_dict['n_cols']
            self.inner_wall = self.env_dict['inner_wall']
            self.up_forbidden = self.env_dict['up_forbidden']
            self.down_forbidden = self.env_dict['down_forbidden']
            self.left_forbidden = self.env_dict['left_forbidden']
            self.right_forbidden = self.env_dict['right_forbidden']
            self.start_state = self.env_dict['start_state']
            self.goal_state = self.env_dict['goal_state']
        
        self.actions = {'U': 0, 'R': 1, 'D': 2, 'L': 3}
        self.num_states = self.n_rows*self.n_cols
        self.num_actions = len(self.actions)
        
    def get_action_key(self, action):
        for key, value in self.actions.items(): 
             if action == value: 
                 return key 
             
    def position_to_location(self, position):
        r = int(position / self.n_cols)
        c = position % self.n_cols
        location = (r, c)
        return location
    
    def location_to_position(self, r, c):
        position = r*self.n_cols + c
        return position
    
    def make_position_matrix(self):
        M = np.zeros((self.n_rows, self.n_cols))
        for r in range(self.n_rows):
             for c in range(self.n_cols):
                  M[r, c] = int('{}'.format(self.location_to_position(r, c)))
        M = M.reshape(self.n_rows*self.n_cols, 1)
        M = [int(sp[0]) for sp in M]
        return M

    def make_position_df(self):
        position_matrix = self.make_position_matrix()
        transition_df = pd.DataFrame(index=position_matrix, columns=self.actions.keys())
        for act in self.actions.keys():
            for i, sp in enumerate(position_matrix):
                if sp in self.inner_wall:
                    nasp = np.nan
                elif sp == self.goal_state:
                    nasp = np.nan
                else:
                    (r, c) = self.position_to_location(sp)
                    if act == 'U' and r > 0:
                        if sp not in self.up_forbidden:
                            r -= 1
                    if act == 'R' and c < self.n_cols-1:
                        if sp not in self.right_forbidden:
                            c += 1
                    if act == 'D' and r < self.n_rows-1:
                        if sp not in self.down_forbidden:
                            r += 1  
                    if act == 'L' and c > 0:
                        if sp not in self.left_forbidden:
                            c -= 1
                    nasp = self.location_to_position(r, c)
                transition_df.loc[sp, act] = nasp

        self.deep_copy_transition_df = transition_df.copy()
        
        self.position_state_dict = {p: 'S{:03d}'.format(i)   for i, p in enumerate(transition_df.index)}
        self.state_position_dict = {'S{:03d}'.format(i): p   for i, p in enumerate(transition_df.index)}
        
        return transition_df
    
    def make_mdp(self):
        transition_df = self.make_position_df()
                
        self.T = transition_df.copy() 
        self.R = pd.DataFrame(data=-1*np.ones((self.num_states, self.num_actions)), index=np.arange(self.num_states), columns=np.arange(self.num_actions))
        self.P = pd.DataFrame(data= np.ones((self.num_states, self.num_actions)), index=np.arange(self.num_states), columns=np.arange(self.num_actions))

        for a in range(self.num_actions):
            for s in range(self.num_states):
                position = self.T.iloc[s, a]
                if np.isnan(position):
                    self.R.iloc[s, a] = np.nan
                    self.P.iloc[s, a] = np.nan
                if position == self.goal_state:
                    self.R.iloc[s, a] = 0
        return self.T, self.R, self.P
    
#%% 
def main():
    ### For the default 5*5 maze
    maze_dict = {'n_rows':5, 'n_cols':5, 
                 'inner_wall_coords':[[1,2],[2,2],[2,3],[2,4]], 
                 'startRow':3, 'startCol':4, 
                 'goalRow':1, 'goalCol':3}
    self1 = MdpMetaData(maze_dict)
    T, R, P = self1.make_mdp()
    return self1

#%% 
if __name__ == '__main__':
    self1 = main()    
