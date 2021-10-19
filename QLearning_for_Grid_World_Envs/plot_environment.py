# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:27:00 2020

@author: RezaKakooee
"""
#%%
import time
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#%%
class PlotEnvironment:
    def __init__(self, env_dict):
        
        self.n_rows = env_dict['n_rows']
        self.n_cols = env_dict['n_cols']
        self.bsize = self.n_rows * self.n_cols
        self.start_state = env_dict['start_state']
        self.goal_state = env_dict['goal_state']
        self.inner_wall = env_dict['inner_wall']
        self.n_blocked_cells = len(self.inner_wall)
        self.EVEN_COLOR = '#FF4500' #'#FF7F50'
        self.ODD_COLOR = '#FF4500' 
        self.rect_w = 1
        self.rect_h = 1
        self.rect_shift_x = 0
        self.rect_shift_y = 0
        
        self.chessboard = np.zeros((self.n_cols, self.n_cols))
        self.cartesian_mat = np.flip(np.arange(self.bsize).reshape(self.n_cols, self.n_rows), axis=0)
        
        self.convert_table_potision_to_cartesian_position()
        
        self.fig = None
        self.ax = None
        self.frame_counter = 0
        
#     def __del__(self):
#         plt.close(self.fig)

    def reset_frame_counter(self):
        self.frame_counter = 0
        
    def p2c(self, p):
        return  int(p / self.n_cols), p % self.n_cols
    
    def tp2cl(self, p): #table position - table location - cartesian location - cartesian position
        # return  int(p / self.n_cols), p % self.n_cols
        y,x = self.p2c(p)
        return self.p2c(self.cartesian_mat[y][x])
    
    def tp2cp(self, p): 
        y,x = self.p2c(p)
        return self.cartesian_mat[y][x]
    
    def convert_table_potision_to_cartesian_position(self):
        r,c = self.p2c(self.start_state)
        self.start_state = self.cartesian_mat[r][c]
        r,c = self.p2c(self.goal_state)
        self.goal_state = self.cartesian_mat[r][c]
        W = []
        for w in self.inner_wall:
         r,c = self.p2c(w)
         W.append(self.cartesian_mat[r][c])
        self.inner_wall = W
        self.wall_cell_coords = [self.p2c(self.inner_wall[i]) for i in range(self.n_blocked_cells)]
        
    def show_image(self):
        if self.n_rows == 5:
            self.fig, self.ax = plt.subplots(figsize=(5,5))
        else:
            self.fig, self.ax = plt.subplots(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])
        c = self.ax.pcolor(self.chessboard, edgecolors='k', linewidths=3, cmap='binary')
        
        for p in range(self.bsize):
          plt.text(self.tp2cl(p)[1]+0.55, self.tp2cl(p)[0]+0.55, str(p), fontsize=12)
        
        plt.text(self.p2c(self.start_state)[1]+0.32, self.p2c(self.start_state)[0]+0.32, 'S', fontsize=28, color='red', fontweight='bold')
        plt.text(self.p2c(self.goal_state)[1]+0.32, self.p2c(self.goal_state)[0]+0.32, 'G', fontsize=28, color='green', fontweight='bold')
        
        for t in range(self.n_blocked_cells):    
            rect_xy = self.wall_cell_coords[t]
            COLOR = self.EVEN_COLOR if t % 2 == 0 else self.ODD_COLOR 
            rect = patches.Rectangle((rect_xy[1]+self.rect_shift_x, rect_xy[0]+self.rect_shift_y), self.rect_w, self.rect_h, linewidth=1, edgecolor='r', facecolor=COLOR)
            self.ax.add_patch(rect)
    
    def plot_map(self, cur_state, new_state):
        if self.frame_counter == 0:
            plt.ion()
            if self.n_rows == 5:
                self.fig, self.ax = plt.subplots(figsize=(5,5))
            else:
                self.fig, self.ax = plt.subplots(figsize=(10,10))
            plt.xticks([])
            plt.yticks([])
            c = self.ax.pcolor(self.chessboard, edgecolors='k', linewidths=3, cmap='binary')
            
            for p in range(self.bsize):
              plt.text(self.tp2cl(p)[1]+0.55, self.tp2cl(p)[0]+0.55, str(p), fontsize=12)
            
            plt.text(self.p2c(self.start_state)[1]+0.32, self.p2c(self.start_state)[0]+0.32, 'S', fontsize=28, color='red', fontweight='bold')
            plt.text(self.p2c(self.goal_state)[1]+0.32, self.p2c(self.goal_state)[0]+0.32, 'G', fontsize=28, color='green', fontweight='bold')
            
            for t in range(self.n_blocked_cells):    
                rect_xy = self.wall_cell_coords[t]
                COLOR = self.EVEN_COLOR if t % 2 == 0 else self.ODD_COLOR 
                rect = patches.Rectangle((rect_xy[1]+self.rect_shift_x, rect_xy[0]+self.rect_shift_y), self.rect_w, self.rect_h, linewidth=1, edgecolor='r', facecolor=COLOR)
                self.ax.add_patch(rect)
            
        arrow_start = np.array(self.tp2cl(cur_state)) + 0.5
        arrow_stop = np.array(self.tp2cl(new_state)) + 0.5
        diff = (arrow_stop[0] - arrow_start[0], arrow_stop[1] - arrow_start[1])
        COLOR = (np.random.rand(), np.random.rand(), np.random.rand())
        if not np.array_equal(arrow_start, arrow_stop):
            plt.arrow(arrow_start[1], arrow_start[0], diff[1], diff[0], head_width=0.2, head_length=0.2, fc=COLOR, ec=COLOR)
        
        plt.draw()
        plt.pause(0.3)
        # time.sleep(3)
        # display.display(plt.gcf())
        # display.clear_output(wait=True)
        self.frame_counter += 1
        return self.frame_counter
        