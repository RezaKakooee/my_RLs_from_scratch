#%%
import numpy as np

#%%
def index_from_xy(x, y, nX, nY):
    return x+y*nX;

def define_environment_map(maze_dict):
    
    n_rows = maze_dict['n_rows']
    n_cols = maze_dict['n_cols']
    inner_wall_coords = maze_dict['inner_wall_coords'] 
    startRow = maze_dict['startRow']
    startCol = maze_dict['startCol']
    goalRow = maze_dict['goalRow']
    goalCol  = maze_dict['goalCol']
    
    bool_2D_np_array = np.zeros((n_rows, n_cols),dtype=bool)
    for coord in inner_wall_coords:
        bool_2D_np_array[coord[0]][coord[1]] = True
    
    nX, nY = bool_2D_np_array.shape
    inner_wall = []
    up_forbidden = []
    down_forbidden = []
    left_forbidden = []
    right_forbidden = []
    for x in range(nX):
        for y in range(nY):
            if bool_2D_np_array[x,y] == True:
                inner_wall.append(index_from_xy(x, y, nX, nY))
                if x > 0 and bool_2D_np_array[x-1, y] == False:
                    right_forbidden.append(index_from_xy(x-1, y, nX, nY))
                if x < nX-1 and bool_2D_np_array[x+1, y] == False:
                    left_forbidden.append(index_from_xy(x+1, y, nX, nY))
                if y > 0 and bool_2D_np_array[x, y-1] == False:  
                    down_forbidden.append(index_from_xy(x, y-1, nX, nY))
                if y < nY-1 and bool_2D_np_array[x,y+1] == False:   
                    up_forbidden.append(index_from_xy(x, y+1, nX, nY))
    start_state = index_from_xy(startCol, startRow, nX, nY)
    goal_state = index_from_xy(goalCol, goalRow, nX, nY)
    
    env_dict = {'inner_wall':inner_wall,
                'up_forbidden':up_forbidden,
                'down_forbidden':down_forbidden,
                'left_forbidden':left_forbidden,
                'right_forbidden':right_forbidden,
                'start_state':start_state,
                'goal_state': goal_state}
    
    return env_dict

#%%
if __name__ == '__main__':
    ### For the default 5*5 maze
    maze_dict = {'n_rows':5, 'n_cols':5, 
                 'inner_wall_coords':[[1,2],[2,2],[2,3],[2,4]], 
                 'startRow':3, 'startCol':4, 
                 'goalRow':1, 'goalCol':3}
    env_dict = define_environment_map(maze_dict)