# A Q-learning from scratch in Python for GridWorld

I made this repo when I was teaching assistant of Advance Computational Design Course at ETH Zurich, 2020.

This repository consists of some Python codes for implementing the Q-Learning algorithm from scratch and training the agent to learn a maze.

`env_map_maker.py`:
This code consists of a helper function for implementing mazes with any arbitrary start and goal points and any inner walls.

`mdp_meta_data.py`:
Here, this is a helper function in which we create the metadata, such as the MDP, that we need for defining the Environment class.

`grid_world_general_env.py`
This piece of code defines the Environment class.

`plot_environment.py`:
This helper class consists of some methods to make an image of the maze and visualize the agent moves within the maze.

`agents.py`:
Here, we implemented Random Agent and Q-learning Agent classes.

`train_QAgent.py`:
We can train the Q-learning agent to find the best policy to navigate inside the maze by this code.

`test_QAgent.py`:
This is for testing the trained agent performance.

`visutils.py`:
This includes some helper functions to visualize the agent performance in the training phase.

`training_pipline.py`:
Here, we call the objects we've defined and run the training and test phases.

`RL_QLearningMaze_training_pipline.ipynb`:
This is similar to `training_pipline.py`, but in the Google colab.

`RL_Tutorial_QLearning_for_Maze.ipynb`:
A tutorial for the programming session of the course.