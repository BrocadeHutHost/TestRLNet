import numpy as np
# 格子世界：0是起点，4是终点，中间是普通格子
# 状态空间：0, 1, 2, 3, 4（共5个状态）
# 动作空间：0(向左), 1(向右)（共2个动作）
n_states = 5    
n_actions = 2   


Q = np.zeros((n_states, n_actions))


learning_rate = 0.1     
discount_factor = 0.9   
epsilon = 0.1           
n_episodes = 100        


def step(state, action):
    if action == 0:  
        next_state = state - 1
    else:             
        next_state = state + 1
    
    
    next_state = max(0, min(next_state, n_states - 1))
    
    
    if next_state == n_states - 1:  
        reward = 10                
        done = True                
    elif next_state == state:      
        reward = -1                
        done = False               
    else:                          
        reward = -0.1              
        done = False               
    
    return next_state, reward, done