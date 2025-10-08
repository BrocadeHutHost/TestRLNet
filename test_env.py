import numpy as np
import torch
n_states = 10    
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

def encode_state(state, n_states):
    
    one_hot = torch.zeros(n_states)
    one_hot[state] = 1.0
    return one_hot.unsqueeze(0)