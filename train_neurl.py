import net
import numpy as np
import torch
import test_env as env
import torch.optim as optim
import torch.nn as nn
import predict_net

learning_rate = 0.01     
discount_factor = 0.9   
epsilon = 0.1           
the_best_loss = 100000.0
n_episodes = 100       
now_loss = []
step_loss = []
step_time = 0
net = net.Network(env.n_states, env.n_actions)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()  
net.load_state_dict(torch.load("TRYBYME/best1.pt",weights_only=True))
for episode in range(n_episodes):
    
    state = 0
    done = False
    total_reward = 0
    
    while not done:
        
        state_tensor = env.encode_state(state, env.n_states)
        
        
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(env.n_actions)
        else:
             
            with torch.no_grad():  
                q_values = net(state_tensor)
                action = torch.argmax(q_values).item()
        
        
        next_state, reward, done = env.step(state, action)
        next_state_tensor = env.encode_state(next_state, env.n_states)
        
        
        with torch.no_grad():
            next_q_values = net(next_state_tensor)
            target_q = reward + discount_factor * torch.max(next_q_values)
        
        
        current_q = net(state_tensor)[0, action]
        
        
        loss = loss_fn(current_q, target_q)
        step_loss.append(loss.item())  
        optimizer.zero_grad()  
        loss.backward()       
        optimizer.step()     
        state = next_state
        total_reward += reward


    now_loss.append(np.mean(step_loss))  
    step_time = 0
    step_loss = []
    if (episode + 1) % 20 == 0:
        print(f"回合 {episode + 1}: 总奖励 = {total_reward:.2f}, 损失 = {loss.item():.6f}")
    if now_loss[-1] < the_best_loss:
        the_best_loss = now_loss[-1]
        torch.save(net.state_dict(), "TRYBYME/best1.pt")
        print('%.6f' % the_best_loss)

torch.save(net.state_dict(), "TRYBYME/last1.pt")
print("测试最优策略：")
predict_net.predict("TRYBYME/best1.pt")


