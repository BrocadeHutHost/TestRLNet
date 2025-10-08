import numpy as np
import test_env as env


for episode in range(env.n_episodes):
    state = 0
    done = False
    total_reward = 0
    
    while not done:

        if np.random.uniform(0, 1) < env.epsilon:
            action = np.random.choice(env.n_actions)

        else:
            action = np.argmax(env.Q[state, :])

        next_state, reward, done = env.step(state, action)
        
        env.Q[state, action] = env.Q[state, action] + env.learning_rate * (reward + env.discount_factor * np.max(env.Q[next_state, :]) - env.Q[state, action])
        
        state = next_state
        total_reward += reward
    
    if (episode + 1) % 10 == 0:
        print(f"回合 {episode + 1}: 总奖励 = {total_reward:.2f}")

print("\n训练完成后的Q表：")
print("状态\\动作 | 向左(0) | 向右(1)")
print("-" * 30)
for i in range(env.n_states):
    print(f"    {i}    | {env.Q[i,0]:.2f}   | {env.Q[i,1]:.2f}")

print("简单测试：")
state = 0
done = False
path = [state]

while not done:
    action = np.argmax(env.Q[state, :])
    next_state, _, done = env.step(state, action)
    path.append(next_state)
    state = next_state

print(f"最优路径: {' → '.join(map(str, path))}")