import numpy as np
import test_env as env

env.Q = np.load("TRYBYME/q_learning_policy1.npy")
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

np.save("TRYBYME/q_learning_policy1.npy", env.Q)

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