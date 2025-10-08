import numpy as np
import test_env as env
import torch
import torch.nn as nn
import net 

def predict(weight_path):
    predict_net = net.Network(env.n_states, env.n_actions)
    predict_net.load_state_dict(torch.load(weight_path,weights_only=True))
    predict_net.eval()
    print("\n测试最优策略:")
    for i in range(10):
        state = i
        done = False
        path = [state]

        while not done:
            state_tensor = env.encode_state(state, env.n_states)
            with torch.no_grad():
                q_values = predict_net(state_tensor)
                action = torch.argmax(q_values).item()
            next_state, _, done = env.step(state, action)
            path.append(next_state)
            state = next_state

        print(f"最优路径: {' → '.join(map(str, path))}")

if __name__ == '__main__':
    predict("TRYBYME/best1.pt")
