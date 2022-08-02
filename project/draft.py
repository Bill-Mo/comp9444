import random
import gym
import torch
import torch.nn as nn

GAMMA = 0.8

# class agent():
#     def __init__(self, env):
#         self.action_size = env.action_space.n
#         self.dqn = dqn(10)
#         self.explore_rate = 1
#         self.exploit_rate = 0.1

#     def get_action(self, obs):
#         if random.random() < self.explore_rate:

#         pole_angle = obs[2]
#         if pole_angle < 0:
#             action = 0
#         else:
#             action = 1
#         return action

# class dqn(nn.Module):
#     def __init__(self, hid) -> None:
#         super(dqn, self).__init__()
#         self.in_to_hid1 = nn.Linear(4, hid)
#         self.in_to_hid2 = nn.Linear(4, hid)
#         self.in_to_out = nn.Linear(4, 2)
#         self.hid1_to_hid2 = nn.Linear(hid, hid)
#         self.hid1_to_out = nn.Linear(hid, 2)
#         self.hid2_to_out = nn.Linear(hid, 2)

#     def forward(self, state):
#         in_hid1_sum = self.in_to_hid1(state)
#         self.hid1 = torch.tanh(in_hid1_sum)
#         in_hid2_sum = self.in_to_hid2(state)
#         hid1_hid2_sum = self.hid1_to_hid2(self.hid1)
#         hid2_sum = in_hid2_sum + hid1_hid2_sum
#         self.hid2 = torch.tanh(hid2_sum)
#         in_out_sum = self.in_to_out(state)
#         hid1_out_sum = self.hid1_to_out(self.hid1)
#         hid2_out_sum = self.hid2_to_out(self.hid2)
#         output_sum = in_out_sum + hid1_out_sum + hid2_out_sum
#         output = torch.sigmoid(output_sum)
#         return output

#     def q_star(self, state, reward, action):
#         q = self.forward(state)
#         return reward + GAMMA * max(q)
        
# env = gym.make("CartPole-v1")
# env.reset()
# agent = agent(env)
# for i_episode in range(100):
#     obs = env.reset()
#     t = 0
#     done = False
#     while not done:
#         action = agent.get_action(obs)
#         obs,reward,done,info = env.step(action)
#         env.render()
#         t += 1
#         if done:
#             print("Done after {} steps".format(t))
#             break;
# env.close()

t = torch.tensor([[1, 20, 500], [2, 30, 500], [0, 0, 0]])
print(t)
t = torch.tensor([1, 2, 3])
t = torch.unsqueeze(t, 0)
print(t)