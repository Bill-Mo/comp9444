from torch.distributions import Categorical
import torch
from collections import deque
import gym
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Hyperparameters
batch_size = 1000
lr = 0.001
gamma = 0.999
hid = 80
num_episodes = 2000
avg_steps = 195
continue_reward = 1
fail_reward = -400
epoch = 50
seed = 100
torch.manual_seed(seed)
random.seed(seed)

# Agent class
class Agent():
    def __init__(self, action_dim, device) -> None:
        self.current_step = 0
        self.action_dim = action_dim
        self.device = device

    # Get actions according to current state and model
    def get_action(self, state, policy_net): 
        self.current_step += 1
        pred = policy_net(torch.tensor(state).to(device).unsqueeze(0))
        distribution = Categorical(pred)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

# Deep Q network
class model(torch.nn.Module): 
    def __init__(self, state_dim, action_dim) -> None:
        # Quite simple fully connected network copied from Assignment 1
        super(model, self).__init__()
        self.in_to_hid1 = torch.nn.Linear(state_dim, hid)
        self.hid1_to_hid2 = torch.nn.Linear(hid, hid)
        self.hid2_to_out = torch.nn.Linear(hid, action_dim)

    # Forward mathod
    # Input: Current state
    # Output: Q value for each action
    def forward(self, input):
        hid1_sum = self.in_to_hid1(input)
        self.hid1 = torch.relu(hid1_sum)
        hid2_sum = self.hid1_to_hid2(self.hid1)
        self.hid2 = torch.relu(hid2_sum)
        output_sum = self.hid2_to_out(self.hid2)
        output = F.softmax(output_sum, dim=1)
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = Agent(action_dim, device)
    policy_net = model(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=lr)

    round_batch = []
    total_round = []
    avg_reward = []
    avg_round = []
    num_success = 0
    for episode in range(1, num_episodes + 1):
        state = env.reset(seed=seed)
        round = 0
        success = False
        acc_reward = 0
        c_reward = continue_reward
        done = False
        reward_batch = deque(maxlen=batch_size)
        log_prob_batch = deque(maxlen=batch_size)

        # Try to play game, if done, the loop ends
        while not done:
            action, log_prob = agent.get_action(state, policy_net)
            state, reward, done, _ = env.step(action)
            if done:
                reward = fail_reward
            else:
                reward = c_reward
                c_reward += 0.1
            acc_reward += reward
            reward_batch.append(reward)
            log_prob_batch.append(log_prob)

            round += 1
            if done:
                round_batch.append(round)
                total_round.append(round)
                avg_reward.append(acc_reward)
                break

        # Training
        discount_batch = [gamma ** i for i in range(len(reward_batch))]
        expected_reward = 0
        for index in range(len(reward_batch)):
            expected_reward += discount_batch[index] * reward_batch[index]
            
        loss_batch = [-lp * expected_reward for lp in log_prob_batch]
        loss = torch.cat(loss_batch).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        # For testing purpose
        if episode > 0 and episode % epoch == 0:
            print("episode {}, avg step: {:.2f}, avg reward: {:.2f}".format(episode, np.mean(round_batch), np.mean(avg_reward)))
            avg_round.append(np.mean(round_batch))
            round_batch = []
            avg_reward = []
    env.close()

    for rounds in avg_round:
        if rounds >= avg_steps:
            num_success += 1
    print('Numbers of success: {} Successful rate: {:.2f} Average steps: {}'.format(num_success, (num_success / (num_episodes / epoch)), np.mean(avg_round)))
    plt.plot(range(num_episodes), [avg_steps] * num_episodes)
    plt.plot(range(num_episodes), total_round)
    plt.plot(np.linspace(epoch, num_episodes, num= int(num_episodes / epoch)), avg_round)
    plt.xlabel('Episode')
    plt.ylabel('Step')
    plt.show()