import sys
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
max_epsilon = 1
min_epsilon = 0.005
epsilon_decay = 0.005
batch_size = 50
replay_capacity = 3000
lr = 0.001
gamma = 0.999
hid = 80
num_episodes = 1000
avg_steps = 195
continue_reward = 1
fail_reward = -400
epoch = 50
seed = 100
torch.manual_seed(seed)
random.seed(seed)

# Agent class
class Agent():
    def __init__(self, strategy, action_dim, device) -> None:
        self.current_step = 0
        
        # Current stretagy is epsilon greedy stretagy
        self.strategy = strategy
        self.action_dim = action_dim
        self.device = device

    # Get actions according to current state and DQN
    def get_action(self, state, policy_net): 
        # with torch.no_grad():
        #     print(state)
        #     print(policy_net(torch.tensor(state).to(device)))
        #     return policy_net(torch.tensor(state).to(device)).argmax().item()

        epsilon = self.strategy.get_epsilon(self.current_step)
        self.current_step += 1

        # Exploration
        if random.random() < epsilon:
            # print('explore')
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation
        else: 
            # print('exploit')
            with torch.no_grad():
                # print('output:', policy_net(torch.tensor(state).to(device)))
                # print('action:', policy_net(torch.tensor(state).to(device)).argmax())
                return policy_net(torch.tensor(state).to(device)).argmax().item()

# Experience buffer
class ReplayMemory():
    def __init__(self) -> None:
        self.capacity = replay_capacity
        self.memory = deque()
        self.push_count = 0
    
    # Store experience in memory buffer
    def push(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))
        
        # If the buffer is full, remove first experience
        if len(self.memory) > self.capacity:
            self.memory.popleft()
        self.push_count += 1
    
    # Get a batch of experience at random
    def sample(self):
        return random.sample(self.memory, batch_size)

    # Check is the number of experience in buffer enough for training
    def enough_sample(self):
        return len(self.memory) > batch_size

# Epsilon greedy strategy
class EpsilonGreedyStrategy():
    def __init__(self) -> None:
        self.epsilon = max_epsilon

    # Decrease epsilon each time epsilon is required in order to give more and more chance to exploit
    def get_epsilon(self, current_step):
        self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * current_step)
        return self.epsilon

# Deep Q network
class DQN(torch.nn.Module): 
    def __init__(self, state_dim, action_dim) -> None:
        # Quite simple fully connected network copied from Assignment 1
        super(DQN, self).__init__()
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
        output = output_sum
        # print('------------------')
        # print('output:', output)
        return output

# Functions that give Q values
class Q_values():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get Q value for current states
    @staticmethod
    def get_current(policy_net, current_state_batch, action_batch):
        # print('current Q:', policy_net(current_state_batch))
        # print(action_batch.unsqueeze(-1))
        return policy_net(current_state_batch).gather(dim=1, index=action_batch.unsqueeze(-1)).flatten()

    # Get Q value for next states
    # Known as target Q value and Q star value
    @staticmethod
    def get_next(target_net, next_state_batch, done_batch):
        index = 0
        for done in done_batch:
            
            # If this is end state, Q star value should be 0
            if done: 
                next_state_batch[index] = 0
                # print('done:',done_batch)
                # print('Q_star:', Q_star)
                # raise ValueError
            index += 1

        # print('next Q:', target_net(next_state_batch))
        Q_star = target_net(next_state_batch).max(dim=1)[0]
        # print(Q_star)
        return Q_star


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    strategy = EpsilonGreedyStrategy()
    agent = Agent(strategy, action_dim, device)
    memory = ReplayMemory()

    # policy net is used for training
    # target net is used for calculating Q next value to evaluate Q value gotten from policy net
    policy_net = DQN(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=lr)

    # total reward is for testing purpose
    round_batch = []
    total_round = []
    avg_reward = []
    avg_round = []
    num_success = 0
    for episode in range(1, num_episodes + 1):
        current_state = env.reset(seed=seed)
        round = 0
        success = False
        acc_reward = 0
        done = False
        c_reward = continue_reward
        # Try to play game, if done, the loop ends
        while not done:
            action = agent.get_action(current_state, policy_net)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = fail_reward
            else:
                reward = c_reward
                c_reward += 0.01
            acc_reward += reward
            acc_reward += reward
            memory.push(current_state, action, reward, next_state, done)
            current_state = next_state

            # Training
            if memory.enough_sample():
                minibatch = memory.sample()

                # Extract components of experiences into different tensors
                current_state_batch = torch.tensor(np.array([data[0] for data in minibatch])).to(device)
                action_batch = torch.tensor([data[1] for data in minibatch]).to(device)
                reward_batch = torch.tensor([data[2] for data in minibatch]).to(device)
                next_state_batch = torch.tensor(np.array([data[3] for data in minibatch])).to(device)
                done_batch = torch.tensor([data[4] for data in minibatch]).to(device)

                # print('-----------------------------------------------------------')
                # print('current:', current_state_batch)
                # print('aciton:', action_batch)
                # print('reward:', reward_batch)
                # print('next:', next_state_batch)
                # print('done:', done_batch)
                current_Q_value_batch = Q_values.get_current(policy_net, current_state_batch, action_batch)
                next_Q_value_batch = Q_values.get_next(policy_net, next_state_batch, done_batch)
                Q_star_batch = reward_batch + gamma * next_Q_value_batch
                # print('current state:', current_state_batch)
                # print('current q:', current_Q_value_batch)
                # print('next state:', next_state_batch)
                # print('next q:', Q_star_batch)
                loss = F.mse_loss(current_Q_value_batch, Q_star_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # env.render()
            round += 1
            if done:
                # print('Q batch:', current_Q_value_batch)
                # print('Q star batch:', Q_star_batch)
                # print("Done after {} steps".format(t))
                round_batch.append(round)
                total_round.append(round)
                avg_reward.append(acc_reward)
                break


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