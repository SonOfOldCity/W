import gym
import torch
import random
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from resnet import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ReplayBeffer():
    def __init__(self, buffer_maxlen, device):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.device = device

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        try:
            state_list = []
            action_list = []
            reward_list = []
            next_state_list = []
            done_list = []

            batch = random.sample(self.buffer, batch_size)
            for experience in batch:
                s, a, r, n_s, d = experience
                # state, action, reward, next_state, done

                state_list.append(s)
                action_list.append(a)
                reward_list.append(r)
                next_state_list.append(n_s)
                done_list.append(d)
            state_list = np.array(state_list)
            action_list = np.array(action_list)
            reward_list = np.array(reward_list)
            next_state_list = np.array(next_state_list)
            done_list = np.array(done_list)
        except Exception as e:
            print(e)
        return torch.FloatTensor(state_list).to(self.device), \
               torch.FloatTensor(action_list).to(self.device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(self.device), \
               torch.FloatTensor(next_state_list).to(self.device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(self.device)

    def buffer_len(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.resnet = resnet34(state_dim)
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x0 = self.resnet(state)
        x1 = F.relu(self.linear1(x0))
        x2 = F.relu(self.linear2(x1))
        x3 = self.linear3(x2)

        return x3


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.resnet = resnet34(state_dim)
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x0 = self.resnet(state)
        x1 = torch.cat([x0, action], 1)
        x2 = F.relu(self.linear1(x1))
        x3 = F.relu(self.linear2(x2))
        x4 = self.linear3(x3)

        return x4


# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, device, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        self.resnet = resnet34(state_dim)

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x0 = self.resnet(state)
        x1 = F.relu(self.linear1(x0))
        x2 = F.relu(self.linear2(x1))

        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        try:
            state = torch.FloatTensor(state).to(self.device)
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(mean, std)
        except Exception as e:
            print(e)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = normal.log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1,keepdim=True)

        return action, log_prob


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device):

        self.env = env
        self.state_dim = env.state_num
        self.action_dim = env.action_num
        self.device = device

        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.value_net = ValueNet(self.state_dim).to(self.device)
        self.target_value_net = ValueNet(self.state_dim).to(self.device)
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(self.device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim,device=self.device).to(self.device)

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = ReplayBeffer(buffer_maxlen,self.device)

    def get_action(self, state):
        action = self.policy_net.action(state)

        return action

    def update(self, batch_size):
        try:
            state, action, reward, next_state, done = self.buffer.sample(batch_size)
            new_action, log_prob = self.policy_net.evaluate(state)

            # V value loss
            value = self.value_net(state)
            new_q1_value = self.q1_net(state, new_action)
            new_q2_value = self.q2_net(state, new_action)
            next_value = torch.min(new_q1_value, new_q2_value) - log_prob
            value_loss = F.mse_loss(value, next_value.detach())

            # Soft q  loss
            q1_value = self.q1_net(state, action)
            q2_value = self.q2_net(state, action)
            target_value = self.target_value_net(next_state)
            target_q_value = reward + done * self.gamma * target_value
            q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
            q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

            # Policy loss
            policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()

            # Update Policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update v
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Update Soft q
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            q1_value_loss.backward()
            q2_value_loss.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.step()
        except Exception as e:
            print(e)

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


def run(env, agent, Episode, batch_size, init_object_factor):
    Return = []
    action_range = env.action_space
    last_actions = init_object_factor

    try:
        for episode in range(Episode):
            score = 0
            state, unit_list = env.reset(episode+1)
            for i in range(10):
                print('=========%s=========%s============='% (episode,i))
                action = agent.get_action(state)
                print('raw: %s / %s / %s'%(action[0][0],action[0][1],action[0][2]))
                # action output range[-1,1],expand to allowable range
                action_in = abs(np.array(action))
                print('abs: %s / %s / %s'%(action_in[0][0],action_in[0][1],action_in[0][2]))
                for j in range(action_in.shape[1]):
                    action_in[0][j] = action_in[0][j] / unit_list[j+1]
                print('unit: %s / %s / %s'%(action_in[0][0],action_in[0][1],action_in[0][2]))

                try:
                    action_in = [0 if (action_in[0][j]) < 0 else (action_in[0][j] / sum(action_in[0])) for j in range(action_in.shape[1])]
                except Exception as e:
                    print(e)
                print('norm: %s / %s / %s'%(action_in[0],action_in[1],action_in[2]))
                next_state, reward, done, unit_list = env.step(action_in, last_actions, episode*300+i)
                done_mask = 0.0 if done else 1.0
                agent.buffer.push((state[0], action[0], reward, next_state[0], done_mask))
                state = next_state
                last_actions = last_actions

                score += reward
                # if done:
                #     break
                if agent.buffer.buffer_len() >= 12:
                    agent.update(batch_size)

            print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
            Return.append(score)
            score = 0
    except Exception as e:
        print(e)

    Return_List = [[0 for i in range(len(Return))] for j in range(2)]
    for i in range(len(Return)):
        Return_List[1][i] = Return[i]
        Return_List[0][i] = i+1
    Return_Array = np.array(Return_List)
    np.savetxt('return.txt',Return_Array,fmt=['%s']*Return_Array.shape[1],newline = '\n')
    plt.plot(Return_List[0],Return_List[1])
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()
    he = 1


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    buffer_maxlen = 50000

    Episode = 100
    batch_size = 128

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
    #main(env, agent, Episode, batch_size)