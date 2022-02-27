import torch
from SAC import *
from environment import *

def main():
    data_path = './dataset/raw'
    object_num = 3
    init_object_factor = [0.3,0.3,0.4]
    epi_len = 2
    state_dim = 10 #状态特征编码的长度
    env = Env(data_path, object_num, init_object_factor, epi_len,state_dim)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    buffer_maxlen = 50000

    Episode = 20
    batch_size = 10
    agent = SAC(env,  gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device)
    run(env, agent, Episode, batch_size, init_object_factor)

main()