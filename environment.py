from data_util import *
from math_model import *
import numpy as np
class Env():
    def __init__(self, data_path, object_num, init_object_factor, epi_len, state_dim):
        self.a = 1
        self.data_path = data_path
        self.object_num = object_num
        self.init_object_factor = init_object_factor
        self.epi_len = epi_len
        self.action_space = [0, 1]
        self.state_space = [0, 1]
        self.action_num = 3
        self.state_num = state_dim
        self.prob = Knap_Prob()
        self.alpha = 0.7

    def cal_reward(self,object_values, actions, obj_val_tilde):
        try:
            # if object_values > obj_val_tilde:
            #     reward = 1
            # elif object_values == obj_val_tilde:
            #     reward = 0
            # else:
            #     reward = -1
            reward = self.alpha * object_values / obj_val_tilde \
                     - (1-self.alpha) * np.linalg.norm(np.array(self.init_object_factor) - np.array(actions)) / math.sqrt(self.object_num)
        except Exception as e:
            print(e)

        return reward

    def step(self, actions, last_actions, call_num):
        next_state, kd, unit_list = init_canvas(self.data_path,self.object_num, call_num, self.epi_len, actions)
        solutions, obj_val = self.prob.reinit(kd, actions)
        solutions_tilde, obj_val_tilde = self.prob.reinit(kd,last_actions)
        
        reward = self.cal_reward(obj_val,actions, obj_val_tilde)
        done = False
        if (call_num+1) % 10 == 0:
            done = True

        return next_state, reward, done, unit_list

    def reset(self, call_num):
        state, kd, unit_list = init_canvas(self.data_path,self.object_num, call_num, self.epi_len,self.init_object_factor)

        return state,unit_list

