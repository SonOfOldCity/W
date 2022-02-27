import random as rand
import math
import numpy as np
import torch

class Knap_Data:
    def __init__(self):
        self.knap_cap = 0
        self.items_num = 0
        self.items = []

def gen_data(data_path, object_num, call_num, epi_len):
    items_num = rand.randint(30, 130) #物品数量
    knap_cap = rand.randint(200, 500) #背包容量
    prob_data = [0 for k in range(3 + (object_num + 1) * items_num)]
    prob_data[0] = call_num #调用的次数
    prob_data[1] = items_num #物品数量
    prob_data[2] = knap_cap #背包容量
    kd = Knap_Data()
    kd.knap_cap = knap_cap
    kd.items_num = items_num
    for j in range(items_num):
        prob_data[3 + j * (object_num + 1)] = rand.randint(20, 100)  # 物品体积
        prob_data[3 + j * (object_num + 1) + 1] = rand.randint(1, 100) # 物品价值1
        prob_data[3 + j * (object_num + 1) + 2] = rand.randint(1000, 5000) #物品价值2
        prob_data[3 + j * (object_num + 1) + 3] = rand.randint(400, 1400) # 物品价值3
        kd.items.append([prob_data[3 + j * (object_num + 1)],prob_data[3 + j * (object_num + 1) + 1],prob_data[3 + j * (object_num + 1) + 2],prob_data[3 + j * (object_num + 1) + 3]])

    data_path += '/single-knap_'+str(math.floor(call_num/epi_len))+'_'+str(knap_cap)+'_'+str(items_num)+'.txt'
    np.savetxt(data_path,prob_data,fmt='%.0f')
    unit_list = [100, 100, 5000, 1400] #换单位1的分母，物品体积、物品价值1、物品价值2、物品价值3
    return kd, prob_data, unit_list

def init_canvas(data_path, object_num, call_num, epi_len, last_action):
    kd, single_data, unit_list = gen_data(data_path, object_num, call_num, epi_len)
    can_wid = 160 #画布宽度的最大尺寸
    can_len = 160 #画布长度的最大尺寸
    can_thick = 3 #画布厚度的最大尺寸，同时也对应通道数channel
    #初始化画布
    canvas = [[[-1 for k in range(can_len)] for j in range(can_wid)] for i in range(can_thick)]
    #绘制画布第一层的图形，对应的是item的信息
    item_info_size = len(single_data) - 3 #前三个是被调用的次数，物品数量和背包容量
    col_index = 0
    row_index = 0
    for i in range(item_info_size):
        canvas[0][row_index][col_index] = single_data[3 + i] / (unit_list[int(i%4)] * 100)
        col_index += 1
        if (i + 1) % can_wid == 0:
            row_index += 1
            col_index = 0
    #绘制画布第二层的图形，对应的是knap的信息
    canvas[1][int(can_wid/2)][int(can_len/2)] = single_data[2]
    for i in range(len(last_action)):
        canvas[2][int(can_wid/2)][int(can_len/2 - len(last_action) / 2 + i)] = last_action[i]

    canvas_array = np.array(canvas)
    canvas_array = np.expand_dims(canvas_array,axis=0)

    #canvas_tensor = torch.from_numpy(canvas_array)

    return canvas_array, kd, unit_list
