# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:27:04 2018

@author: v_wangxiangqing
"""
import numpy as np
import random
from collections import Counter
#process file
def signals_cha(data):
    return data[1:] - data[:-1]

def load_original_data(patient_name, file_path=r'ECG_signals/'):
    return [patient_name, np.load(file_path+str(patient_name)+r'_signal_.npy'), np.load(file_path+str(patient_name)+r'_R_.npy'), 
            np.load(file_path+str(patient_name)+r'_flag_.npy'), len(np.load(file_path+str(patient_name)+r'_flag_.npy'))]

def seg_original_data(data, signals_len, cha_flag=True):
    patient_name = data[0]
    Signals = []
    for j in range(data[-1]):
        R_index = data[2][j, 0]
        flag = str(data[3][j, 0])
        if (R_index-signals_len[0]>=0) and (R_index+signals_len[1]<len(data[1])):
            signal = data[1][R_index-signals_len[0]:R_index+signals_len[1]]
            if cha_flag:
                Signals.append([patient_name, signals_cha(signal), flag])
            else:
                Signals.append([patient_name, signal, flag])
    return Signals

def build_pool_data(data, model, test_index, train_index):
    if model == "N_others":
        pool_test_N = []
        pool_test_notN = []
        pool_train_N = []
        pool_train_notN = []
        for i in data:
            for j in i:
                if (j[0] in test_index) and (j[-1] == "N"):
                    pool_test_N.append(j)
                if (j[0] in test_index) and (j[-1] != "N"):
                    pool_test_notN.append(j)
                    
                if (j[0] in train_index) and (j[-1] == "N"):
                    pool_train_N.append(j)
                if (j[0] in train_index) and (j[-1] != "N"):
                    pool_train_notN.append(j)
        
        return pool_test_N,pool_test_notN,pool_train_N,pool_train_notN
    
def data_batch(pool_test_N,pool_test_notN,pool_train_N,pool_train_notN,batch_size = 128, style="train"):
    if style == "train":
        pool_N = pool_train_N[:]
        pool_notN = pool_train_notN[:]
    if style == "test":
        pool_N = pool_test_N[:]
        pool_notN = pool_test_notN[:]
    assert batch_size%2 == 0, '在model="Mul_CNN"的情况下batch_size必须是2的整倍数'
    temp_size = int(batch_size/2)
    batch_list = []
    for i in range(temp_size):
        temp_index = random.randint(0,(len(pool_N)-1))
        batch_list.append([pool_N[temp_index][1], [1,0]])
        temp_index = random.randint(0,(len(pool_notN)-1))
        batch_list.append([pool_notN[temp_index][1], [0,1]])
    random.shuffle(batch_list)
    batch_list_x = np.array([n[0] for n in batch_list]).reshape(batch_size, -1, 1)
    batch_list_y = np.array([n[1] for n in batch_list]).reshape(batch_size, 2)
    return batch_list_x,batch_list_y

    
"""统计下错误的样本是分布在哪里的"""
def error_info(data, pool):
    error_result = []
    for i in range(len(data)):
        if data[i] == 0:
            error_result.append(i)
    del i
    return Counter([pool[n][0] for n in error_result]), Counter([pool[n][-1] for n in error_result])


















# =============================================================================
# def fun_label(x):
#     if x in ['N',  'L',  'R',  'e', 'j']:
#         return 'N'
#     if x in ['A',  'a',  'J',  'S']:
#         return 'S'
#     if x in ['V',  'E']:
#         return 'V'
#     if x in ['/',  'f',  'Q']:
#         return 'Q'
#     if x not in ['N',  'L',  'R',  'e', 'j']+['A',  'a',  'J',  'S']+['V',  'E']+['/',  'f',  'Q']:
#         return x
# 
# def fun_label_n(x):
#     if x == 'N':
#         return 'N'
#     else:
#         return 'notN'
# 
# 
# 
# def load_data_test(signals_len, patient_name_list, file_path=r'ECG_signals/', label=['N','V','S','F'], cha_flag=True):
#     patient_information_list = []
#     for i in patient_name_list:
#         patient_information_list.append([i, 
#                                          np.load(file_path+str(i)+r'_signal_.npy'), 
#                                          np.load(file_path+str(i)+r'_R_.npy'), 
#                                          np.load(file_path+str(i)+r'_flag_.npy'), 
#                                          len(np.load(file_path+str(i)+r'_flag_.npy'))
#                                          ])
#     Signals = []
#     for i in patient_information_list:
#         patient_name = i[0]
#         for j in range(i[-1]):
#             R_index = i[2][j, 0]
#             flag = fun_label(str(i[3][j, 0]))
#             if flag in label:
#                 if (R_index-signals_len[0]>=0) and (R_index+signals_len[1]<len(i[1])):
#                     signal = i[1][R_index-signals_len[0]:R_index+signals_len[1]]
#                     if cha_flag:
#                         Signals.append([patient_name, signals_cha(signal), fun_label_n(flag)])
#                     else:
#                         Signals.append([patient_name, signal, fun_label_n(flag)])
#     temp = [[[],[]] for n in patient_name_list]
#     for i in Signals:
#         if i[-1] == 'N':
#             temp[patient_name_list.index(i[0])][0].append(i)
#         if i[-1] == 'notN':
#             temp[patient_name_list.index(i[0])][1].append(i)
#     
#     return temp
# 
# =============================================================================










