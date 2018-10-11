# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:23:00 2018

@author: v_wangxiangqing
"""

#Download ECG signals

import wfdb
import numpy as np

def Download(patient_name):
    patient_info = []
    for i in patient_name:
        sig_pb, fields_pb = wfdb.rdsamp(str(i), pb_dir = 'mitdb')
        pbannotation = wfdb.rdann(str(i), 'atr', pb_dir='mitdb')
        pbannotation_symbol = pbannotation.symbol
        pbannotation_sample = pbannotation.sample
        patient_info.append([sig_pb, fields_pb, pbannotation_symbol, 
                             pbannotation_sample, pbannotation])
        print("%5.2f"%((patient_name.index(i)+1)/len(patient_name)))
    return patient_info

def Save(patient_name, data, save_path="signals/"):
    for i in range(len(data)):
        if "MLII" in data[i][1]['sig_name']:
            sig = data[i][0][:, data[i][1]['sig_name'].index("MLII")]
            label_index = data[i][3]
            label = np.array(data[i][2])
            np.save(save_path + str(patient_name[i]) + "_sig.npy", sig)
            np.save(save_path + str(patient_name[i]) + "_label_index.npy", label_index)
            np.save(save_path + str(patient_name[i]) + "_label.npy", label)
    print("Already saved")


def Read(patient_name, read_path="signals/"):
    return_list = []
    for i in range(len(patient_name)):
        return_list.append([ np.load(read_path + str(patient_name[i]) + "_label.npy"), 
                             np.load(read_path + str(patient_name[i]) + "_label_index.npy"), 
                             np.load(read_path + str(patient_name[i]) + "_sig.npy")
                           ])
    return return_list

# =============================================================================
# sig_pb, fields_pb = wfdb.rdsamp('100', pb_dir = 'mitdb')
# pbannotation = wfdb.rdann('100', 'atr', pb_dir='mitdb')#, return_label_elements=['label_store', 'symbol'])
# a = pbannotation.symbol
# b = pbannotation.sample
# =============================================================================











