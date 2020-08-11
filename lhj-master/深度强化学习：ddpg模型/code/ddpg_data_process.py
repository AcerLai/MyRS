# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:44:01 2019

@author: gzs13133
"""

import numpy as np

def dict_gen(x):
    
    res = {-1:1}
    k = 2
    for i in set(x):
        res[i] = k
        k += 1
        
    return res

def convert(x):
    
    res = []
    for i in x:
        temp = i.split(':')
        temp[0], temp[1] = int(temp[0]), float(temp[1])
        res.append(temp)
        
    return res

def dict_convert(x, dic):
    
    max_len = 0
    for i in range(len(x)):
        if max_len < len(x[i]):
            max_len = len(x[i])
        for j in range(len(x[i])):
            x[i][j][0] = dic[x[i][j][0]]
    
    return x, max_len
   
def test_hist_gen(data_list, hist_len, user_list):
    
    watch_hist_list = []
    watch_user_list = []
    watch_time_list = []
    next_item = []
    next_item_time = []
    for i in range(len(data_list)):
        temp_item = [j[0] for j in data_list[i]]
        temp_time = [j[1] for j in data_list[i]]
        temp_time_sum = sum(temp_time)
        temp_time = [j/temp_time_sum for j in temp_time]
        temp_user = user_list[i]
        l = len(data_list[i])
        if l > hist_len+1:
            for j in range(l-hist_len-1):
                watch_hist_list.append(temp_item[j:j+hist_len])
                next_item.append(temp_item[j+hist_len])
                watch_user_list.append(temp_user)
                watch_time_list.append(temp_time[j:j+hist_len])
                next_item_time.append(temp_time[j+hist_len])
        else:
            watch_hist_list.append(temp_item[:-2]+[0]*(hist_len+2-l))
            next_item.append(temp_item[-2])
            watch_user_list.append(temp_user)
            watch_time_list.append(temp_time[:-2]+[0]*(hist_len+2-l))
            next_item_time.append(temp_time[-2])
                
    return np.array(watch_hist_list), np.array(next_item), np.array(watch_user_list), \
            np.array(watch_time_list), np.array(next_item_time)
            
def train_hist_gen(data_list, hist_len, user_list, min_hist_len=10, step=5):
    
    watch_hist_list = []
    watch_user_list = []
    watch_time_list = []
    next_item = []
    next_item_time = []
    for i in range(len(data_list)):
        temp_item = [j[0] for j in data_list[i]]
        temp_time = [j[1] for j in data_list[i]]
        temp_time_sum = sum(temp_time)
        temp_time = [j/temp_time_sum for j in temp_time]
        temp_user = user_list[i]
        l = len(data_list[i])
        if l > min_hist_len+1:
            j = 0
#            for j in range(l-min_hist_len-1):
            while j < l-min_hist_len-1:
                watch_hist_list.append(temp_item[:j+min_hist_len]+[0]*(hist_len-j-min_hist_len))
                next_item.append(temp_item[j+min_hist_len])
                watch_user_list.append(temp_user)
                watch_time_list.append(temp_time[:j+min_hist_len]+[0]*(hist_len-j-min_hist_len))
                next_item_time.append(temp_time[j+min_hist_len])
                j += step
        else:
            watch_hist_list.append(temp_item[:-2]+[0]*(hist_len+2-l))
            next_item.append(temp_item[-2])
            watch_user_list.append(temp_user)
            watch_time_list.append(temp_time[:-2]+[0]*(hist_len+2-l))
            next_item_time.append(temp_time[-2])
                
    return np.array(watch_hist_list), np.array(next_item), np.array(watch_user_list), \
            np.array(watch_time_list), np.array(next_item_time)
            
            
