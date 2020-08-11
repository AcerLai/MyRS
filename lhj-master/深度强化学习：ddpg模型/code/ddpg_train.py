# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:38:12 2019

@author: gzs13133
"""

import pandas as pd
import numpy as np
#import random
import matplotlib.pylab as plt

import os
os.chdir('C:\\Users\\GZS13133\\intern\\data\\ddpg_data')
from ddpg_data_process import dict_gen, convert, dict_convert, test_hist_gen, train_hist_gen
from ddpg import ccDDPG
import os
os.chdir('C:\\Users\\GZS13133\\intern\\data\\ddpg_data')

data = pd.read_csv('watch_seq_file', sep='\t', header=None)
data.columns = ['user', 'watch_hist']
data['hist_len'] = [len(i.split(',')) for i in data['watch_hist'].values]

#1907
temp = max([len(i.split(',')) for i in data['watch_hist'].values])

data = data[data['hist_len'] > 2]
data_hist_len = data['hist_len'].values
hist_len_percentile = np.percentile(data_hist_len, 90) #52

data = data[data['hist_len'] <= hist_len_percentile]
data = data.iloc[:,:2]

temp_split = [convert(i.split(',')) for i in data['watch_hist'].values]
temp_item = []
for i in temp_split:
    item, _ = zip(*i)
    temp_item.extend(item)

item_dict = dict_gen(temp_item)
item_num = len(item_dict)-1

train_index = np.loadtxt('train_index.txt', dtype=np.int32)
test_index = np.loadtxt('test_index.txt', dtype=np.int32)

data, data_in = data.iloc[train_index,], data.iloc[test_index,]


### train
user_list = [i for i in data['user'].values]

temp_split = [convert(i.split(',')) for i in data['watch_hist'].values]
temp_item = []
for i in temp_split:
    item, _ = zip(*i)
    temp_item.extend(item)

item_multi_list = np.array([[0], [1]] + [[i+2] for i in range(item_num)])
temp_split_hot, max_hist_len = dict_convert(temp_split, item_dict)

#过长的历史进行窗口式截断，由于先前对序列进行了截断，所以不需要窗口式截断
split_hist_len = int(hist_len_percentile - 2)
watch_hist_list, next_item, watch_user_list, watch_time_list, next_item_time = train_hist_gen(temp_split_hot, split_hist_len, user_list)

### test
user_list_in = [i for i in data_in['user'].values]

temp_split_in = [convert(i.split(',')) for i in data_in['watch_hist'].values]
temp_item_in = []
for i in temp_split_in:
    item, _ = zip(*i)
    temp_item_in.extend(item)

temp_split_hot_in, _ = dict_convert(temp_split_in, item_dict)

#过长的历史进行窗口式截断，由于先前对序列进行了截断，所以不需要窗口式截断
watch_hist_list_in, next_item_in, watch_user_list_in, watch_time_list_in, next_item_time_in = test_hist_gen(temp_split_hot_in, split_hist_len, user_list_in)

#########################################################
all_num = len(train_index) + len(test_index)
test_num = 10000
#test_index = random.sample(range(all_num), test_num)
#train_index = list(set(range(all_num)).difference(set(test_index)))

#np.savetxt('train_index.txt', train_index)
#np.savetxt('test_index.txt', test_index)
#train_index = np.loadtxt('train_index.txt', dtype=np.int32)
#test_index = np.loadtxt('test_index.txt', dtype=np.int32)

cc_train_rand = np.random.randn(all_num-test_num)
cc_eval_rand = np.random.randn(test_num)
cc_train_sample = [[watch_hist_list, next_item, watch_time_list, next_item_time], cc_train_rand]
cc_eval_sample = [[watch_hist_list_in, next_item_in, watch_time_list_in, next_item_time_in], cc_eval_rand]


batch_size, epoch, record_num, update_time, parm = 64, 1, 600, 1000, [0.002, 0.004]
add_index =  update_time*batch_size + 2

use_time = False
cc_ddpg = ccDDPG(item_num, split_hist_len, item_multi_list, 1, use_time=use_time) 
temp = []
for i in range(epoch):
    print(i)
    start, end = 0, add_index
    while end < all_num-test_num:
        print('\n')
        print('training in ' + str(end//(add_index)) + ' time -------------')
        index = range(start, end)
        cc_train_sample = [[watch_hist_list[index], next_item[index], watch_time_list[index], next_item_time[index]], cc_train_rand[index]]

        if i == 0 and start == 0:
            ddpg_actor_dict = cc_ddpg.train_actor_with_update_cur(cc_train_sample, cc_eval_sample, batch_size, 1, record_num, opt_parm=parm)
            ddpg_critic_dict = cc_ddpg.train_critic_with_update_cur(cc_train_sample, cc_eval_sample, batch_size, 1, record_num, weight_dict=ddpg_actor_dict, opt_parm=parm)
        else:
            ddpg_actor_dict = cc_ddpg.train_actor_with_update_cur(cc_train_sample, cc_eval_sample, batch_size, 1, record_num, opt_parm=parm)
            ddpg_critic_dict['cur_actor_weight_1'] = ddpg_actor_dict['cur_actor_weight_1']
            ddpg_critic_dict['cur_actor_weight_2'] = ddpg_actor_dict['cur_actor_weight_2']
            ddpg_critic_dict = cc_ddpg.train_critic_with_update_cur(cc_train_sample, cc_eval_sample, batch_size, 1, record_num, weight_dict=ddpg_critic_dict, opt_parm=parm)
        
#        _, _, y_pred_val, _ = cc_ddpg.actor_model.predict(cc_eval_sample[0])
#        temp_acc = np.mean([cc_eval_sample[0][1][i]-2 in y_pred_val[i] for i in range(len(y_pred_val))])
#        temp.append(temp_acc)
        
#        parm[0] *= 0.5
#        parm[1] *= 0.5
        start, end = end, end+add_index
    
    if start < all_num-test_num:
        index = range(start, all_num-test_num)
        cc_train_sample = [[watch_hist_list[index], next_item[index], watch_time_list[index], next_item_time[index]], cc_train_rand[index]]
        ddpg_actor_dict = cc_ddpg.train_actor_with_update_cur(cc_train_sample, cc_eval_sample, batch_size, 1, record_num, weight_dict=ddpg_critic_dict, opt_parm=parm)
        ddpg_critic_dict['cur_actor_weight_1'] = ddpg_actor_dict['cur_actor_weight_1']
        ddpg_critic_dict['cur_actor_weight_2'] = ddpg_actor_dict['cur_actor_weight_2']
        ddpg_critic_dict = cc_ddpg.train_critic_with_update_cur(cc_train_sample, cc_eval_sample, batch_size, 1, record_num, weight_dict=ddpg_critic_dict, opt_parm=parm)
        
#        _, _, y_pred_val, _ = cc_ddpg.actor_model.predict(cc_eval_sample[0])
#        temp_acc = np.mean([cc_eval_sample[0][1][i]-2 in y_pred_val[i] for i in range(len(y_pred_val))])
#        temp.append(temp_acc)
   

np.savetxt('ddpg_acc_3epoch_top20_noweighted.txt', temp)
np.savetxt('ddpg_acc_3epoch_top20_weighted.txt', temp)

import os
os.chdir('C:\\Users\\GZS13133\\intern\\data\\ddpg_data')

name = ['ddpg_acc_3epoch_top20_noweighted', 'ddpg_acc_3epoch_top20_weighted']
temp1 = np.loadtxt('ddpg_acc_3epoch_top20_noweighted.txt')
temp2 = np.loadtxt('ddpg_acc_3epoch_top20_weighted.txt')

x_index = (np.arange(len(temp1))+1) / 14
plt.plot(x_index, temp1, label=name[0])
plt.plot(x_index, temp2, label=name[1])
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.title('The training process')

'''
temp = []
for i in cc_train_sample[0][0]:
    temp += list(i)
    
temp1 = {}
for i in temp:
    if temp1.get(i):
        temp1[i] += 1
    else:
        temp1[i] = 1
        
temp2 = sorted(temp1.items(), key=lambda x:x[1], reverse=True)
tt = cc_ddpg.actor_model.get_layer('item_embedding').get_weights()
tt = tt[0]
tt[temp2[1][0]]
'''

cc_ddpg.actor_model.get_layer('cur_actor_dense_2').set_weights(ddpg_critic_dict['tar_actor_weight_2'])
y_cur_state, y_cur_action, y_pred_val, y_qvalue = cc_ddpg.actor_model.predict(cc_eval_sample[0])
y_tar_state, critic_loss = cc_ddpg.critic_model.predict(cc_eval_sample[0])
np.mean([cc_eval_sample[0][1][i]-2 in y_pred_val[i] for i in range(len(y_pred_val))])

wrong = []
for i in range(len(y_pred_val)):
    if cc_eval_sample[0][1][i]-2 not in y_pred_val[i]:
        wrong.append(i)

def find(x):
    temp = {}
    for i in x:
        if i == 0:
            break
        if temp.get(i):
            temp[i] += 1
        else:
            temp[i] = 1
    
    res = list(temp.items())
    res = sorted(res, key=lambda x:x[1], reverse=True)
    
    return res[0][0]

y_temp = [find(cc_eval_sample[0][0][i]) for i in range(len(y_pred_val))]
np.mean([cc_eval_sample[0][1][i] == y_temp[i] for i in range(len(y_temp))])
# 0.36
#############################################
#item_dict_inv = {}
#for i in item_dict.items():
#    item_dict_inv[i[1]] = i[0]
#    
#with open('res.txt', 'w') as f:
#    count = 0
#    user_temp = []
#    while count < 100:
#        i = 1000 + count
#        temp1 = str(cc_eval_user[i])
#        if temp1 in user_temp:
#            continue
#        temp2 = ','.join([str(item_dict_inv[j+2]) for j in y_pred_val[i]])
#        f.write(temp1 + '::' + temp2 + '\n')
#        count += 1
#        user_temp.append(temp1)
    



