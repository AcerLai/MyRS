# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:34:08 2019

@author: gzs13133
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pylab as plt

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import os 
os.chdir('C://Users//GZS13133//intern//data//data_with_hist')
import keras.backend as K
from base_structure import draw
from model import ccBaseModel, ccDINModelwithOneHist, ccdeepFM
from data_process import judge_multi_feature, multi_column_info, one_column_inde_info, train_test_split, \
                    sample_weight_generate, nlp_weight_dict, nlp_wor2vec_dict, gametype_word2evc_list, \
                    train_test_split_all, column_count_bar, data_build


'''数据类型
sn, gametype, label, 
android_app: 1001 types, 250 length, sn
ios_app: 1001 types, 8 length, sn
mobile_model: 1001 types, 20 length, sn 
interest_gametype: : 24 types, 9 length, sn
gametype_label: 1000 types, 20 length, game
gametype_hotrank_by_aid_cnt: 51 types, 1 length, game 
gametype_hotrank_by_uid_cnt: 51 types, 1 length, game
gametype_id: 127 types, 1 length, same as gametype
'''

'''建模的两种思路
1.将有长度的seqence看成是multi-hot编码的特征输入到DIN模型
2.将有长度的seqence看成是多个hist序列进行attention求和处理 （不可行，因为特征空间不同，可以引入transformer结构）
(理解成其他不同的历史行为对用户的推荐产生不一样的影响)
本质上两者的区别在于是否有attention
'''


############    数据读取
columns = 'sn, gametype, label, android_app, ios_app, mobile_model, \
interest_gametype, gametype_label, gametype_hotrank_by_aid_cnt, \
gametype_hotrank_by_uid_cnt, gametype_id'

#data_route = 'temp'
#data_route = 'temp_05_09'
data_route = 'temp_05_22_train'
data_all = pd.read_csv(data_route, header=None, sep='\t', low_memory=False)
data_all.columns = columns.split(', ')

#gametype & gametype_id 一致
#a, b = list(data['gametype'].values),list(data['gametype_id'].values)
#sum([1*(a[i]==b[i]) for i in range(len(a))])
data_all = data_all.iloc[:,:-1]
    
data_all = data_all.fillna('-1')

data_route = 'temp_05_22_test'
data_all_1 = pd.read_csv(data_route, header=None, sep='\t', low_memory=False)
data_all_1.columns = columns.split(', ')
data_all_1 = data_all_1.iloc[:,:-1]
data_all_1 = data_all_1.fillna('-1')

data_alll = pd.concat([data_all, data_all_1])
data, data_in = train_test_split_all(data_alll, test_num=150000)
del data_alll, data_all, data_all_1
#data_all_with_hist = data_all[data_all['interest_gametype'] != '-1']
#data_all_without_hist = data_all[data_all['interest_gametype'] == '-1']

#data_all = pd.read_csv('C:/Users/GZS13133/intern/data/csv_data/data_train_all.csv', low_memory=False)
#data_in = pd.read_csv('C:/Users/GZS13133/intern/data/csv_data/data_test_all.csv', low_memory=False)

data_all = pd.read_csv('temp_hist_0610', low_memory=False)

data_all.columns = columns.split(', ')
data_all = data_all.iloc[:,:-1] 
data_all = data_all.fillna('-1')
data_all = data_all[data_all['gametype'] != -1]
data_all_hist = data_all[data_all['interest_gametype'] != '-1']
data_all_hist.to_csv('temp_hist_0610', header=True, index=False)
#train_num = 5000000
#data_all = data_build(data_all, train_num, app_minus_inclued=False, sep='::')
column_count_bar(data_all_hist, 'gametype', 'All')

data = pd.read_csv('temp_train_0610_100w', low_memory=False)
data_in = pd.read_csv('temp_test_0610_1w', low_memory=False)
data_all = pd.concat([data, data_in])

data_temp = data_build(data_all, 800000)
data, data_in = train_test_split_all(data_temp, test_num=150000)
#data.to_csv('temp_train_0627_400w', header=True, index=False)
#data_in.to_csv('temp_test_0627_15w', header=True, index=False)
del data_all, data_temp


#不同gametype下的1-0比例不一致，需要先进行分析
def fun1(x):
    return sum(x==1)/sum(x==0)
def fun2(x):
    return sum(x==1)
def fun3(x):
    return sum(x==0)
def fun4(x):
    return sum(x==1)/len(x)
def fun5(x, total=len(data)):
    return len(x)/total

gametype_ratio = data.groupby('gametype').agg({'label':[fun1, fun2, fun3, fun4, fun5]})
gametype_ratio.columns = ['label_1/0_rate', 'label_1_num', 'label_0_num', 'label_1_ratio', 'gametype_ratio']

#gametype:[pos_ratio, gametype_ratio]
gametype_rate_dict = {}
for i in zip(gametype_ratio.index, gametype_ratio.iloc[:,3].values, gametype_ratio.iloc[:,4].values):
    gametype_rate_dict[i[0]] = [i[1], i[2]]

sample_nums, train_prop = len(data), 0.97
#data = data_all
#del data_all
train_num = int(sample_nums)
#data = data_build(data_all, sample_nums, gametype_rate_dict)
#data = data_build(data_all_with_hist, sample_nums, app_minus_inclued=False)
#613229个sn,即用户;48个gametype
#len(set(list(data['sn'].values))) 
#temp_0 = data.groupby('sn').agg(len)


############    数据预处理
def entropy(label):
    
    temp_1 = np.sum([label == 1]) / len(label)
    temp_2 = 1 - temp_1
    entropy = -(temp_1*np.log(temp_1) + temp_2*np.log(temp_2))
    
    return entropy

#max:0.6931471805599453
entropy(np.array(data['label'].values))  # 0.6197236875877505
temp =data.groupby('gametype_hotrank_by_uid_cnt').agg({'label':entropy})
np.sum(temp['label'].values)/len(temp) # 0.40144343228980817
temp =data.groupby('gametype_hotrank_by_aid_cnt').agg({'label':entropy})
np.sum(temp['label'].values)/len(temp) # 0.4163850320997233

gametype_num = len(set(data['gametype']))
#发现有个含有-1，可以填充；有个9022的，出现了一个子label，即变短了，也可以填充。
gametype_label_sp = judge_multi_feature(data, 'gametype', 'gametype_label')

######    game的feature_list生成  
#生成对应的单长度序列(one:gametype_hotrank_by_aid_cnt & gametype_hotrank_by_uid_cnt),作为两个独立的特征
hotrank_aid_num, hotrank_aid_list, hotrank_aid_dict = one_column_inde_info(data, 'gametype_hotrank_by_aid_cnt')
hotrank_uid_num, hotrank_uid_list, hotrank_uid_dict = one_column_inde_info(data, 'gametype_hotrank_by_uid_cnt')

#或者多长度序列(multi:gametype_label)
gametype_label_num, gametype_label_len, gametype_dict, gametype_label_list, gametype_label_dict, _ = multi_column_info(data, 'gametype_label', 'gametype', sep='::')

#nlp特征
r1 = 'p2'
r2 = 'dout2'

gametype_nlp_weight_dict = nlp_weight_dict(r1, gametype_dict)
word2vec = nlp_wor2vec_dict(r2)

word2vec_list = gametype_word2evc_list(gametype_nlp_weight_dict, word2vec, gametype_num+1)
del word2vec

######    sn的feature_list生成  
sn_num = len(set(data['sn']))
#一致性检查，发现很多sn下对应multi特征并不一样，但是只需要找到其中最长的作为其特征表示即可
sn_android_app_sp = judge_multi_feature(data, 'sn', 'android_app')
sn_ios_app_sp = judge_multi_feature(data, 'sn', 'ios_app')
sn_mobile_model_sp = judge_multi_feature(data, 'sn', 'mobile_model')
sn_interest_gametype_sp = judge_multi_feature(data, 'sn', 'interest_gametype')

android_app_num, android_app_len, sn_dict, android_app_list, android_app_dict, _ = multi_column_info(data, 'android_app', 'sn', sep='::')
ios_app_num, ios_app_len, _, ios_app_list, ios_app_dict, _ = multi_column_info(data, 'ios_app', 'sn', sep='::')
mobile_model_num, mobile_model_len, _, mobile_model_list, mobile_model_dict, _ = multi_column_info(data, 'mobile_model', 'sn', sep='::')
interest_gametype_num, interest_gametype_len, _, interest_gametype_list, interest_gametype_dict, tr_list = multi_column_info(data, 'interest_gametype', 'sn', gametype_dict=gametype_dict, sep='::')


######    model输入生成
###    train准备
sn_list = list(data['sn'].apply(lambda x:sn_dict[x]).values)
gametype_list = list(data['gametype'].apply(lambda x:gametype_dict[x]).values)
#以interest作为hist(其他也转化hist备用)
hist_list = [interest_gametype_list[i] for i in sn_list]
hist_last_sequence_list = [tr_list[0][i] for i in sn_list]
pos = [tr_list[1][i] for i in sn_list]
neg = [tr_list[2][i] for i in sn_list]
android_hist_list = [android_app_list[i] for i in sn_list]
ios_hist_list = [ios_app_list[i] for i in sn_list]
mobile_hist_list = [mobile_model_list[i] for i in sn_list]

label = np.array(list(data['label'].values))
#根据gametype下的gametype比例以及0-1比例设置sample_weight
sample_weight_gametype_and_label = sample_weight_generate(gametype_rate_dict)

sample_weight = []
for i in range(len(data)):
    temp = str(data['gametype'].iloc[i]) + str(data['label'].iloc[i])
    sample_weight.append(sample_weight_gametype_and_label[temp]) 
sample_weight = np.array(sample_weight)
 
#构造训练集和测试集
train_sample_index, test_sample_index = train_test_split(data, train_num, gametype_rate_dict)
sample_weight_train = sample_weight[train_sample_index]

# one_hist
#同分布
x_all = list(zip(sn_list, gametype_list, hotrank_aid_list, hotrank_uid_list, hist_list))
x_all = np.array(x_all)
x_train = x_all[train_sample_index].tolist()
x_train = [list(i) for i in list(zip(*x_train))]
x_eval = x_all[test_sample_index].tolist()
x_eval = [list(i) for i in list(zip(*x_eval))]
cc_train_sample = [x_train, label[train_sample_index]]
cc_eval_sample = [x_eval, label[test_sample_index]]

#尽量均衡
eval_bal_nums = 90000

#data_all_1_with_hist = data_all_1[data_all_1['interest_gametype'] != '-1']
#data_all_1_without_hist = data_all_1[data_all_1['interest_gametype'] == '-1']
#diff_list = list(set(data_all_1['sn']).difference(set(data['sn'])))  
#in_list = list(set(sn_list))
#in_index = [i for i in range(len(data_all_1)) if data_all_1['sn'].iloc[i] in in_list]  
#
#temp_ratio = 1/len(gametype_rate_dict)
#gametype_rate_dict_balance = {}
#for i in gametype_rate_dict.keys():
#    gametype_rate_dict_balance[i] = [0.5, temp_ratio]
#    
#data_in = data_build(data_all_1_with_hist, eval_bal_nums, gametype_rate_dict_balance)
#del data_all_1, data_all_1_with_hist, data_all_1_without_hist

_, hotrank_aid_list_in, _ = one_column_inde_info(data_in, 'gametype_hotrank_by_aid_cnt', base_dict=hotrank_aid_dict)
_, hotrank_uid_list_in, _ = one_column_inde_info(data_in, 'gametype_hotrank_by_uid_cnt', base_dict=hotrank_uid_dict)
_, _, _, gametype_label_list_in, _, _ = multi_column_info(data_in, 'gametype_label', 'gametype', maxlen=gametype_label_len, dict_base=gametype_label_dict)

_, _, sn_dict_in, android_app_list_in,_, _ = multi_column_info(data_in, 'android_app', 'sn', maxlen=android_app_len, dict_base=android_app_dict, sep='::')
_, _, _, ios_app_list_in, _, _ = multi_column_info(data_in, 'ios_app', 'sn', maxlen=ios_app_len, dict_base=ios_app_dict, sep='::')
_, _, _, mobile_model_list_in, _, _ = multi_column_info(data_in, 'mobile_model', 'sn', maxlen=mobile_model_len, dict_base=mobile_model_dict, sep='::')
_, _, _, interest_gametype_list_in, _, tr_list_in = multi_column_info(data_in, 'interest_gametype', 'sn', maxlen=interest_gametype_len, gametype_dict=gametype_dict, sep='::')

sn_list_in = list(data_in['sn'].apply(lambda x:sn_dict[x] if x in sn_dict.keys() else 1).values)
gametype_list_in = list(data_in['gametype'].apply(lambda x:gametype_dict[x] if x in gametype_dict.keys() else 1).values)
hist_list_in = [interest_gametype_list_in[sn_dict_in[i]] for i in data_in['sn']]
hist_last_sequence_list_in = [tr_list_in[0][sn_dict_in[i]] for i in data_in['sn']]
pos_in = [tr_list_in[1][sn_dict_in[i]] for i in data_in['sn']]
neg_in = [tr_list_in[2][sn_dict_in[i]] for i in data_in['sn']]

label_eval = np.array(list(data_in['label'].values))
label_rand = np.zeros(len(label))
label_eval_rand = np.zeros(len(label_eval))

#length = interest_gametype_len*interest_gametype_len
#auxiliary_index_ltr = np.array([np.tril(np.ones(interest_gametype_len, dtype='int32')) for i in range(len(sn_list))]).reshape((-1, length))
#auxiliary_index_i = np.array([np.eye(interest_gametype_len, dtype='int32') for i in range(len(sn_list))]).reshape((-1, length))
#auxiliary_index_ltr_in = np.array([np.tril(np.ones(interest_gametype_len, dtype='int32')) for i in range(len(sn_list_in))]).reshape((-1, length))
#auxiliary_index_i_in = np.array([np.eye(interest_gametype_len, dtype='int32') for i in range(len(sn_list_in))]).reshape((-1, length))


x_train = [sn_list, gametype_list, hotrank_aid_list, hotrank_uid_list, hist_list]
cc_train_sample = [x_train, label, label_rand]
x_eval = [sn_list_in, gametype_list_in, hotrank_aid_list_in, hotrank_uid_list_in, hist_list_in]
cc_eval_sample = [x_eval, label_eval, label_eval_rand]

x_train_tr = [sn_list, gametype_list, hotrank_aid_list, hotrank_uid_list, hist_list, hist_last_sequence_list, pos, neg]
cc_train_sample_tr = [x_train_tr, label, label_rand]
x_eval_tr = [sn_list_in, gametype_list_in, hotrank_aid_list_in, hotrank_uid_list_in, hist_list_in, hist_last_sequence_list_in, pos_in, neg_in]
cc_eval_sample_tr = [x_eval_tr, label_eval, label_eval_rand]

# multi_hist
#同分布
x_all_multi = list(zip(sn_list, gametype_list, hotrank_aid_list, hotrank_uid_list, hist_list, android_hist_list, ios_hist_list, mobile_hist_list))
x_all_multi = np.array(x_all_multi)
x_train_multi = x_all_multi[train_sample_index].tolist()
x_train_multi = [list(i) for i in list(zip(*x_train_multi))]
x_eval_multi = x_all_multi[test_sample_index].tolist()
x_eval_multi = [list(i) for i in list(zip(*x_eval_multi))]
cc_train_multi_sample = [x_train_multi, label[train_sample_index]]
cc_eval_multi_sample = [x_eval_multi, label[test_sample_index]]

#均衡
x_train_multi = [sn_list, gametype_list, hotrank_aid_list, hotrank_uid_list, hist_list, android_hist_list, ios_hist_list, mobile_hist_list]
cc_train_multi_sample = [x_train_multi, label]

android_hist_list_in = [android_app_list_in[sn_dict_in[i]] for i in data_in['sn']]
ios_hist_list_in = [ios_app_list_in[sn_dict_in[i]] for i in data_in['sn']]
mobile_hist_list_in = [mobile_model_list_in[sn_dict_in[i]] for i in data_in['sn']]

x_eval_multi = [sn_list_in, gametype_list_in, hotrank_aid_list_in, hotrank_uid_list_in, hist_list_in, android_hist_list_in, ios_hist_list_in, mobile_hist_list_in]
cc_eval_multi_sample = [x_eval_multi, label_eval]


###    model参数
#简单规则：train的分布区预测test
def fun1(x):
    return sum(x==1)/len(x)

data_dict = {}
temp = data.groupby('gametype').agg({'label':fun1})
temp.columns = ['label_1_ratio']
for i in zip(temp.index, temp['label_1_ratio']):
    data_dict[i[0]] = i[1]

y_pred = []
for i in data_in['gametype']:
    if i in data_dict.keys():
        y_pred.append(data_dict[i])
    else:
        y_pred.append(0)
    
roc_auc_score(data_in['label'], y_pred) #0.5779539111727456
#column_count_bar(data_in, 'gametype', t='test')
#column_count_bar(data, 'gametype', t='train')
'''
######    绘制对比效果图

i = 0
for use_Activa in use_Activas:
#    draw_epoch(cc_DIN_One[i].auc_val, 'One'+use_Activa, batch_size, train_num, record_num, i)
#    draw_epoch(cc_DIN_Multi[i].auc_val, 'Multi'+use_Activa, batch_size, train_num, record_num, i)
    draw(cc_DIN_One[i].auc_val, cc_DIN_Multi[i].auc_val, train_num, batch_size, record_num, use_Activa, i+1)   
    i += 1

i = 0    
One_Dataframe = []
Multi_Dataframe = []
for use_Activa in use_Activas:
    One_dict = {'epoch_1': cc_DIN_One[i].auc_val_item[0], 'epoch_2': cc_DIN_One[i].auc_val_item[1], 'epoch_3': cc_DIN_One[i].auc_val_item[2]}
    temp_One = pd.DataFrame(One_dict)
    temp_One.index = cc_DIN_One[i].item_val  
    One_Dataframe.append(temp_One)
    
    Multi_dict = {'epoch_1': cc_DIN_Multi[i].auc_val_item[0], 'epoch_2': cc_DIN_Multi[i].auc_val_item[1], 'epoch_3': cc_DIN_Multi[i].auc_val_item[2]}
    temp_Multi = pd.DataFrame(Multi_dict)
    temp_Multi.index = cc_DIN_Multi[i].item_val  
    Multi_Dataframe.append(temp_Multi)
    
    i += 1

  
############   数据过多，因此分块进行训练
#block_size = 10
#block_num = sample_nums // 10
#cc_DIN_One_s = []
#
#for block in range(1, block_size+1):
#    cc_train_sample_temp = [[i[block_num*(block-1):block_num*block] for i in cc_train_sample[0]], cc_train_sample[1][block_num*(block-1):block_num*block]]
#
#    if block == 1:
#        ccDIN_s = ccDINModelwithOneHist(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
#                           android_app_num, ios_app_num, mobile_model_num, 
#                           android_app_list, ios_app_list, mobile_model_list,
#                           gametype_label_num, gametype_label_list, use_Activa='Sigmoid')
#        
#        ccDIN_res_s = ccDIN_s.train_model(cc_train_sample_temp, cc_eval_sample, batch_size, epoch, record_num)
#        cc_DIN_One_s.append(ccDIN_res_s)
#    else:
#        ccDIN_res_s = ccDIN_s.train_model(cc_train_sample_temp, cc_eval_sample, batch_size, epoch, record_num)
#        cc_DIN_One_s.append(ccDIN_res_s)

#val_all = []
#for res in cc_DIN_One_s:
#    val_all.extend(res.auc_val )
#    
#import matplotlib.pylab as plt
#plt.plot(list(range(len(val_all))), val_all)
#plt.show()
'''

#gdbt + LR
def generate_array(raw_list, l):
    
    res = [0] * (l+2)
    for i in raw_list:
        if i == 0:
            break
        res[i] = 1
   
    return res
     
hotrank_aid_list_array = np.array(hotrank_aid_list, dtype='int32').reshape(-1, 1)
hotrank_uid_list_array = np.array(hotrank_uid_list, dtype='int32').reshape(-1, 1)
gt_array = np.array(gametype_list, dtype='int32').reshape(-1, 1)
gtl_array = np.array([generate_array(gametype_label_list[i], gametype_label_num) for i in gametype_list], dtype='int32').reshape(-1, gametype_label_num+2)
hist_array = np.array([generate_array(i, gametype_num) for i in hist_list], dtype='int32').reshape(-1, gametype_num+2)
android_hist_array = np.array([generate_array(i, android_app_num) for i in android_hist_list], dtype='int32').reshape(-1, android_app_num+2)
ios_hist_array = np.array([generate_array(i, ios_app_num) for i in ios_hist_list], dtype='int32').reshape(-1, ios_app_num+2)
mobile_hist_array = np.array([generate_array(i, mobile_model_num) for i in mobile_hist_list], dtype='int32').reshape(-1, mobile_model_num+2)
gl_train = np.concatenate((hotrank_aid_list_array, hotrank_uid_list_array, gt_array, gtl_array, hist_array, android_hist_array, ios_hist_array, mobile_hist_array), axis=-1)

hotrank_aid_list_array_eval = np.array(hotrank_aid_list_in, dtype='int32').reshape(-1, 1)
hotrank_uid_list_array_eval = np.array(hotrank_uid_list_in, dtype='int32').reshape(-1, 1)
gt_array_eval = np.array(gametype_list_in, dtype='int32').reshape(-1, 1)
gtl_array_eval = np.array([generate_array(gametype_label_list_in[i], gametype_label_num) for i in gametype_list_in], dtype='int32').reshape(-1, gametype_label_num+2)
hist_array_eval = np.array([generate_array(i, gametype_num) for i in hist_list_in], dtype='int32').reshape(-1, gametype_num+2)
android_hist_list_in = [android_app_list_in[sn_dict_in[i]] for i in data_in['sn']]
android_hist_array_eval = np.array([generate_array(i, android_app_num) for i in android_hist_list_in], dtype='int32').reshape(-1, android_app_num+2)
ios_hist_list_in = [ios_app_list_in[sn_dict_in[i]] for i in data_in['sn']]
ios_hist_array_eval = np.array([generate_array(i, ios_app_num) for i in ios_hist_list_in], dtype='int32').reshape(-1, ios_app_num+2)
mobile_hist_list_in = [mobile_model_list_in[sn_dict_in[i]] for i in data_in['sn']]
mobile_hist_array_eval = np.array([generate_array(i, mobile_model_num) for i in mobile_hist_list_in], dtype='int32').reshape(-1, mobile_model_num+2)
gl_eval = np.concatenate((hotrank_aid_list_array_eval, hotrank_uid_list_array_eval, gt_array_eval, gtl_array_eval, hist_array_eval, android_hist_array_eval, ios_hist_array_eval, mobile_hist_array_eval), axis=-1)

name = [str(i) for i in range(gl_train.shape[1])]

lgb_train = lgb.Dataset(gl_train, label)
lgb_eval = lgb.Dataset(gl_eval, label_eval, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 6,
    'num_trees': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# number of leaves,will be used in feature transformation
num_leaf = 6

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train,
                feature_name=name,
                categorical_feature=name)

#print('Save model...')
## save model to file
#gbm.save_model('model.txt')

print('Start predicting...')
# predict and get data on leaves, training data
y_pred = gbm.predict(gl_train, pred_leaf=True)

print(np.array(y_pred).shape)
print(y_pred[:10])

print('Writing transformed training data')
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], \
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1


y_pred = gbm.predict(gl_eval, pred_leaf=True)
print('Writing transformed testing data')
transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_testing_matrix[i][temp] += 1


lm = LogisticRegression(penalty='l2', C=0.05) # logestic model construction
lm.fit(transformed_training_matrix, label)  # fitting the data
y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label

eval_auc = roc_auc_score(label_eval, y_pred_test)
print(eval_auc)

temp, batch_size, epoch, record_num, save_path = [], 64, 3, 1000, 'D:\\my\\netease_data\\model'
os.chdir(save_path)

ccbase = ccBaseModel(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len)
ccbase_res = ccbase.train_model(cc_train_sample, cc_eval_sample, batch_size, epoch, record_num)
ccbase.model.save('base_model.h5')

ccbase_tr = ccBaseModel(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len,
                   weighted=True, use_Transformer=True)
ccbase_tr_res = ccbase_tr.train_model(cc_train_sample_tr, cc_eval_sample_tr, batch_size, epoch, record_num)
ccbase_tr.model.save('basetr_model.h5')

ccdin = ccDINModelwithOneHist(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len)
ccdin_res = ccdin.train_model(cc_train_sample, cc_eval_sample, batch_size, epoch, record_num)
ccdin.model.save('din_model.h5')

ccdin_tr = ccDINModelwithOneHist(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len,
                   weighted=True, use_Transformer=True)
ccdin_tr_res = ccdin_tr.train_model(cc_train_sample_tr, cc_eval_sample_tr, batch_size, epoch, record_num)
ccdin_tr.model.save('dintr_model.h5')

ccpnn = ccPNN(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len)
ccpnn_res = ccpnn.train_model(cc_train_sample, cc_eval_sample, batch_size, epoch, record_num)
ccpnn.model.save('pnn_model.h5')

ccpnn_tr = ccPNN(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len, weighted=True, use_Transformer=True, w=1e-12)
ccpnn_tr_res = ccpnn_tr.train_model(cc_train_sample_tr, cc_eval_sample_tr, batch_size, epoch, record_num)
ccpnn_tr.model.save('pnntr_model.h5')

ccdeepfm = ccdeepFM(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len)
ccdeepfm_res = ccdeepfm.train_model(cc_train_sample, cc_eval_sample, batch_size, epoch, record_num)
ccdeepfm.model.save('deepfm_model.h5')

ccdeepfm_tr = ccdeepFM(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                   android_app_num, ios_app_num, mobile_model_num, 
                   android_app_list, ios_app_list, mobile_model_list,
                   gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                   user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len,
                   weighted=True, use_Transformer=True)
ccdeepfm_tr_res = ccdeepfm_tr.train_model(cc_train_sample_tr, cc_eval_sample_tr, batch_size, epoch, record_num)
ccdeepfm_tr.model.save('deepfmtr_model.h5')

start, end, model = 1, 6, 'base'
for i in range(start, end):
    print('#'*50)
    print(i)
    print('#'*50)

    K.clear_session()
    ccbase = ccBaseModel(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                       android_app_num, ios_app_num, mobile_model_num, 
                       android_app_list, ios_app_list, mobile_model_list,
                       gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                       user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len)
    ccbase_res = ccbase.train_model(cc_train_sample, cc_eval_sample, batch_size, epoch, record_num)
#    np.savetxt(model + '_auc_' + str(i) + '.txt', ccbase_res.auc_val)
    
    K.clear_session()
    ccbase_tr = ccBaseModel(sn_num, gametype_num, hotrank_aid_num, hotrank_uid_num, interest_gametype_len, 
                       android_app_num, ios_app_num, mobile_model_num, 
                       android_app_list, ios_app_list, mobile_model_list,
                       gametype_label_num, gametype_label_list, user_multi_cate_len_1=android_app_len, user_multi_cate_len_2=ios_app_len, 
                       user_multi_cate_len_3=mobile_model_len, item_multi_cate_len=gametype_label_len,
                       weighted=True, use_Transformer=True)
    ccbase_tr_res = ccbase_tr.train_model(cc_train_sample_tr, cc_eval_sample_tr, batch_size, epoch, record_num)
#    np.savetxt(model + 'tr_auc_' + str(i) + '.txt', ccbase_tr_res.auc_val)


draw(temp, train_num, batch_size, record_num, 'ReLU', 1, ['deepfm', 'base', 'base_tr', 'din', 'din_tr', 'pnn']) 


temp1_mean, temp2_mean = 0, 0
for i in range(start, end):
    temp1 = np.loadtxt(model + '_auc_' + str(i) + '.txt')
    temp2 = np.loadtxt(model + 'tr_auc_' + str(i) + '.txt')
    temp1_mean = temp1_mean + temp1
    temp2_mean = temp2_mean + temp2

temp1_mean = temp1_mean / (end - start)
temp2_mean = temp2_mean / (end - start)

x_index = (np.arange(len(temp1_mean))+1) / (train_num // batch_size // record_num)
plt.plot(x_index, temp1_mean, label=model)
plt.plot(x_index, temp2_mean, label=model+'_tr')
plt.xlabel('epochs')
plt.ylabel('auc')
plt.legend()
plt.title('The training process')





