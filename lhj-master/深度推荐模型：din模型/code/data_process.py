# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:47:51 2019

@author: gzs13133
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pylab as plt


#判断multi类型的特征再同一type下是否一致，并确定是否进行处理：如gametype下的label
def judge_multi_feature(data, feature, feature_label, sep=','):
        
    def judge(x):
        
        temp = list(x.values)
        temp = [j.split(sep) for j in temp if j != '-1']
                
        if not temp:
            return 1
        
        temp_set = [set(temp[0])]
        for j in temp:
            if set(j) not in temp_set:
                temp_set.append(set(j))
        
        if len(temp_set) > 1:
            return 0
        return 1
    
    temp = data.groupby(feature).agg({feature_label:judge})
    index = [i == 0 for i in temp[feature_label].values]
    res = list(temp.index[index])

    return res


#字典映射，作为Embedding的索引
def dict_generate(datas):
    
    k = 2
    data_dict = {-1:1}
        
    for data in datas:
        
        if data == -1:
            continue
        data_dict[data] = k
        k += 1

    
    return data_dict

def fill_na(data, column):
    
    temp = data[column].values
    res = []
    for i in temp:
        if type(i) is np.float64 or type(i) is float:
            res.append('-1')
        else:
            res.append(i)
            
    return res
            
#处理序列类型的feature，得到对应的类型数目，序列长度，column类型对应编码字典，column-feature对应列表，feature类型对应编码字典，最后一个元素所在位置+下一feature_pos序列+下一feature_neg序列组成的res
#对于同一sn或者featuretype下multi特征不统一的处理函数，取到对应长度最长的序列作为最终序列
#同一长度下（长度>1）的序列只是顺序不一样
def multi_column_info(data, column, feature, maxlen=None, dict_base=None, gametype_dict=None, sep=',', index=True, item_cate=None):
    
    featuretype = list(set(data[feature].values))
    featuretype_dict = dict_generate(featuretype)
    padding = {}
    
    print(column, len(featuretype))
    
    def find(x):
        x_temp = [i for i in x.values if i != '-1']
        x_len = [len(str(i).split(sep)) for i in x_temp]
        if x_len:
            return x_temp[np.argmax(x_len)]
        return '-1'
     
    if index:
        temp = data.groupby(feature).agg({column:find})
        for i in zip(list(temp.index), list(temp[column].values)):
            padding[featuretype_dict[i[0]]] = list(str(i[1]).split(sep))
    else:
        for i in zip(list(data[feature].values), list(data[column].values)):
            padding[featuretype_dict[i[0]]] = list(str(i[1]).split(sep))

    if maxlen is not None:
        max_len = maxlen
        featuretype_label = []
        for i in range(2, len(featuretype)+2):
            featuretype_label += padding[i]
    else:
        featuretype_label = []
        max_len = 0
        for i in range(2, len(featuretype)+2):
            featuretype_label += padding[i]
            if max_len < len(padding[i]):
                max_len = len(padding[i])
     
    if dict_base is None:
        featuretype_label = set(featuretype_label)
        featuretype_label_num = len(featuretype_label)
        featuretype_label_dict = dict_generate(featuretype_label)
    else:
        featuretype_label_num = len(dict_base.keys())
        featuretype_label_dict = dict_base
       
    featuretype_label_list = [[0] * max_len, [1]+[0] * (max_len-1)]
    last_sequence_list = [[0] * max_len, [1]+[0] * (max_len-1)]
    
    pos = [[0] * max_len, [1]+[0] * (max_len-1)]
    neg = [[0] * max_len, [1]+[0] * (max_len-1)]
    if gametype_dict:
        gd = list(gametype_dict.values())
        random.shuffle(gd)
        gdl = len(gd)
    
    t = 0
    for i in range(2, len(featuretype)+2):
        if max_len-len(padding[i]) < 0:
            temp_l = len(padding[i])
            padding[i] = padding[i][temp_l-max_len:temp_l]
        if column in ['interest_gametype', 'pv']:
            if i % 10000 == 0:
                print(i)

            temp = [gametype_dict[int(j)] if int(j) in gametype_dict.keys() else 1 for j in padding[i]] + [0] * (max_len-len(padding[i]))
            temp1 = [0] * (len(padding[i])-1) + [1] + [0] * (max_len-len(padding[i]))
            featuretype_label_list.append(temp)
            last_sequence_list.append(temp1)
            pos.append(temp[1:]+[0])
            if index:
                neg_list = list(set(gametype_dict.values()).difference(set(temp)))
                neg.append([random.choice(neg_list) for i in range(max_len)])
            else:
                cate_list = [item_cate[j][0] for j in temp]
                neg_list = []
                j = gd[t%gdl]
                for i in range(max_len):
                    while j in set(temp) or item_cate[j][0] in cate_list:
                        t += 1
                        j = gd[t%gdl]
                    
                    neg_list.append(j)
                    t += 1
                neg.append(neg_list)
        else:
            temp = [featuretype_label_dict[j] if j in featuretype_label_dict.keys() else 1 for j in padding[i]] + [0] * (max_len-len(padding[i]))
            temp1 = [0] * (len(padding[i])-1) + [1] + [0] * (max_len-len(padding[i]))
            featuretype_label_list.append(temp)
            last_sequence_list.append(temp1)
            
    res = [np.array(last_sequence_list), pos, neg]
            
    return featuretype_label_num, max_len, featuretype_dict, np.array(featuretype_label_list), featuretype_label_dict, res
            

#处理单一长度类型的feature，若是跟sn或者gametype独立的特征，则需要得到对应的以样本数为行数的list
#若是非独立的特征，则需要返回对应sn或者gametype个数为行数的list 
def one_column_inde_info(data, column, base_dict=None):
    
    data_column_list = list(data[column].values)

    if base_dict is not None:
        data_column_dict = base_dict
        k = len(base_dict.keys()) - 1
    else:
        data_column_set = list(set(data_column_list))
        data_column_dict = {-1:1}
        k = 2
        
        for i in data_column_set:
            if i == -1:
                continue
            
            data_column_dict[i] = k
            k += 1

    data_column_list = [data_column_dict[i] if i in data_column_dict.keys() else 1 for i in data_column_list]
        
    return k-1, data_column_list, data_column_dict
    

def one_column_de_info(data, column, depend):
    
    pass
  

#按照gametype进行数据采样，app_minus_inclued控制是否包含两个app_list都是-1的样本
def data_build(data, num, app_minus_inclued=True, sep=','):
    
    k = 1
    gametype_rate_dict = {}
    
    if not app_minus_inclued:
        index_1 = list(data['android_app'] != '-1')
        index_2 = list(data['ios_app'] != '-1')
        index = [index_1[i] or index_2[i] for i in range(len(data))]
        data_1 = data[index]
        
        index_1 = list(data_1['android_app'] != '-1')
        index_2 = list(data_1['ios_app'] != '-1')
        index_temp = [index_1[i] and index_2[i] for i in range(len(data_1))]
        
        if sum(index_temp) > 0:
            temp = data_1[index_temp].agg(lambda x:x['android_app'] if len(x['android_app'].split(sep)) > len(x['ios_app'].split(sep)) else x['ios_app'])
            data_1[index_temp]['android_app'] = temp
            data_1[index_temp]['ios_app'] = '-1'
        
        data = data_1
        
    def fun1(x):
        return sum(x==1)/len(x)
    def fun2(x, total=len(data)):
        return len(x)/total
    
    gametype_ratio = data.groupby('gametype').agg({'label':[fun1, fun2]})
    gametype_ratio.columns = ['label_1_ratio', 'gametype_ratio']
    
    #gametype:[pos_ratio, gametype_ratio]
    for i in zip(gametype_ratio.index, gametype_ratio.iloc[:,0].values, gametype_ratio.iloc[:,1].values):
        gametype_rate_dict[i[0]] = [i[1], i[2]]
    
    num_temp = num // len(gametype_rate_dict.keys())

    for i in gametype_rate_dict.keys():
        data_temp = data[data['gametype'] == i]
        if len(data_temp) == 0:
            continue
        gametype_num = int(num*gametype_rate_dict[i][1])
        
#        if gametype_num < 100:
#            continue
        if gametype_num < num_temp:
            data_temp_res = data_temp
        else:
            pos_num = int(gametype_num*gametype_rate_dict[i][0])
            neg_num = gametype_num - pos_num
                    
            data_temp_pos = data_temp[data_temp['label'] == 1]
            data_temp_neg = data_temp[data_temp['label'] == 0]
            pos_index = random.sample(list(range(len(data_temp_pos))), pos_num)
            neg_index = random.sample(list(range(len(data_temp_neg))), neg_num)
            data_temp_res = pd.concat([data_temp_neg.iloc[neg_index,:], data_temp_pos.iloc[pos_index,:]])
 
        if k == 1:
            data_res = data_temp_res
        else:
            data_res = pd.concat([data_res, data_temp_res])     
        k += 1
        print(k)
   
    return data_res
   
#保存数据
def app_to_df(data, sep=','):
    
    index_1 = list(data['android_app'] != '-1')
    index_2 = list(data['ios_app'] != '-1')
    index = [index_1[i] or index_2[i] for i in range(len(data))]
    data_temp = data[index]
    
    index_1 = list(data_temp['android_app'] != '-1')
    index_2 = list(data_temp['ios_app'] != '-1')
    index_temp = [index_1[i] and index_2[i] for i in range(len(data_temp))]
    
    if sum(index_temp) > 0:
        temp = data_temp[index_temp].agg(lambda x:x['android_app'] if len(x['android_app'].split(sep)) > len(x['ios_app'].split(sep)) else x['ios_app'])
        data_temp[index_temp]['android_app'] = temp
        data_temp[index_temp]['ios_app'] = '-1'
    
    android_app = list(data_temp['android_app'].values)
    ios_app = list(data_temp['ios_app'].values)
    
    app_temp = [ios_app[i] if android_app[i] == '-1' else android_app[i] for i in range(len(data_temp))]
    hist_temp = data_temp['interest_gametype'].values
    
    res = pd.DataFrame({'gametype':data_temp['gametype'].values, 'label':data_temp['label'].values, 'app_all':app_temp, 'hist':hist_temp})
    
    return res

#特征截断
def column_len_density_cut(data, column, reserve_ratio=1.0, sep=','):
    
    column_list = data[column].values
    column_len = [len(i.split(sep)) for i in column_list if i != '-1']
    reverse_num = len(column_len) * reserve_ratio
    
    temp_1, temp_2, _ = plt.hist(column_len, bins=100)
    plt.xlabel('squence_length')
    plt.ylabel('count_ratio')
    plt.title(column+' histplot')
    plt.show()
    
    res = 0
    for i in range(len(temp_1)):
        res += temp_1[i]
        if res >= reverse_num:
            break
    
    return temp_2[i]

#特征柱状图
def column_count_bar(data, column, t='test'):
    
    temp_dict = {}
    for i in data[column]:
        if temp_dict.get(i):
            temp_dict[i] += 1
        else:
            temp_dict[i] = 1
            
    temp1, temp2 = zip(*list(temp_dict.items()))
    plt.bar(list(range(len(temp1))), temp2)
    plt.xticks(list(range(len(temp1))), temp1, fontsize=2)
    plt.xlabel(column)
    plt.ylabel('count')
    plt.title(t + ' ' + column + ' barplot')
    plt.show()
    
#计算特征的tf-idf值用于筛选特征，未使用
def column_tf_generate(data, column, reserve_num, sep=','):
    
    temp = list(data[column].values)
    l = len(temp) + 1
    idf_dict = {}
    
    temp_list_all = []
    for i in temp:
        temp_list = i.split(sep)
        temp_list_all.append(temp_list)
        for j in set(temp_list):
            if idf_dict.get(j):
                idf_dict[j] += 1/l
            else:
                idf_dict[j] = 2/l
    
    res = []
    for i in temp_list_all:
        temp_idf = [[j, idf_dict[j]] for j in i]
        temp_idf = sorted(temp_idf, key=lambda x:x[1], reverse=True)
        temp_idf = [j[0] for j in temp_idf]
        res.append(temp_idf[:reserve_num])
    
    return res

#训练集分布与原始数据集分布一致
def train_test_split(data, train_num, gametype_rate_dict, adjusted=True):
    
    train_index = []   
    test_index = []
    index_all = np.arange(len(data))
    for i in gametype_rate_dict.keys():
        index_temp_pos = index_all[(data['gametype'] == i).values * (data['label'] == 1).values]
        index_temp_neg = index_all[(data['gametype'] == i).values * (data['label'] == 0).values]

        gametype_num = int(train_num*gametype_rate_dict[i][1])
        pos_num = int(gametype_num*gametype_rate_dict[i][0])
        neg_num = gametype_num - pos_num
        
        if adjusted:
            if len(index_temp_pos) - pos_num < 5:
                pos_num = len(index_temp_pos) - 5
            if len(index_temp_neg) - neg_num < 5:
                neg_num = len(index_temp_neg) - 5
                
        pos_index_train = random.sample(index_temp_pos.tolist(), pos_num)
        neg_index_train = random.sample(index_temp_neg.tolist(), neg_num)
        train_index = train_index + pos_index_train + neg_index_train
        
        pos_index_test = list(set(index_temp_pos).difference(set(pos_index_train)))
        neg_index_test = list(set(index_temp_neg).difference(set(neg_index_train)))
        test_index = test_index + pos_index_test + neg_index_test
   
    return train_index, test_index

#测试集均匀分布
def train_test_split_all(data, test_num, column='gametype'):
    
    temp_dict = {}
    for i in data[column]:
        if temp_dict.get(i):
            temp_dict[i] += 1
        else:
            temp_dict[i] = 1
            
    temp_list = list(temp_dict.items())
    num = test_num // len(temp_list)
    
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()

    for i in temp_list:
        print(i[0])
        temp_data = data[data[column] == i[0]]
        
        if i[1] < 100:
            data_test = pd.concat([data_test, temp_data])
        
        elif i[1] < num*2:
            temp_index_pos = temp_data[temp_data['label'] == 1]
            temp_index_neg = temp_data[temp_data['label'] == 0]
    
            pos_num = len(temp_index_pos) // 2
            neg_num = len(temp_index_neg) // 2
            
            if pos_num < 5:
                data_test = pd.concat([data_test, temp_index_pos])
            elif neg_num < 5:
                data_test = pd.concat([data_test, temp_index_neg])
            else:
                index_temp_pos = np.arange(len(temp_index_pos))
                index_temp_neg = np.arange(len(temp_index_neg))
                
                pos_index_test = random.sample(index_temp_pos.tolist(), pos_num)
                neg_index_test = random.sample(index_temp_neg.tolist(), neg_num)
                pos_index_train = list(set(index_temp_pos).difference(set(pos_index_test)))
                neg_index_train = list(set(index_temp_neg).difference(set(neg_index_test)))  
                
                data_test = pd.concat([data_test, temp_index_pos.iloc[pos_index_test]])
                data_test = pd.concat([data_test, temp_index_neg.iloc[neg_index_test]])
                data_train = pd.concat([data_train, temp_index_pos.iloc[pos_index_train]])
                data_train = pd.concat([data_train, temp_index_neg.iloc[neg_index_train]])
        
        else:
            temp_index_pos = temp_data[temp_data['label'] == 1]
            temp_index_neg = temp_data[temp_data['label'] == 0]
    
            pos_num = neg_num = num
            
            if len(temp_index_pos) < pos_num:
                pos_num = len(temp_index_pos) // 2
            elif len(temp_index_neg) < neg_num:
                neg_num = len(temp_index_neg) // 2

            index_temp_pos = np.arange(len(temp_index_pos))
            index_temp_neg = np.arange(len(temp_index_neg))
            
            pos_index_test = random.sample(index_temp_pos.tolist(), pos_num)
            neg_index_test = random.sample(index_temp_neg.tolist(), neg_num)   
            pos_index_train = list(set(index_temp_pos).difference(set(pos_index_test)))
            neg_index_train = list(set(index_temp_neg).difference(set(neg_index_test)))   
            
            data_test = pd.concat([data_test, temp_index_pos.iloc[pos_index_test]])
            data_test = pd.concat([data_test, temp_index_neg.iloc[neg_index_test]])
            data_train = pd.concat([data_train, temp_index_pos.iloc[pos_index_train]])
            data_train = pd.concat([data_train, temp_index_neg.iloc[neg_index_train]])

    return data_train, data_test            
    
                         

def sample_weight_generate(gametype_rate_dict):
    
    temp = gametype_rate_dict.values()
    _, gametype_ratio_temp = [i[0] for i in temp], [i[1] for i in temp]
    gametype_ratio_temp_inv = list(map(lambda x:1/(x+0.0001), gametype_ratio_temp))
#    gametype_ratio_temp_inv_sum = sum(gametype_ratio_temp_inv)
#    gametype_ratio_temp_softmax = list(map(lambda x:x/gametype_ratio_temp_inv_sum, gametype_ratio_temp_inv))
        
    k = 0
    sample_weght = {}
    for i in gametype_rate_dict.keys():
        sample_weght[str(i) + '0'] = gametype_ratio_temp_inv[k]
        sample_weght[str(i) + '1'] = gametype_ratio_temp_inv[k]
        k += 1
        
    return sample_weght
   
#nlp feature
def nlp_weight_dict(r, gametype_dict):
    
    gametype_nlp_weight = pd.read_csv(r, header=None, sep='\t')
    gametype_nlp_weight.columns = ['gametype', 'nlp_weight']

    nlp_weights = gametype_nlp_weight['nlp_weight'].values
    gametypes = gametype_nlp_weight['gametype'].values
    
    l = len(gametype_nlp_weight)
    gametype_nlp_weight_dict = {}
    keys = gametype_dict.keys()
    
    for i in range(l):
        nlp_weight = nlp_weights[i]
        nlp_weight_list = nlp_weight.split('#ws#')
        nlp_weight_list = [j.split('#dist#') for j in nlp_weight_list]
        nlp_weight_list = [[j[0], float(j[1])] for j in nlp_weight_list]                                
        nlp_weight_dict = dict(nlp_weight_list)
        if gametypes[i] not in keys:
            continue
        gametype_nlp_weight_dict[gametype_dict[gametypes[i]]] = nlp_weight_dict
        
    return gametype_nlp_weight_dict

def nlp_wor2vec_dict(r):
    
    word2vec = {}  
    with open(r, 'rb') as f:
        temp = f.readline().decode('utf-8')
        while temp != '':
            temp = temp.replace('\n', '')
            temp_list = temp.split(' ')
            word2vec[temp_list[0]] = list(map(float, temp_list[1:-1]))
            temp1 = f.readline()
            index = True
            while index:
                try:
                    temp = temp1.decode('utf-8')
                    index = False
                except:
                    temp1 = f.readline()
                    
    return word2vec

def gametype_word2evc_list(gametype_nlp_weight_dict, word2vec, gametype_num, weight=True):
    
    l = 200
    word2evc_list = [[0.0]*l for i in range(gametype_num+1)]


    for i in gametype_nlp_weight_dict.items():
        gametype = i[0]
        word2vec_weights = i[1]
        temp = None
        for j in word2vec_weights.keys():
            if temp is None:
                temp = np.array(word2vec[j]) * word2vec_weights[j]
                sum_temp = word2vec_weights[j]
            else:
                temp += np.array(word2vec[j]) * word2vec_weights[j]
                sum_temp += word2vec_weights[j]
              
        if weight:
            temp = temp / sum_temp
        else:
            temp = temp / l    
            
        temp_norm = temp/np.sqrt(sum(temp ** 2))
            
        word2evc_list[gametype] = temp_norm.tolist()
    
    return np.array(word2evc_list, np.float32)

#old版本
#def multi_column_info(data, column, feature):
#    
#    featuretype = list(set(data[feature].values))
#    featuretype_dict = dict_generate(featuretype)
#    padding = {}
#    
#    print(column, len(featuretype))
#
#    for i in featuretype:
#        temp = list(data[data[feature] == i][column].values)
#        max_len, temp_res = 0, []
#        for j in temp:
#            temp_split = j.split(',')
#            if len(temp_split) >= max_len:
#                if max_len == 1 and temp_split == ['-1']:
#                    continue
#                max_len = len(temp_split)
#                temp_res = [int(k) for k in temp_split]
#        
#        padding[featuretype_dict[i]] = temp_res
#
##        e = time.time()
##        k += 1
##        if(k % 100) == 0:
##            print(e-b)
##        print(k, i)
#                         
##        temp = [j.split(',') for j in temp if j != '-1']
##        
##        if not temp:
##           padding[featuretype_dict[i]] = [-1] 
##           k += 1
##           continue
##       
##        res = []
##        for j in temp:
##            if set(j) in res:
##                continue
##            res.append(set(j))
##        
##        max_len = max([len(j) for j in res])
##        for j in res:
##            if len(j) == max_len:
##                padding[featuretype_dict[i]] = [int(k) for k in list(j)]
##                break
#    featuretype_label = []
#    max_len = 0
#    for i in range(1, len(featuretype)+1):
#        featuretype_label += padding[i]
#        if max_len < len(padding[i]):
#            max_len = len(padding[i])
#        
#    featuretype_label = set(featuretype_label)
#    featuretype_label_num = len(featuretype_label)
#    featuretype_label_dict = dict_generate(featuretype_label)
#    
#    featuretype_label_list = [[0] * max_len]
#    for i in range(1, len(featuretype)+1):
#        temp = [featuretype_label_dict[j] for j in padding[i]] + [0] * (max_len-len(padding[i]))
#        featuretype_label_list.append(temp)
#    
#    return featuretype_label_num, max_len, featuretype_dict, np.array(featuretype_label_list)    











