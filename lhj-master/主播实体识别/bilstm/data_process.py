# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:52:19 2019

@author: gzs13133
"""

import random
import numpy as np
import os
from collections import defaultdict
import re
from RegexFilterUtil import RegexFilterUtil

def emoj_sub(text):
    emt2c = defaultdict(lambda: 'ᚠ')
    emt2c['[emts]_sys_0016_电到了(R)[/emts]'] = 'ᚰ'
    emt2c['[emts]_sys_0015_啵(BO)[/emts]'] = 'ᛀ'
    emt2c['[emts]_sys_0009_尴尬:$[/emts]'] = 'ᛐ'
    emt2c['[emts]_sys_0063_亲亲(K)[/emts]'] = 'ᛠ'
    EMT_PATTERN = r'\[emts\].*?\[/emts\]'  # 懒惰匹配，应对多个表情
    return re.sub(EMT_PATTERN, lambda x: emt2c[x.group()], text)

def url_sub(text):
    for pattern in (RegexFilterUtil.IMG_PATTERN,
                    RegexFilterUtil.USER_CARD_PATTERN,
                    RegexFilterUtil.GROUP_LINK_PATTERN,
                    RegexFilterUtil.ROOM_LINK_PATTERN,
                    # RegexFilterUtil.URL_PATTERN,
                    ):
        if re.search(pattern, text) is not None:
            text = re.sub(pattern, '', text)
    return text


def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

def get_X_Y(train_data_x, train_data_y, label_size, word2index, label2index, UNK_ID, test_mode=False, csv_form=True):
    """
    get X and Y given input and labels
    input:
    train_data_x:
    train_data_y:
    label_size: number of total unique labels(e.g. 1999 in this task)
    output:
    X,Y
    """
    X=[]
    Y=[]
    if test_mode:
        train_data_x_tiny_test=train_data_x[0:1000] # todo todo todo todo todo todo todo todo todo todo todo todo 
        train_data_y_tiny_test=train_data_y[0:1000] # todo todo todo todo todo todo todo todo todo todo todo todo 
    else:
        if csv_form:
            train_data_x_tiny_test=train_data_x.values
            train_data_y_tiny_test=train_data_y.values
        else:
            train_data_x_tiny_test=train_data_x
            train_data_y_tiny_test=train_data_y

    for row in train_data_x_tiny_test:
        if type(row) is float:
            row_all_id_list = []
        else:
            row_all = [i for i in row]
        # transform to indices
            row_all_id_list=[word2index.get(x,UNK_ID) for x in row_all]
        # merge title and desc: in the middle is special token 'SEP'
        X.append(row_all_id_list)
#         if index<3: print(index,title_char_id_list)
#         if index%100000==0: print(index,title_char_id_list)

    for row in train_data_y_tiny_test:
        row_index = label2index[row]
        label_list_sparse=transform_multilabel_as_multihot(row_index, label_size)
        Y.append(label_list_sparse)
#         if index%100000==0: print(index,";label_list_dense:",label_list_dense)

    return X, Y

def get_text_X(test_x, index2word):
    with open('test_x.txt', 'w', encoding='utf-8') as f:
        for line in test_x:
            temp = [index2word[i] for i in list(line) if i > 0]
            temp = ''.join(temp) + '\n'
            f.write(temp)    
        
def train_test_split(X, Y, num_test, max_sentence_length):
    X = pad_sequences(X, maxlen=max_sentence_length, value=0.)# padding to max length

    xy=list(zip(X, Y))
    random.Random(10000).shuffle(xy)
    X, Y=zip(*xy)
    X=np.array(X); Y=np.array(Y)
    num_examples=len(X)
    num_train=num_examples - num_test
    train_X, train_Y=X[0:num_train], Y[0:num_train]
    test_X, test_Y=X[num_train:], Y[num_train:]
    
    return train_X, train_Y, test_X, test_Y

def clean_doc():
    if os.path.exists('vocab_all.txt'):
        os.remove('vocab_all.txt')
    if os.path.exists('label_set.txt'):
        os.remove('label_set.txt')
    if os.path.exists('data.h5'):
        os.remove('data.h5')
    if os.path.exists('vocab_label.pik'):
        os.remove('vocab_label.pik')        
    if os.path.exists('test_x.txt'):
        os.remove('test_x.txt')
        