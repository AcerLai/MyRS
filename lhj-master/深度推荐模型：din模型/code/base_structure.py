# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:34:19 2019

@author: gzs13133
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import roc_auc_score
import keras
from keras import backend as K
from keras.layers import Layer
from keras.layers.advanced_activations import ReLU, Dice
from keras.layers import Reshape, Dense, Lambda, Multiply, Add, Concatenate


def draw(auc_val_list, train_num, batch_size, record_num, use_Activa, i, use_weight=None):
    if use_weight is not None:
        label = use_weight
    else:
        label = ['auc_val_One', 'auc_val_Multi']

    x_index = (np.arange(len(auc_val_list[0].auc_val))+1) / (train_num // batch_size // record_num)
    plt.figure(i)
    for i in range(len(auc_val_list)):
        plt.plot(x_index, auc_val_list[i].auc_val, label=label[i])
    plt.xlabel('epochs')
    plt.ylabel('auc')
    plt.legend()
    plt.title('The training process using '+use_Activa)
    
def draw_epoch(auc_val, use_Activa, batch_size, train_num, record_num, i):
    x_index = (np.arange(len(auc_val))+1) / (train_num // batch_size // record_num)
    plt.figure(i)
    plt.plot(x_index, auc_val, label='auc_val_'+use_Activa)
    plt.xlabel('epochs')
    plt.ylabel('auc')
    plt.legend()
    plt.title('The training process')
    
def dict_generate(x):
    
    temp = {}
    for i in x:
        if temp.get(i):
            temp[i] += 1
        else:
            temp[i] = 1
    
    return temp
    
def avg_auc(y_true, y_pred, x, x_dict):
    
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    x = np.array(x)
    auc_sum = 0
    for i in x_dict.keys():
        index = (x == i)
        auc = roc_auc_score(y_true_array[index], y_pred_array[index])
        auc_sum += auc * x_dict[i]
    
    avg_auc = auc_sum / len(y_true)
    
    return avg_auc
       
#AUC的计算需要整体数据，如果直接在batch里算，误差就比较大，不能合理反映整体情况。这里采用回调函数写法，每个epoch计算一次
class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, record_num=100, use_Transformer=False, use_avgauc=False):
        
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]        
        self.y_val = validation_data[1]
        self.record_num = record_num
        self.use_Transformer = use_Transformer
        self.gametype_dict = dict_generate(self.x_val[1])
        self.use_avgauc = use_avgauc
#        self.item_val = set(validation_data[0][1])
#        self.val_len = len(validation_data[0][1])
#        self.x_val_sample = np.array([list(i) for i in zip(*self.x_val)])
        
    
    def on_train_begin(self, logs={}):
#        self.auc_train = []
#        self.auc_val_item = []
        self.auc_val = []
        self.loss = []
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
#        y_pred = self.model.predict(self.x)
#        roc = roc_auc_score(self.y, y_pred)  str(round(roc,4)), \rauc_train: %s -    
        if self.use_Transformer:
            y_pred_val, _, _, _, _, _ = self.model.predict(self.x_val)
        else:
            y_pred_val, _, _, _, _ = self.model.predict(self.x_val)
        
        if self.use_avgauc:
            roc_val = avg_auc(self.y_val, y_pred_val, self.x_val[1], self.gametype_dict) 
        else:
            roc_val = roc_auc_score(self.y_val, y_pred_val) 
             
#        auc_temp = []
#        for i in self.item_val:
#            index = np.arange(self.val_len)[np.array(self.x_val[1]) == i]
#            x_val = [list(i) for i in zip(*self.x_val_sample[index].tolist())]
#            y_pred_val_temp = self.model.predict(x_val)
#            roc_val_temp = roc_auc_score(self.y_val[index], y_pred_val_temp) 
#            auc_temp.append(roc_val_temp)
#        
#        self.auc_val_item.append(auc_temp)
        
        print('auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        if batch % self.record_num == 0:
#            y_pred = self.model.predict(self.x)
#            roc = roc_auc_score(self.y, y_pred)      
            
            if self.use_Transformer:
                y_pred_val, _, _, _, _, _ = self.model.predict(self.x_val)
            else:
                y_pred_val, _, _, _, _ = self.model.predict(self.x_val)
                
            if self.use_avgauc:
                roc_val = avg_auc(self.y_val, y_pred_val, self.x_val[1], self.gametype_dict) 
            else:
                roc_val = roc_auc_score(self.y_val, y_pred_val)   
            
#            loss, _ = self.model.evaluate(self.x, self.y)          
#            self.auc_train.append(roc)
            self.auc_val.append(roc_val)
#            self.loss.append(loss)

        return  
    
    def vis_auc(self, use_Activa, batch_size, train_num, i):
        x_index = (np.arange(len(self.auc_val))+1) / (train_num // batch_size // self.record_num)
        plt.figure(i)
#        plt.plot(x_index, self.auc_train, label='auc_train_'+use_Activa)
        plt.plot(x_index, self.auc_val, label='auc_val_'+use_Activa)
        plt.xlabel('epochs')
        plt.ylabel('auc')
        plt.legend()
        plt.title('The training process')
        
    def vis_loss(self, use_Activa, batch_size, train_num, i):
        x_index = (np.arange(len(self.loss))+1) / (train_num // batch_size // self.record_num)
        plt.figure(i)
#        plt.plot(x_index, self.auc_train, label='auc_train_'+use_Activa)
        plt.plot(x_index, self.loss, label='loss_'+use_Activa)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.title('The training process')
      
#index:是否针对多个item求embed; item+weighted:是否对特征针对item加权；index_full：针对不同任务的合并设计；l：item的长度
def merge_sequence(sequence, sequence_all, hidden_num, index=True, item=False, weighted=False, weight=None, index_full=0, l=None):
    
    if index_full == 1:
        temp = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(sequence)
        mask = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, 1, 1, hidden_num]))(temp)
        length = Lambda(lambda x:K.sum(x, axis=-1))(temp)
        length_expand = Lambda(lambda x:K.reshape(K.tile(K.expand_dims(x, axis=-1),[1, 1, 1, hidden_num]), (-1, l, l, hidden_num)))(length)
        length_inv = Lambda(lambda x:(x+1)**(-1))(length_expand)

        hc_all_mask = Multiply()([sequence_all, mask])
        if item and weighted:
            weight = Lambda(lambda x:K.softmax(weight, axis=-1))(weight)
            mask_weight = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, 1, 1, hidden_num]))(weight)
            hc_all_mask = Multiply()([hc_all_mask, mask_weight])
        hc_sum = Lambda(lambda x:K.sum(x, axis=-2))(hc_all_mask)
        hc_sum = Reshape((-1, l, l, hidden_num))(hc_sum)
        
        hc_mean = Multiply()([hc_sum, length_inv])
        
    elif index_full == 0:
        temp = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(sequence)
        mask = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, 1, hidden_num]))(temp)
        length = Lambda(lambda x:K.sum(x, axis=-1))(temp)
        length_expand = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, hidden_num]))(length)
        length_inv = Lambda(lambda x:(x+1)**(-1))(length_expand)
        
        hc_all_mask = Multiply()([sequence_all, mask])
        if item and weighted:
            weight = Lambda(lambda x:K.softmax(weight, axis=-1))(weight)
            mask_weight = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, 1, hidden_num]))(weight)
            hc_all_mask = Multiply()([hc_all_mask, mask_weight])
        hc_sum = Lambda(lambda x:K.sum(x, axis=-2))(hc_all_mask)

        if index:
            hc_sum = Reshape((-1, hidden_num))(hc_sum)
            length_inv = Reshape((-1, hidden_num))(length_inv)
            hc_mean = Multiply()([hc_sum, length_inv])
        else:
            hc_sum = Reshape((-1, l, hidden_num))(hc_sum)
            length_inv = Reshape((-1, l, hidden_num))(length_inv)
            hc_mean = Multiply()([hc_sum, length_inv])

    else:
        temp = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(sequence)
        mask = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, hidden_num]))(temp)
        length = Lambda(lambda x:K.sum(x, axis=-1))(temp)
        length_expand = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, hidden_num]))(length)
        length_inv = Lambda(lambda x:(x+1)**(-1))(length_expand)
        
        hc_all_mask = Multiply()([sequence_all, mask])
        if weighted:
#            weight = Lambda(lambda x:K.softmax(weight, axis=-1))(weight)
            weight_sum = Lambda(lambda x:K.sum(x, axis=-1))(weight)
            weight_sum = Lambda(lambda x:K.reshape(x,(-1, 1)))(weight_sum)
            weight_sum = Lambda(lambda x:1/K.tile(x, (1, hidden_num)))(weight_sum)
            mask_weight = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1),[1, 1, hidden_num]))(weight)
            hc_all_mask = Multiply()([hc_all_mask, mask_weight])
            hc_sum = Lambda(lambda x:K.sum(x, axis=-2))(hc_all_mask)
            hc_sum = Multiply()([hc_sum, weight_sum])
            hc_mean = Reshape((-1, hidden_num))(hc_sum)
        else:
            hc_sum = Lambda(lambda x:K.sum(x, axis=-2))(hc_all_mask)
            hc_sum = Reshape((-1, hidden_num))(hc_sum)
            length_inv = Reshape((-1, hidden_num))(length_inv)
            hc_mean = Multiply()([hc_sum, length_inv])
           
    return hc_mean

def DINattention(sequence, item_all, hist_all, hidden_num, max_hist_len, use_Activa, same=True):
    
    hist_index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(sequence)
    item_all_hist = Lambda(lambda x:K.tile(x, [1, max_hist_len, 1]))(item_all)
    if same:
        item_hist_outp = Add()([item_all_hist, Lambda(lambda x:(-1.)*x)(hist_all)])
        dense_input = Concatenate(axis=-1)([hist_all, item_all_hist])
        dense_input = Concatenate(axis=-1)([dense_input, item_hist_outp])
    else:
        dense_input = Concatenate(axis=-1)([hist_all, item_all_hist])

    
    if use_Activa == 'Sigmoid':
        dense_layer_1_out = Dense(18, activation='sigmoid')(dense_input)
    elif use_Activa == 'ReLU':
        dense_layer_1_out = Dense(18)(dense_input)
        dense_layer_1_out = ReLU()(dense_layer_1_out)
    elif use_Activa == 'Dice':
        dense_layer_1_out = Dense(18)(dense_input)
        dense_layer_1_out = Dice(shared_axes=[1])(dense_layer_1_out)
    else:
        dense_layer_1_out = Dense(18, activation='tanh')(dense_input)
        
    dense_layer_2_out = Dense(1, activation='sigmoid')(dense_layer_1_out)
    
    dense_layer_2_out = Lambda(lambda x:K.tile(x, [1, 1, hidden_num]))(dense_layer_2_out)
    hist_index = Lambda(lambda x:K.tile(K.expand_dims(x, -1), [1, 1, hidden_num]))(hist_index)
    weight = Multiply()([dense_layer_2_out, hist_index])
#    weight = Lambda(lambda x:K.exp(x))(weight)
#    weight_sum = Lambda(lambda x:K.sum(x, axis=-2))(weight)    
    
    h_all = Multiply()([hist_all, weight])
    h_all = Lambda(lambda x:K.sum(x, axis=-2))(h_all)
#    h_all = Multiply()([h_all, Lambda(lambda x:x**-1)(weight_sum)])
    h_all = Reshape((-1, hidden_num))(h_all)
    
    return h_all   

def attention(sequence, item_all, hist_all, hidden_num, max_hist_len, use_weight=None):
    
    hist_index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(sequence)
    hist_index = Lambda(lambda x:K.tile(K.expand_dims(x, -1), [1, 1, hidden_num]))(hist_index)
    item_all_hist = Lambda(lambda x:K.tile(x, [1, max_hist_len, 1]))(item_all)
    dot_product = Multiply()([hist_all, item_all_hist])   
    dot_product = Lambda(lambda x:K.sum(x, axis=-1))(dot_product)
    scale = Lambda(lambda x:1/(K.sum(x ** 2, axis=-1)+0.01) ** (1/2))
    dot_product = Multiply()([dot_product, scale(item_all_hist)]) 
    dot_product = Multiply()([dot_product, scale(hist_all)]) 
    dot_product = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1), [1, 1, hidden_num]))(dot_product)
    weight = dot_product
    
    if use_weight is None:
        h_all = Multiply()([hist_all, weight])
        h_all = Lambda(lambda x:K.sum(x, axis=-2))(h_all)
        h_all = Reshape((-1, hidden_num))(h_all)
    elif use_weight ==  'sum':
        weight_sum = Lambda(lambda x:K.sum(x, axis=-2))(weight)
        h_all = Multiply()([hist_all, weight])
        h_all = Lambda(lambda x:K.sum(x, axis=-2))(h_all)
        h_all = Multiply()([h_all, Lambda(lambda x:x**-1)(weight_sum)])
        h_all = Reshape((-1, hidden_num))(h_all)
    elif use_weight == 'softmax':
        weight = Lambda(lambda x:K.exp(x[0])*x[1])([weight, hist_index])
        weight_sum = Lambda(lambda x:1/K.sum(x, axis=-2))(weight)
        h_all = Multiply()([hist_all, weight])
        h_all = Multiply()([h_all, weight_sum])
        h_all = Lambda(lambda x:K.sum(x, axis=-2))(h_all)
        h_all = Reshape((-1, hidden_num))(h_all)

    return h_all

#class NLPDefaultLayer(Layer):     
#    
#    def __init__(self, initializer, **kwargs):  
#        self.initializer = initializer
#        super(NLPDefaultLayer, self).__init__(**kwargs)     
#    
#    def build(self, input_shape):                
#        self.default_word2vec = self.add_weight(name='default_word2vec', 
#                                      dtype='float32',
#                                      shape=(2, 2),                                      
#                                      initializer=np.tril(np.ones((2,2), dtype='int32')),                                      
#                                      trainable=False)        
#        self.built = True    
#    
#    def call(self, x):
#        item_index = K.cast(K.cast(K.sum(K.cast(K.cast(x, dtype='bool'), dtype='float32'), \
#                                             axis=-1), dtype='bool'), dtype='float32')
#        item_index = 1-item_index
#        item_index = K.reshape(item_index, (-1, x.shape[1], 1))
#        DefaultWord2vec = K.dot(item_index, self.default_word2vec)
#
#        return DefaultWord2vec      
#    
#    def compute_output_shape(self, input_shape):
#        return input_shape
    

# for a-li-data, use relu + only one hid-hid layer for two coder
# for cc-data, use sigmoid + nums=4 encoder + nums=1 decoder
class InterestTransformerEncoder(Layer):     
    
    def __init__(self, initializer, max_hist_len, hidden_num, nums=4, use_weight='softmax', use_Activa='sigmoid', **kwargs): 
        
        self.initializer = initializer
        self.max_hist_len = max_hist_len
        self.hidden_num = hidden_num
        self.nums = nums
        self.use_weight = use_weight
        self.use_Activa = use_Activa
        super(InterestTransformerEncoder, self).__init__(**kwargs)     
    
    def build(self, input_shape):    
        
        assert isinstance(input_shape, list)        
        self.dense_weight_1 = self.add_weight(name='encoder_dense_weight_1', 
                                      dtype='float32',
                                      shape=(self.hidden_num, self.hidden_num),                                      
                                      initializer=self.initializer,                                      
                                      trainable=True)
#        self.dense_weight_2 = self.add_weight(name='encoder_dense_weight_2', 
#                                      dtype='float32',
#                                      shape=(self.hidden_num*self.nums, self.hidden_num),                                      
#                                      initializer=self.initializer,                                      
#                                      trainable=True)         
        self.built = True    
    
    def call(self, x):
        
        assert isinstance(x, list)

        sequence, hist_all  = x
        sequence_temp = K.reshape(sequence, (-1, self.max_hist_len))
        hist_all_temp = K.reshape(hist_all, (-1, self.max_hist_len, self.hidden_num))
    
        hist_index = K.cast(K.cast(sequence_temp, dtype='bool'), dtype='float32')
        hist_index = K.tile(K.expand_dims(K.tile(K.expand_dims(hist_index, -1), [1, 1, self.hidden_num]), axis=1), [1, self.max_hist_len, 1, 1])
        hist_all_1 = K.tile(K.expand_dims(hist_all_temp, 1), [1, self.max_hist_len, 1, 1]) #第二层：每个维度重复hist_all
        hist_all_2 = K.tile(K.expand_dims(hist_all_temp, -2), [1, 1, self.max_hist_len, 1])#第二层：每个维度重复对应的hist_all的某个hist
    
        dot_product = hist_all_1 * hist_all_2   
        dot_product = K.sum(dot_product, axis=-1)
        
        scale_1 = 1 / (K.sum(hist_all_1 ** 2, axis=-1) + 0.01) ** (1/2)
        scale_2 = 1 / (K.sum(hist_all_2 ** 2, axis=-1) + 0.01) ** (1/2)
        dot_product = dot_product * scale_1
        dot_product = dot_product * scale_2
    
        dot_product = K.tile(K.expand_dims(dot_product, axis=-1), [1, 1, 1, self.hidden_num])
        weight = dot_product 
        
        if self.use_weight is None:
            h_all = hist_all_1* weight
            h_all = K.sum(h_all, axis=-2)
        elif self.use_weight ==  'sum':
            weight_sum = K.sum(weight, axis=-2)
            h_all = hist_all_1 * weight
            h_all = K.sum(h_all, axis=-2)
            h_all = h_all / weight_sum
        elif self.use_weight == 'softmax':
            weight = K.exp(weight) * hist_index
            weight_sum = K.sum(weight, axis=-2)
            h_all = hist_all_1 * weight
            h_all = K.sum(h_all, axis=-2)
            h_all = h_all / weight_sum
            
        h_all = K.reshape(h_all, (-1, self.max_hist_len, self.hidden_num))
#        hist_all_temp
        h_all_add = h_all + hist_all_temp
        
        hist_index = K.cast(K.cast(sequence_temp, dtype='bool'), dtype='float32')
        hist_index = K.tile(K.expand_dims(hist_index, -1), [1, 1, self.hidden_num])

        h_all_add = K.relu(K.dot(h_all_add, self.dense_weight_1))
#        h_all_add = K.sigmoid(K.dot(h_all_add, self.dense_weight_2))
        h_all_add = h_all_add * hist_index

        return h_all_add      
    
    def compute_output_shape(self, input_shape):
        
        assert isinstance(input_shape, list)
        _, shape_b = input_shape
        
        return (shape_b[0], self.max_hist_len, self.hidden_num)

class InterestTransformerDecoder(Layer):     
    
    def __init__(self, initializer, max_hist_len, max_hist_len_1, hidden_num, \
                 nums=1, use_weight='softmax', use_self_weight='softmax', \
                 **kwargs): 
        
        self.initializer = initializer
        self.max_hist_len = max_hist_len
        self.max_hist_len_1 = max_hist_len_1
        self.hidden_num = hidden_num
        self.nums = nums
        self.use_weight = use_weight
        self.use_self_weight = use_self_weight
        super(InterestTransformerDecoder, self).__init__(**kwargs)     
    
    def build(self, input_shape):  
        
        assert isinstance(input_shape, list)        
        self.dense_weight_1 = self.add_weight(name='decoder_dense_weight_1', 
                                      dtype='float32',
                                      shape=(self.hidden_num, self.hidden_num*self.nums),                                      
                                      initializer=self.initializer,                                      
                                      trainable=False)
#        self.dense_weight_2 = self.add_weight(name='decoder_dense_weight_2', 
#                                      dtype='float32',
#                                      shape=(self.hidden_num*self.nums, self.hidden_num),                                      
#                                      initializer=self.initializer,                                      
#                                      trainable=True)         
        self.built = True       
        
    def call(self, x):
        
        assert isinstance(x, list)
        #sequence:记录长度;last_sequenct:记录最后一个元素的位置;hist_i:历史嵌入向量序列(?,l,h)
        sequence, last_sequence, sequence_1, hist_1, hist_2  = x
        if len(sequence.shape) == 2:
            index = True
        else:
            index = False
        
        if index:
            sequence_temp = K.reshape(sequence, (-1, self.max_hist_len))
            sequence_1_temp = K.reshape(sequence_1, (-1, self.max_hist_len_1))
            last_sequence_temp = K.reshape(K.cast(last_sequence, dtype='float32'), (-1, self.max_hist_len))
            last_sequence_temp = K.tile(K.expand_dims(last_sequence_temp, -1), [1, 1, self.hidden_num])
            hist_1_temp = K.reshape(hist_1, (-1, self.max_hist_len, self.hidden_num))
        
            hist_index = K.cast(K.cast(sequence_temp, dtype='bool'), dtype='float32')
            hist_index = K.tile(K.expand_dims(K.tile(K.expand_dims(hist_index, -1), [1, 1, self.hidden_num]), axis=1), [1, self.max_hist_len, 1, 1])
            hist_all_1 = K.tile(K.expand_dims(hist_1_temp, -3), [1, self.max_hist_len, 1, 1]) #第二层：每个维度重复hist_all
            hist_all_2 = K.tile(K.expand_dims(hist_1_temp, -2), [1, 1, self.max_hist_len, 1])#第二层：每个维度重复对应的hist_all的某个hist
            
            self_dot_product = hist_all_1 * hist_all_2   
            self_dot_product = K.sum(self_dot_product, axis=-1)
    
            self_scale_1 = 1 / (K.sum(hist_all_1 ** 2, axis=-1) + 0.01) ** (1/2)
            self_scale_2 = 1 / (K.sum(hist_all_2 ** 2, axis=-1) + 0.01) ** (1/2)
    
            self_dot_product = self_dot_product * self_scale_1
            self_dot_product = self_dot_product * self_scale_2
    
            self_dot_product = K.tile(K.expand_dims(self_dot_product, axis=-1), [1, 1, 1, self.hidden_num])
            self_weight = self_dot_product
            
            if self.use_self_weight is None:
                h_self = hist_all_1* self_weight
                h_self = K.sum(h_self, axis=-2)
            elif self.use_self_weight ==  'sum':
                self_weight_sum = K.sum(self_weight, axis=-2)
                h_self = hist_all_1 * self_weight
                h_self = K.sum(h_self, axis=-2)
                h_self = h_self / self_weight_sum
            elif self.use_self_weight == 'softmax':
                self_weight = K.exp(self_weight) * hist_index
                self_weight_sum = K.sum(self_weight, axis=-2)
                h_self = hist_all_1 * self_weight
                h_self = K.sum(h_self, axis=-2)
                h_self = h_self / self_weight_sum

                
            h_self = K.reshape(h_self, (-1, self.max_hist_len, self.hidden_num))
    #        hist_1_temp
            h_all_add = h_self + hist_1_temp
            
            hist_index = K.cast(K.cast(sequence_temp, dtype='bool'), dtype='float32')
            hist_index = K.tile(K.expand_dims(hist_index, -1), [1, 1, self.hidden_num])
            h_all_add = h_all_add * hist_index

            h_last = h_all_add * last_sequence_temp
            h_last = K.reshape(K.sum(h_last, axis=-2), (-1, 1, self.hidden_num))
            
            hist_index_1 = K.cast(K.cast(sequence_1_temp, dtype='bool'), dtype='float32')
            hist_index_1 = K.tile(K.expand_dims(hist_index_1, -1), [1, 1, self.hidden_num])
    
            hist_last_all = K.tile(h_last, [1, self.max_hist_len_1, 1])
            dot_product = hist_2 * hist_last_all
            dot_product = K.sum(dot_product, axis=-1)
            
            scale_1 = 1 / (K.sum(hist_2 ** 2, axis=-1) + 0.01) ** (1/2)
            scale_2 = 1 / (K.sum(hist_last_all ** 2, axis=-1) + 0.01) ** (1/2)
            dot_product = dot_product * scale_1
            dot_product = dot_product * scale_2
            dot_product = K.tile(K.expand_dims(dot_product, axis=-1), [1, 1, self.hidden_num])
            weight = dot_product 
            
            if self.use_weight is None:
                h = hist_2 * weight
                h = K.sum(h, axis=-2)
            elif self.use_weight ==  'sum':
                weight_sum = K.sum(weight, axis=-2)
                h = hist_2 * weight
                h = K.sum(h, axis=-2)
                h = h / weight_sum
            elif self.use_weight == 'softmax':
                weight = K.exp(weight) * hist_index_1
                weight_sum = K.sum(weight, axis=-2)
                h = hist_2 * weight
                h = K.sum(h, axis=-2)
                h = h / weight_sum
                        
            h = K.relu(K.dot(h, self.dense_weight_1))
#            h = K.sigmoid(K.dot(h, self.dense_weight_2))
            h = K.reshape(h, (-1, 1, self.hidden_num))
 
        else:
            sequence_temp = K.reshape(sequence, (-1, self.max_hist_len, self.max_hist_len))
            sequence_1_temp = K.tile(K.reshape(sequence_1, (-1, 1, self.max_hist_len_1)), (1, self.max_hist_len, 1))
            last_sequence_temp = K.cast(last_sequence, dtype='float32')
            last_sequence_temp = K.tile(K.expand_dims(last_sequence_temp, -1), [1, 1, self.hidden_num])
            hist_1_temp = K.reshape(hist_1, (-1, self.max_hist_len, self.max_hist_len, self.hidden_num))
        
            hist_index = K.cast(K.cast(sequence_temp, dtype='bool'), dtype='float32')
            hist_index = K.tile(K.expand_dims(K.tile(K.expand_dims(hist_index, -1), [1, 1, 1, self.hidden_num]), axis=-3), [1, 1, self.max_hist_len, 1, 1])
            hist_all_1 = K.tile(K.expand_dims(hist_1_temp, -3), [1, 1, self.max_hist_len, 1, 1]) #第二层：每个维度重复hist_all
            hist_all_2 = K.tile(K.expand_dims(hist_1_temp, -2), [1, 1, 1, self.max_hist_len, 1])#第二层：每个维度重复对应的hist_all的某个hist
            
            self_dot_product = hist_all_1 * hist_all_2   
            self_dot_product = K.sum(self_dot_product, axis=-1)
    
            self_scale_1 = 1 / (K.sum(hist_all_1 ** 2, axis=-1) + 0.01) ** (1/2)
            self_scale_2 = 1 / (K.sum(hist_all_2 ** 2, axis=-1) + 0.01) ** (1/2)
    
            self_dot_product = self_dot_product * self_scale_1
            self_dot_product = self_dot_product * self_scale_2
    
            self_dot_product = K.tile(K.expand_dims(self_dot_product, axis=-1), [1, 1, 1, 1, self.hidden_num])
            self_weight = self_dot_product
            
            if self.use_self_weight is None:
                h_self = hist_all_1* self_weight
                h_self = K.sum(h_self, axis=-2)
            elif self.use_self_weight ==  'sum':
                self_weight_sum = K.sum(self_weight, axis=-2)
                h_self = hist_all_1 * self_weight
                h_self = K.sum(h_self, axis=-2)
                h_self = h_self / self_weight_sum
            elif self.use_self_weight == 'softmax':
                self_weight = K.exp(self_weight) * hist_index
                self_weight_sum = K.sum(self_weight, axis=-2)
                h_self = hist_all_1 * self_weight
                h_self = K.sum(h_self, axis=-2)
                h_self = h_self / self_weight_sum
                
            h_self = K.reshape(h_self, (-1, self.max_hist_len, self.max_hist_len, self.hidden_num))
    #        hist_1_temp
            h_all_add = h_self + hist_1_temp
            
            hist_index = K.cast(K.cast(sequence_temp, dtype='bool'), dtype='float32')
            hist_index = K.tile(K.expand_dims(hist_index, -1), [1, 1, 1, self.hidden_num])
            h_all_add = h_all_add * hist_index

            h_last = h_all_add * last_sequence_temp
            h_last = K.reshape(K.sum(h_last, axis=-2), (-1, self.max_hist_len, 1, self.hidden_num))
            
            hist_index_1 = K.cast(K.cast(sequence_1_temp, dtype='bool'), dtype='float32')
            hist_index_1 = K.tile(K.expand_dims(hist_index_1, -1), [1, 1, 1, self.hidden_num])
    
            hist_last_all = K.tile(h_last, [1, 1, self.max_hist_len_1, 1])
            hist_2_temp = K.tile(K.expand_dims(hist_2, axis=-3), [1, self.max_hist_len, 1, 1])
            dot_product = hist_2_temp * hist_last_all
            dot_product = K.sum(dot_product, axis=-1)
            
            scale_1 = 1 / (K.sum(hist_2_temp ** 2, axis=-1) + 0.01) ** (1/2)
            scale_2 = 1 / (K.sum(hist_last_all ** 2, axis=-1) + 0.01) ** (1/2)
            dot_product = dot_product * scale_1
            dot_product = dot_product * scale_2
            dot_product = K.tile(K.expand_dims(dot_product, axis=-1), [1, 1, 1, self.hidden_num])
            weight = dot_product
            
            if self.use_weight is None:
                h = hist_2_temp * weight
                h = K.sum(h, axis=-2)
            elif self.use_weight ==  'sum':
                weight_sum = K.sum(weight, axis=-2)
                h = hist_2_temp * weight
                h = K.sum(h, axis=-2)
                h = h / weight_sum
            elif self.use_weight == 'softmax':
                weight = K.exp(weight) * hist_index_1
                weight_sum = K.sum(weight, axis=-2)
                h = hist_2_temp * weight
                h = K.sum(h, axis=-2)
                h = h / weight_sum
            
            h = K.relu(K.dot(h, self.dense_weight_1))
#            h = K.sigmoid(K.dot(h, self.dense_weight_2))
            h = K.reshape(h, (-1, self.max_hist_len, self.hidden_num))

        return h  
    
    def compute_output_shape(self, input_shape):
        
        assert isinstance(input_shape, list)
        shape, _, _, _, _ = input_shape
        if len(shape) == 2:
            res = (shape[0], 1, self.hidden_num)
        else:
            res = (shape[0], self.max_hist_len, self.hidden_num)
        
        return res

class IdentityLayer(Layer):     
    
    def __init__(self, max_hist_len, **kwargs):  
        self.max_hist_len = max_hist_len
        super(IdentityLayer, self).__init__(**kwargs)     
    
    def build(self, input_shape):                
        self.auxiliary_index_i = self.add_weight(name='index_i', 
                                      dtype='float32',
                                      shape=(self.max_hist_len, self.max_hist_len),                                      
                                      initializer='zeros',                                      
                                      trainable=False)        
        self.auxiliary_index_i = K.variable(np.identity(self.max_hist_len)) 
        self.built = True    
    
    def call(self, x):
        return self.auxiliary_index_i      
    
    def compute_output_shape(self, input_shape):
        return (self.max_hist_len, self.max_hist_len)

class LowTriLayer(Layer):     
    
    def __init__(self, max_hist_len, **kwargs):  
        self.max_hist_len = max_hist_len
        super(LowTriLayer, self).__init__(**kwargs)     
    
    def build(self, input_shape):                
        self.auxiliary_index_ltr = self.add_weight(name='index_ltr', 
                                      dtype='float32',
                                      shape=(self.max_hist_len, self.max_hist_len),                                      
                                      initializer='zeros',                                      
                                      trainable=False)        
        self.auxiliary_index_ltr = K.variable(np.tril(np.ones((self.max_hist_len, self.max_hist_len)))) 
        self.built = True    
    
    def call(self, x):
        return self.auxiliary_index_ltr      
    
    def compute_output_shape(self, input_shape):
        return (self.max_hist_len, self.max_hist_len)
    
def simcos(x, y):
    
    L2norm = Lambda(lambda x:1/((K.sum(x ** 2, axis=-1)+0.01) ** (1/2)))
    dotpro = Lambda(lambda x:K.sum(x[0]*x[1], axis=-1))
    L2norm_x = L2norm(x)
    L2norm_y = L2norm(y)
    res = Multiply()([dotpro([x, y]), L2norm_x])
    res = Multiply()([dotpro([x, y]), L2norm_y])
    
    return res

def fm_merge(one_features, hidden_num, multi_features=[]):
    
    res = 0
    len_one = len(one_features)
    len_multi = len(multi_features)
   
    for i in range(len_one):
        for j in range(i+1, len_one):
            one_feature_i = Lambda(lambda x:K.reshape(x, (-1, 1, hidden_num)))(one_features[i])
            one_feature_j = Lambda(lambda x:K.reshape(x, (-1, 1, hidden_num)))(one_features[j])
            
            temp = Lambda(lambda x:K.reshape(K.sum(x[0] * x[1], axis=-1), (-1, 1)))([one_feature_i, one_feature_j])
            if res == 0:
                res = temp
            else:
                res = Add()([res, temp])
            
    for i in range(len_multi):
        for j in range(i+1, len_multi):
            multi_feature_index_i = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(multi_features[i][0])
            multi_feature_index_i = Lambda(lambda x:K.reshape(x, (-1, multi_features[i][2])))(multi_feature_index_i)
            multi_feature_index_i = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1), (1, 1, multi_features[j][2])))(multi_feature_index_i)
            
            multi_feature_i = Lambda(lambda x:K.reshape(x, (-1, multi_features[i][2], hidden_num)))(multi_features[i][1])
            multi_feature_i = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-2), (1, 1, multi_features[j][2], 1)))(multi_feature_i)
                        
            multi_feature_index_j = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(multi_features[j][0])
            multi_feature_index_j = Lambda(lambda x:K.reshape(x, (-1, multi_features[j][2])))(multi_feature_index_j)
            multi_feature_index_j = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-2), (1, multi_features[i][2], 1)))(multi_feature_index_j)
            
            multi_feature_j = Lambda(lambda x:K.reshape(x, (-1, multi_features[j][2], hidden_num)))(multi_features[j][1])
            multi_feature_j = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-3), (1, multi_features[i][2], 1, 1)))(multi_feature_j)

            temp = Lambda(lambda x:K.sum(x[0] * x[1], axis=-1))([multi_feature_i, multi_feature_j]) 
            temp = Multiply()([temp, multi_feature_index_i]) 
            temp = Multiply()([temp, multi_feature_index_j])
            temp = Lambda(lambda x:K.reshape(K.sum(K.sum(x, axis=-1), axis=-1), (-1, 1)))(temp)
            res = Add()([res, temp])
            
    for i in range(len_one):
        for j in range(len_multi):
            one_feature_i = Lambda(lambda x:K.reshape(x, (-1, 1, hidden_num)))(one_features[i])
            one_feature_i = Lambda(lambda x:K.tile(x, (1, multi_features[j][2], 1)))(one_feature_i)
            
            multi_feature_index_j = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(multi_features[j][0])
            multi_feature_index_j = Lambda(lambda x:K.reshape(x, (-1, multi_features[j][2])))(multi_feature_index_j)
            multi_feature_j = Lambda(lambda x:K.reshape(x, (-1, multi_features[j][2], hidden_num)))(multi_features[j][1])

            temp = Lambda(lambda x:K.sum(x[0] * x[1], axis=-1))([one_feature_i, multi_feature_j])
            temp = Multiply()([temp, multi_feature_index_j])
            temp = Lambda(lambda x:K.reshape(K.sum(x, axis=-1), (-1, 1)))(temp)
            res = Add()([res, temp])
         
    res = Lambda(lambda x:K.reshape(x, (-1, 1, 1)))(res)
    
    return res