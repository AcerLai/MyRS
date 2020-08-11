# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:34:47 2019

@author: gzs13133
"""
import tensorflow.keras as keras
import numpy as np
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.models import Model
from keras.initializers import RandomNormal
from keras.layers import Layer
from keras.layers import Input, Activation, Flatten, Embedding, Dense, Lambda, Multiply, Add, Concatenate, Reshape, BatchNormalization
from sklearn.metrics import roc_auc_score


### 自定义callback，用于记录训练进程，可自定义记录的周期、记录的指标等
class roc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, record_num=100):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.record_num = record_num
        self.auc_val = []
        self.loss = []

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #        y_pred = self.model.predict(self.x)
        #        roc = roc_auc_score(self.y, y_pred)  str(round(roc,4)), \rauc_train: %s -

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        # self.auc_val.append(roc_val)

        print('auc_val: %s' % (str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        if batch % self.record_num == 0:
        #     #            y_pred = self.model.predict(self.x)
        #     #            roc = roc_auc_score(self.y, y_pred)

            y_pred_val = self.model.predict(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val)
            self.auc_val.append(roc_val)

        return

### 普通DNN网络，注意输入的是够需要将对应的aid相关的和gametype相关的放在一起
class ccDeep(object):

    def __init__(self, hidden_num=4, dnn_layer=[], prov_num=37, city_num=529, aid_num=8192, gametype_num=157, a_dist_num=122, u_dist_num=81, bn=False):

        self.hidden_num = hidden_num
        self.prov_len = 2
        self.city_len = 2
        self.aid_len = 30 + 30
        self.gametype_len = 30 + 30 + 30 + 1
        self.a_dist_len = 13
        self.u_dist_len = 11
        self.bn = bn

        self.prov = Input(shape=(self.prov_len,), dtype='int32', name='f_prov')
        self.city = Input(shape=(self.city_len,), dtype='int32', name='f_city')
        self.aid = Input(shape=(self.aid_len,), dtype='int32', name='f_aid')
        self.gametype = Input(shape=(self.gametype_len,), dtype='int32', name='f_gametype')
        self.a_dist = Input(shape=(self.a_dist_len,), dtype='int32', name='f_a_dist')
        self.u_dist = Input(shape=(self.u_dist_len,), dtype='int32', name='f_u_dist')
        self.a_conts = Input(shape=(3,), dtype='float32', name='f_a_conts')
        self.u_conts = Input(shape=(1,), dtype='float32', name='f_u_conts')
        self.target = Input(shape=(1,), dtype='int32', name='f_a_aid')
        

        # weight
        self.prov_weight = Input(shape=(self.prov_len,), dtype='float32', name='w_prov')
        self.city_weight = Input(shape=(self.city_len,), dtype='float32', name='w_city')
        self.aid_weight = Input(shape=(self.aid_len,), dtype='float32', name='w_aid')
        self.gametype_weight = Input(shape=(self.gametype_len,), dtype='float32', name='w_gametype')
        self.a_dist_weight = Input(shape=(self.a_dist_len,), dtype='float32', name='w_a_dist')
        self.u_dist_weight = Input(shape=(self.u_dist_len,), dtype='float32', name='w_u_dist')
        self.target_weight = Input(shape=(1,), dtype='float32', name='w_a_aid')
        
        # Embedding
        rn = RandomNormal(mean=0, stddev=0.5)
        prov_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=prov_num, embeddings_initializer=rn,
                      input_length=self.prov_len)
        city_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=city_num, embeddings_initializer=rn,
                      input_length=self.city_len)
        aid_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=aid_num, embeddings_initializer=rn,
                      input_length=self.aid_len)
        gametype_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=gametype_num, embeddings_initializer=rn,
                      input_length=self.gametype_len)
        a_dist_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=a_dist_num, embeddings_initializer=rn,
                      input_length=self.a_dist_len)
        u_dist_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=u_dist_num, embeddings_initializer=rn,
                      input_length=self.u_dist_len)
        ib_Embedding = \
            Embedding(output_dim=1, input_dim=aid_num, embeddings_initializer=rn,
                      input_length=1)

        prov_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.prov_weight)
        prov_input_mul = Multiply()([prov_Embedding(self.prov), prov_weight_exp])
        prov_input = Lambda(lambda x: K.sum(x, axis=-2))(prov_input_mul)

        city_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.city_weight)
        city_input_mul = Multiply()([city_Embedding(self.city), city_weight_exp])
        city_input = Lambda(lambda x: K.sum(x, axis=-2))(city_input_mul)

        aid_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.aid_weight)
        aid_input_mul = Multiply()([aid_Embedding(self.aid), aid_weight_exp])
        aid_input = Lambda(lambda x: K.sum(x, axis=-2))(aid_input_mul)

        gametype_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.gametype_weight)
        gametype_input_mul = Multiply()([gametype_Embedding(self.gametype), gametype_weight_exp])
        gametype_input = Lambda(lambda x: K.sum(x, axis=-2))(gametype_input_mul)

        a_dist_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.a_dist_weight)
        a_dist_input_mul = Multiply()([a_dist_Embedding(self.a_dist), a_dist_weight_exp])
        a_dist_input = Lambda(lambda x: K.sum(x, axis=-2))(a_dist_input_mul)

        u_dist_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.u_dist_weight)
        u_dist_input_mul = Multiply()([u_dist_Embedding(self.u_dist), u_dist_weight_exp])
        u_dist_input = Lambda(lambda x: K.sum(x, axis=-2))(u_dist_input_mul)

        target_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))\
            (self.target_weight)
        target_input_mul = Multiply()([aid_Embedding(self.target), target_weight_exp])
        target_input = Lambda(lambda x: K.sum(x, axis=-2))(target_input_mul)

        model_input = Concatenate()([prov_input, city_input, aid_input, gametype_input, a_dist_input, u_dist_input,
                                     self.a_conts, self.u_conts, target_input])

        # BN对比测试
        if not dnn_layer:
            if self.bn:
                dense_layer_1_out = Dense(512, activation='linear')(model_input)
                dense_layer_1_out = BatchNormalization()(dense_layer_1_out)
                dense_layer_1_out = Activation('relu')(dense_layer_1_out)
                dense_layer_2_out = Dense(256, activation='linear')(dense_layer_1_out)
                dense_layer_2_out = BatchNormalization()(dense_layer_2_out)
                dense_layer_2_out = Activation('relu')(dense_layer_2_out)
                dense_layer_3_out = Dense(128, activation='linear')(dense_layer_2_out)
                dense_layer_3_out = BatchNormalization()(dense_layer_3_out)
                dense_layer_3_out = Activation('relu')(dense_layer_3_out)
            else:
                dense_layer_1_out = Dense(512, activation='relu')(model_input)
                dense_layer_2_out = Dense(256, activation='relu')(dense_layer_1_out)
                dense_layer_3_out = Dense(128, activation='relu')(dense_layer_2_out)

        i_b = Flatten()(ib_Embedding(self.target))

        dense_layer_out = Dense(1, activation='linear')(dense_layer_3_out)
        score = Add()([i_b, dense_layer_out])

        self.prediction = Activation('sigmoid', name='prediction')(score)

        opt = optimizers.Nadam()

        self.model = Model(
            inputs=[self.prov, self.prov_weight, self.city, self.city_weight, self.aid, self.aid_weight, self.gametype,
                    self.gametype_weight, self.a_dist, self.a_dist_weight, self.u_dist, self.u_dist_weight, self.target,
                    self.target_weight, self.a_conts, self.u_conts],\
            outputs=[self.prediction])
        
        self.model.compile(opt, loss=['binary_crossentropy'], \
                               metrics={'prediction':'accuracy'})
                
    def train_model(self, train_feature, train_label, test_feature, test_label, sample_weights=[], batch_size=128, epoch=3, record_num=1000):
        if len(sample_weights) == 0:
            sample_weights = np.array([1.]*len(train_label))
        roc = roc_callback(training_data=[train_feature, train_label], validation_data=[test_feature, test_label], record_num=record_num)
        # sample_weight=sample_weights,
        self.model.fit(train_feature, train_label, batch_size, epochs=epoch, 
                            callbacks=[roc])
        return roc

    def eval_model(self, test_feature, test_label):
        loss, acc = self.model.evaluate(test_feature, test_label)
        return loss, acc

### 固定Attention生成层，构造需要训练的向量用于与序列特征形成Attention层
### AttentionFix层直接输出Attention后的结果向量，AttentionFixWeight只是输出Attention权重向量并tile,主要是为了更灵活地利用权重向量。
class AttentionFix(Layer):     
    
    def __init__(self, initializer, hidden_num, **kwargs): 
        
        self.initializer = initializer
        self.hidden_num = hidden_num
        super(AttentionFix, self).__init__(**kwargs)     
    
    def build(self, input_shape):    
        
        assert isinstance(input_shape, list)        
        self.attention_weight = self.add_weight(name='attention_weight', 
                                      dtype='float32',
                                      shape=(self.hidden_num, 1),                                      
                                      initializer=self.initializer,                                      
                                      trainable=True)
        self.built = True    
    
    def call(self, x):
        
        assert isinstance(x, list)
        feature, weight = x
        index = K.cast(K.cast(weight, dtype='bool'), dtype='float32') # (?,l)
        index_expand = K.tile(K.expand_dims(index, axis=-1), [1, 1, self.hidden_num]) # (?,l,h)
        weight_expand = K.tile(K.expand_dims(weight, axis=-1), [1, 1, self.hidden_num]) # (?,l,h)
        att_res = K.dot(feature, self.attention_weight) # (?,l,1)
        att_res_expand = K.tile(att_res, [1, 1, self.hidden_num]) # (?,l,h)
        att_input = att_res_expand * weight_expand # (?,l,h)
        att_input_exp = K.exp(att_input) * index_expand # (?,l,h)
        att_input_exp_sum = 1/(K.sum(att_input_exp, axis=-2)+1e-6) # (?,h)
        att_output = K.sum(feature * att_input_exp) # (?,h)
        att_output = att_output * att_input_exp_sum # (?,h)

        return att_output      
    
    def compute_output_shape(self, input_shape):
    
        assert isinstance(input_shape, list)
        shape, _ = input_shape        
        return (shape[0], self.hidden_num)


class AttentionFixWeight(Layer):     
    
    def __init__(self, initializer, hidden_num, **kwargs): 
        
        self.initializer = initializer
        self.hidden_num = hidden_num
        super(AttentionFixWeight, self).__init__(**kwargs)     
    
    def build(self, input_shape):    
        
        # assert isinstance(input_shape, list)        
        self.attention_weight = self.add_weight(name='attention_weight', 
                                      dtype='float32',
                                      shape=(self.hidden_num, 1),                                      
                                      initializer=self.initializer,                                      
                                      trainable=True)
        self.built = True    
    
    def call(self, x):
        
        # assert isinstance(x, list)
        att_res = K.dot(x, self.attention_weight) # (?,l,1)
        att_res_expand = K.tile(att_res, [1, 1, self.hidden_num]) # (?,l,h)

        return att_res_expand      
    
    def compute_output_shape(self, input_shape):
    
        # assert isinstance(input_shape, list)
        # shape, _ = input_shape        
        return (input_shape[0], input_shape[1], self.hidden_num)
    
### FM计算函数
def cal_fm(fm_input):
        fm_sum = Lambda(lambda x:K.sum(x, axis=-2))
        fm_sq = Lambda(lambda x:x * x)
        fm_res =  Lambda(lambda x:K.sum(x[0] - x[1], axis=-1, keepdims=True) / 2)
        fm_input_1 = fm_sum(fm_input)
        fm_input_1 = fm_sq(fm_input_1)
        fm_input_2 = fm_sq(fm_input)
        fm_input_2 = fm_sum(fm_input_2)
        fm_output = fm_res([fm_input_1, fm_input_2])
        return fm_output
    
### 带Attention的DNN函数，其中norm代表是否Attention-softmax归一化, attfix代表是否使用固定Attention向量模块, 
### fm代表是否使用fm模块, lr代表是否使用lr模块, reg代表是否对lr和fm进行正则化, sp代表fm在计算特征交互时是否按照特征进行分开交互计算
class ccDeepAtt(object):

    def __init__(self, hidden_num=4, dnn_layer=[], prov_num=37, city_num=529, aid_num=8192, gametype_num=157, 
                 a_dist_num=122, u_dist_num=81, norm=False, attfix=False, fm=False, lr=False, reg=False, sp=False):

        self.hidden_num = hidden_num
        self.prov_len = 2
        self.city_len = 2
        self.aid_len = 30
        self.gametype_len = 30
        self.a_dist_len = 13
        self.u_dist_len = 11
        self.norm = norm
        self.attfix = attfix
        self.fm = fm
        self.lr = lr
        self.reg = reg
        self.sp = sp

        self.prov = Input(shape=(self.prov_len,), dtype='int32', name='f_prov')
        self.city = Input(shape=(self.city_len,), dtype='int32', name='f_city')
        self.aid_1 = Input(shape=(self.aid_len,), dtype='int32', name='f_aid_1')
        self.aid_2 = Input(shape=(self.aid_len,), dtype='int32', name='f_aid_2')
        self.gametype_1 = Input(shape=(self.gametype_len,), dtype='int32', name='f_gametype_1')
        self.gametype_2 = Input(shape=(self.gametype_len,), dtype='int32', name='f_gametype_2')
        self.gametype_3 = Input(shape=(self.gametype_len,), dtype='int32', name='f_gametype_3')
        self.a_dist = Input(shape=(self.a_dist_len,), dtype='int32', name='f_a_dist')
        self.u_dist = Input(shape=(self.u_dist_len,), dtype='int32', name='f_u_dist')
        self.a_conts = Input(shape=(3,), dtype='float32', name='f_a_conts')
        self.u_conts = Input(shape=(1,), dtype='float32', name='f_u_conts')
        self.target_id = Input(shape=(1,), dtype='int32', name='f_a_aid')
        self.target_gametype = Input(shape=(1,), dtype='int32', name='f_a_gametype')

        # weight
        self.prov_weight = Input(shape=(self.prov_len,), dtype='float32', name='w_prov')
        self.city_weight = Input(shape=(self.city_len,), dtype='float32', name='w_city')
        self.aid_weight_1 = Input(shape=(self.aid_len,), dtype='float32', name='w_aid_1')
        self.aid_weight_2 = Input(shape=(self.aid_len,), dtype='float32', name='w_aid_2')
        self.gametype_weight_1 = Input(shape=(self.gametype_len,), dtype='float32', name='w_gametype_1')
        self.gametype_weight_2 = Input(shape=(self.gametype_len,), dtype='float32', name='w_gametype_2')
        self.gametype_weight_3 = Input(shape=(self.gametype_len,), dtype='float32', name='w_gametype_3')
        self.a_dist_weight = Input(shape=(self.a_dist_len,), dtype='float32', name='w_a_dist')
        self.u_dist_weight = Input(shape=(self.u_dist_len,), dtype='float32', name='w_u_dist')
        self.target_id_weight = Input(shape=(1,), dtype='float32', name='w_a_aid')
        self.target_gametype_weight = Input(shape=(1,), dtype='float32', name='w_a_gametype')

# , embeddings_regularizer=regularizers.l2(0.001)
        rn = RandomNormal(mean=0, stddev=0.5)
        prov_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=prov_num, embeddings_initializer=rn,
                      input_length=self.prov_len)
        city_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=city_num, embeddings_initializer=rn,
                      input_length=self.city_len)
        aid_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=aid_num, embeddings_initializer=rn,
                      input_length=self.aid_len)
        gametype_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=gametype_num, embeddings_initializer=rn,
                      input_length=self.gametype_len)
        a_dist_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=a_dist_num, embeddings_initializer=rn,
                      input_length=self.a_dist_len)
        u_dist_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=u_dist_num, embeddings_initializer=rn,
                      input_length=self.u_dist_len)
        ib_id_Embedding = \
            Embedding(output_dim=1, input_dim=aid_num, embeddings_initializer=rn,
                      input_length=1)
        ib_gametype_Embedding = \
            Embedding(output_dim=1, input_dim=gametype_num, embeddings_initializer=rn,
                      input_length=1)
            
        #### FM Layer 
        
        if self.reg:
            prov_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=prov_num, embeddings_initializer=rn,
                          input_length=self.prov_len, embeddings_regularizer=regularizers.l2())
            city_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=city_num, embeddings_initializer=rn,
                          input_length=self.city_len, embeddings_regularizer=regularizers.l2())
            aid_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=aid_num, embeddings_initializer=rn,
                          input_length=self.aid_len * 2 + 1, embeddings_regularizer=regularizers.l2())
            gametype_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=gametype_num, embeddings_initializer=rn,
                          input_length=self.gametype_len * 3 + 1, embeddings_regularizer=regularizers.l2())
            a_dist_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=a_dist_num, embeddings_initializer=rn,
                          input_length=self.a_dist_len, embeddings_regularizer=regularizers.l2())
            u_dist_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=u_dist_num, embeddings_initializer=rn,
                          input_length=self.u_dist_len, embeddings_regularizer=regularizers.l2())
            a_conts_fm_Embedding = Dense(hidden_num // 2, activation='linear', kernel_regularizer=regularizers.l2())
            u_conts_fm_Embedding = Dense(hidden_num // 2, activation='linear', kernel_regularizer=regularizers.l2())
        else:
            prov_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=prov_num, embeddings_initializer=rn,
                          input_length=self.prov_len)
            city_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=city_num, embeddings_initializer=rn,
                          input_length=self.city_len)
            aid_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=aid_num, embeddings_initializer=rn,
                          input_length=self.aid_len * 2 + 1)
            gametype_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=gametype_num, embeddings_initializer=rn,
                          input_length=self.gametype_len * 3 + 1)
            a_dist_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=a_dist_num, embeddings_initializer=rn,
                          input_length=self.a_dist_len)
            u_dist_fm_Embedding = \
                Embedding(output_dim=hidden_num // 2, input_dim=u_dist_num, embeddings_initializer=rn,
                          input_length=self.u_dist_len)
            a_conts_fm_Embedding = Dense(hidden_num // 2, activation='linear')
            u_conts_fm_Embedding = Dense(hidden_num // 2, activation='linear')
        
        
        prov_weight_exp_fm = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.prov_weight)
        prov_fm = prov_fm_Embedding(self.prov)
        prov_fm = Multiply()([prov_fm, prov_weight_exp_fm])
        
        city_weight_exp_fm = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.city_weight)
        city_fm = city_fm_Embedding(self.city)
        city_fm = Multiply()([city_fm, city_weight_exp_fm])     
        
        a_dist_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.a_dist_weight)
        a_dist_fm = a_dist_fm_Embedding(self.a_dist)
        a_dist_fm = Multiply()([a_dist_fm, a_dist_weight_exp])
        
        u_dist_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.u_dist_weight)
        u_dist_fm = u_dist_fm_Embedding(self.u_dist)
        u_dist_fm = Multiply()([u_dist_fm, u_dist_weight_exp])
        
        a_conts_fm = a_conts_fm_Embedding(self.a_conts)
        a_conts_fm = Reshape([-1, hidden_num // 2])(a_conts_fm)
        u_conts_fm = u_conts_fm_Embedding(self.u_conts)
        u_conts_fm = Reshape([-1, hidden_num // 2])(u_conts_fm)        
        
        if not self.sp:
            prov_fm = cal_fm(prov_fm)
            city_fm = cal_fm(city_fm)
            a_dist_fm = cal_fm(a_dist_fm)
            u_dist_fm = cal_fm(u_dist_fm)
            a_conts_fm = cal_fm(a_conts_fm)
            u_conts_fm = cal_fm(u_conts_fm)
        else:
            fm_output = cal_fm(prov_fm)
            fm_output = Add()([cal_fm(city_fm), fm_output])
            fm_output = Add()([cal_fm(a_dist_fm), fm_output])
            fm_output = Add()([cal_fm(u_dist_fm), fm_output])
            fm_output = Add()([cal_fm(a_conts_fm), fm_output])
            fm_output = Add()([cal_fm(u_conts_fm), fm_output])
        
        if not self.sp:
            aid_fm_input = Concatenate(axis=-1)([self.aid_1, self.aid_2, self.target_id])
            aid_fm_weight = Concatenate(axis=-1)([self.aid_weight_1, self.aid_weight_2, self.target_id_weight])
            aid_fm_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(aid_fm_weight)
            aid_fm = aid_fm_Embedding(aid_fm_input)
            aid_fm = Multiply()([aid_fm, aid_fm_weight_exp])
            aid_fm = cal_fm(aid_fm)

            gametype_fm_input = Concatenate(axis=-1)([self.gametype_1, self.gametype_2, self.gametype_3, self.target_gametype])
            gametype_fm_weight = Concatenate(axis=-1)([self.gametype_weight_1, self.gametype_weight_2, self.gametype_weight_3, self.target_gametype_weight])
            gametype_fm_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(gametype_fm_weight)
            gametype_fm = gametype_fm_Embedding(gametype_fm_input)
            gametype_fm = Multiply()([gametype_fm, gametype_fm_weight_exp])
            gametype_fm = cal_fm(gametype_fm)
        else:
            aid_all = [self.aid_1, self.aid_2, self.target_id]
            aid_weight_all = [self.aid_weight_1, self.aid_weight_2, self.target_id_weight]
            for i in range(3):
                aid_fm_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(aid_weight_all[i])
                aid_fm = aid_fm_Embedding(aid_all[i])
                aid_fm = Multiply()([aid_fm, aid_fm_weight_exp])
                fm_output = Add()([cal_fm(aid_fm), fm_output])
                
            gametype_all = [self.gametype_1, self.gametype_2, self.gametype_3, self.target_gametype]
            gametype_weight_all = [self.gametype_weight_1, self.gametype_weight_2, self.gametype_weight_3, self.target_gametype_weight]
            for i in range(3):
                gametype_fm_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(gametype_weight_all[i])
                gametype_fm = gametype_fm_Embedding(gametype_all[i])
                gametype_fm = Multiply()([gametype_fm, gametype_fm_weight_exp])
                fm_output = Add()([cal_fm(gametype_fm), fm_output])
        
        if not self.sp:
            fm_input = Concatenate(axis=-2)([prov_fm, city_fm, aid_fm, gametype_fm, a_dist_fm, u_dist_fm, a_conts_fm, u_conts_fm])
            fm_output = cal_fm(fm_input)
        ####
            
        #### LR Layer
        if self.reg:
            prov_lr_Embedding = \
                Embedding(output_dim=1, input_dim=prov_num, embeddings_initializer=rn,
                          input_length=self.prov_len, embeddings_regularizer=regularizers.l1())
            city_lr_Embedding = \
                Embedding(output_dim=1, input_dim=city_num, embeddings_initializer=rn,
                          input_length=self.city_len, embeddings_regularizer=regularizers.l1())
            aid_lr_Embedding = \
                Embedding(output_dim=1, input_dim=aid_num, embeddings_initializer=rn,
                          input_length=self.aid_len * 2 + 1, embeddings_regularizer=regularizers.l1())
            gametype_lr_Embedding = \
                Embedding(output_dim=1, input_dim=gametype_num, embeddings_initializer=rn,
                          input_length=self.gametype_len * 3 + 1, embeddings_regularizer=regularizers.l1())
            a_dist_lr_Embedding = \
                Embedding(output_dim=1, input_dim=a_dist_num, embeddings_initializer=rn,
                          input_length=self.a_dist_len, embeddings_regularizer=regularizers.l1())
            u_dist_lr_Embedding = \
                Embedding(output_dim=1, input_dim=u_dist_num, embeddings_initializer=rn,
                          input_length=self.u_dist_len, embeddings_regularizer=regularizers.l1())
            a_conts_lr_Embedding = Dense(1, activation='linear', kernel_regularizer=regularizers.l1())
            u_conts_lr_Embedding = Dense(1, activation='linear', kernel_regularizer=regularizers.l1())
        else:
            prov_lr_Embedding = \
                Embedding(output_dim=1, input_dim=prov_num, embeddings_initializer=rn,
                          input_length=self.prov_len)
            city_lr_Embedding = \
                Embedding(output_dim=1, input_dim=city_num, embeddings_initializer=rn,
                          input_length=self.city_len)
            aid_lr_Embedding = \
                Embedding(output_dim=1, input_dim=aid_num, embeddings_initializer=rn,
                          input_length=self.aid_len * 2 + 1)
            gametype_lr_Embedding = \
                Embedding(output_dim=1, input_dim=gametype_num, embeddings_initializer=rn,
                          input_length=self.gametype_len * 3 + 1)
            a_dist_lr_Embedding = \
                Embedding(output_dim=1, input_dim=a_dist_num, embeddings_initializer=rn,
                          input_length=self.a_dist_len)
            u_dist_lr_Embedding = \
                Embedding(output_dim=1, input_dim=u_dist_num, embeddings_initializer=rn,
                          input_length=self.u_dist_len)
            a_conts_lr_Embedding = Dense(1, activation='linear')
            u_conts_lr_Embedding = Dense(1, activation='linear')
        
        prov_weight_exp_lr = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, 1]))(self.prov_weight)
        prov_lr = prov_lr_Embedding(self.prov)
        prov_lr = Multiply()([prov_lr, prov_weight_exp_lr])
        
        city_weight_exp_lr = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, 1]))(self.city_weight)
        city_lr = city_lr_Embedding(self.city)
        city_fm = Multiply()([city_fm, city_weight_exp_lr])
        
        aid_lr_input = Concatenate(axis=-1)([self.aid_1, self.aid_2, self.target_id])
        aid_lr_weight = Concatenate(axis=-1)([self.aid_weight_1, self.aid_weight_2, self.target_id_weight])
        aid_lr_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, 1]))(aid_lr_weight)
        aid_lr = aid_lr_Embedding(aid_lr_input)
        aid_lr = Multiply()([aid_lr, aid_lr_weight_exp])
        
        gametype_lr_input = Concatenate(axis=-1)([self.gametype_1, self.gametype_2, self.gametype_3, self.target_gametype])
        gametype_lr_weight = Concatenate(axis=-1)([self.gametype_weight_1, self.gametype_weight_2, self.gametype_weight_3, self.target_gametype_weight])
        gametype_lr_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, 1]))(gametype_lr_weight)
        gametype_lr = gametype_lr_Embedding(gametype_lr_input)
        gametype_lr = Multiply()([gametype_lr, gametype_lr_weight_exp])
        
        a_dist_weight_exp_lr = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, 1]))(self.a_dist_weight)
        a_dist_lr = a_dist_lr_Embedding(self.a_dist)
        a_dist_lr = Multiply()([a_dist_lr, a_dist_weight_exp_lr])
        
        u_dist_weight_exp_lr = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, 1]))(self.u_dist_weight)
        u_dist_lr = u_dist_lr_Embedding(self.u_dist)
        u_dist_lr = Multiply()([u_dist_lr, u_dist_weight_exp_lr])
        a_conts_lr = a_conts_lr_Embedding(self.a_conts)
        a_conts_lr = Reshape([-1, 1])(a_conts_lr)
        u_conts_lr = u_conts_lr_Embedding(self.u_conts)
        u_conts_lr = Reshape([-1, 1])(u_conts_lr)
        
        lr_input = Concatenate(axis=-2)([prov_lr, city_lr, aid_lr, gametype_lr, a_dist_lr, u_dist_lr, a_conts_lr, u_conts_lr])
        lr_output =  Lambda(lambda x:K.sum(x, axis=-2))(lr_input)
        ####
        
        prov_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.prov_weight)
        prov_input_mul = Multiply()([prov_Embedding(self.prov), prov_weight_exp])
        prov_input = Lambda(lambda x: K.sum(x, axis=-2))(prov_input_mul)


        city_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.city_weight)
        city_input_mul = Multiply()([city_Embedding(self.city), city_weight_exp])
        city_input = Lambda(lambda x: K.sum(x, axis=-2))(city_input_mul)

        # aid_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.aid_weight)
        # aid_input_mul = Multiply()([aid_Embedding(self.aid), aid_weight_exp])
        # aid_input = Lambda(lambda x: K.sum(x, axis=-2))(aid_input_mul)

        # gametype_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
        #     (self.gametype_weight)
        # gametype_input_mul = Multiply()([gametype_Embedding(self.gametype), gametype_weight_exp])
        # gametype_input = Lambda(lambda x: K.sum(x, axis=-2))(gametype_input_mul)
        

        a_dist_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.a_dist_weight)
        a_dist_input_mul = Multiply()([a_dist_Embedding(self.a_dist), a_dist_weight_exp])
        a_dist_input = Lambda(lambda x: K.sum(x, axis=-2))(a_dist_input_mul)

        u_dist_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.u_dist_weight)
        u_dist_input_mul = Multiply()([u_dist_Embedding(self.u_dist), u_dist_weight_exp])
        u_dist_input = Lambda(lambda x: K.sum(x, axis=-2))(u_dist_input_mul)


        index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32')) # (?,l)
        ti = aid_Embedding(self.target_id)
        aid_1 = aid_Embedding(self.aid_1)
        aid_2 = aid_Embedding(self.aid_2)

        ti_exp = Lambda(lambda x: K.tile(x, [1, self.aid_len, 1]))(ti)
        
        tg = gametype_Embedding(self.target_gametype)
        tg_exp = Lambda(lambda x: K.tile(x, [1, self.gametype_len, 1]))(tg)
        gametype_1 = gametype_Embedding(self.gametype_1)
        gametype_2 = gametype_Embedding(self.gametype_2)
        gametype_3 = gametype_Embedding(self.gametype_3)
        
        gametype_weight_exp_2 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.gametype_weight_2)
        gametype_input_mul_2 = Multiply()([gametype_2, gametype_weight_exp_2])
        gametype_input_2 = Lambda(lambda x: K.sum(x, axis=-2))(gametype_input_mul_2)

        
        if self.norm:
            index_1 = index(self.aid_weight_1)
            index_1_expand = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(index_1)
            aid_weight_exp_1 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.aid_weight_1)
            aid_ti_mul_1 = Multiply()([aid_1, ti_exp])
            aid_ti_wei_mul_1 = Multiply()([aid_ti_mul_1, aid_weight_exp_1])
            
            if self.attfix:
                aid_fixatt_1 = AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num)
                fixatt_output_1 = aid_fixatt_1(aid_1)
                aid_ti_wei_mul_1 = Multiply()([aid_ti_mul_1, fixatt_output_1])
             
            aid_ti_att_1 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(aid_ti_wei_mul_1)
            aid_ti_att_exp_1 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num]))(aid_ti_att_1)
            aid_ti_att_exp_1 = Lambda(lambda x: K.exp(x))(aid_ti_att_exp_1)
            aid_ti_att_exp_1 = Multiply()([aid_ti_att_exp_1, index_1_expand])
            aid_ti_att_exp_sum_1 = Lambda(lambda x:1/(K.sum(x, axis=-2)+1e-6))(aid_ti_att_exp_1)
            aid_ti_att_res_1 = Multiply()([aid_1, aid_ti_att_exp_1])
            aid_ti_att_res_1 = Lambda(lambda x: K.sum(x, axis=-2))(aid_ti_att_res_1)
            aid_ti_input_1 = Multiply()([aid_ti_att_exp_sum_1, aid_ti_att_res_1])
            
            index_2 = index(self.aid_weight_2)
            index_2_expand = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(index_2)
            aid_weight_exp_2 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.aid_weight_2)
            aid_ti_mul_2 = Multiply()([aid_2, ti_exp])
            aid_ti_wei_mul_2 = Multiply()([aid_ti_mul_2, aid_weight_exp_2])
            
            if self.attfix:
                aid_fixatt_2 = AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num)
                fixatt_output_2 = aid_fixatt_2(aid_2)
                aid_ti_wei_mul_2 = Multiply()([aid_ti_mul_2, fixatt_output_2])
            
            aid_ti_att_2 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(aid_ti_wei_mul_2)
            aid_ti_att_exp_2 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num]))(aid_ti_att_2)
            aid_ti_att_exp_2 = Lambda(lambda x: K.exp(x))(aid_ti_att_exp_2)
            aid_ti_att_exp_2 = Multiply()([aid_ti_att_exp_2, index_2_expand])
            aid_ti_att_exp_sum_2 = Lambda(lambda x:1/(K.sum(x, axis=-2)+1e-6))(aid_ti_att_exp_2)
            aid_ti_att_res_2 = Multiply()([aid_2, aid_ti_att_exp_2])
            aid_ti_att_res_2 = Lambda(lambda x: K.sum(x, axis=-2))(aid_ti_att_res_2)
            aid_ti_input_2 = Multiply()([aid_ti_att_exp_sum_2, aid_ti_att_res_2])
            
            index_3 = index(self.gametype_weight_1)
            index_3_expand = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(index_3)
            gametype_weight_exp_1 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.gametype_weight_1)
            gametype_tg_mul_1 = Multiply()([gametype_1, tg_exp])
            gametype_tg_wei_mul_1 = Multiply()([gametype_tg_mul_1, gametype_weight_exp_1])
            
            if self.attfix:
                gametype_fixatt_1 = AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num // 2)
                fixatt_output_3 = gametype_fixatt_1(gametype_1)
                gametype_tg_wei_mul_1 = Multiply()([gametype_tg_wei_mul_1, fixatt_output_3])
                
            gametype_tg_att_1 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(gametype_tg_wei_mul_1)
            gametype_tg_att_exp_1 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num // 2]))(gametype_tg_att_1)
            gametype_tg_att_exp_1 = Lambda(lambda x: K.exp(x))(gametype_tg_att_exp_1)
            gametype_tg_att_exp_1 = Multiply()([gametype_tg_att_exp_1, index_3_expand])
            gametype_tg_att_exp_sum_1 = Lambda(lambda x:1/(K.sum(x, axis=-2)+1e-6))(gametype_tg_att_exp_1)
            gametype_tg_att_res_1 = Multiply()([gametype_1, gametype_tg_att_exp_1])
            gametype_tg_att_res_1 = Lambda(lambda x: K.sum(x, axis=-2))(gametype_tg_att_res_1)
            gametype_tg_input_1 = Multiply()([gametype_tg_att_exp_sum_1, gametype_tg_att_res_1])
            
            index_4 = index(self.gametype_weight_3)
            index_4_expand = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(index_4)
            gametype_weight_exp_3 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.gametype_weight_3)
            gametype_tg_mul_3 = Multiply()([gametype_3, tg_exp])
            gametype_tg_wei_mul_3 = Multiply()([gametype_tg_mul_3, gametype_weight_exp_3])
            
            if self.attfix:
                gametype_fixatt_2 = AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num // 2)
                fixatt_output_4 = gametype_fixatt_2(gametype_3)
                gametype_tg_wei_mul_3 = Multiply()([gametype_tg_wei_mul_3, fixatt_output_4])
            
            gametype_tg_att_3 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(gametype_tg_wei_mul_3)
            gametype_tg_att_exp_3 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num // 2]))(gametype_tg_att_3)
            gametype_tg_att_exp_3 = Lambda(lambda x: K.exp(x))(gametype_tg_att_exp_3)
            gametype_tg_att_exp_3 = Multiply()([gametype_tg_att_exp_3, index_4_expand])
            gametype_tg_att_exp_sum_3 = Lambda(lambda x:1/(K.sum(x, axis=-2)+1e-6))(gametype_tg_att_exp_3)
            gametype_tg_att_res_3 = Multiply()([gametype_3, gametype_tg_att_exp_3])
            gametype_tg_att_res_3 = Lambda(lambda x: K.sum(x, axis=-2))(gametype_tg_att_res_3)
            gametype_tg_input_3 = Multiply()([gametype_tg_att_exp_sum_3, gametype_tg_att_res_3])
        else:
            aid_weight_exp_1 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.aid_weight_1)
            aid_ti_mul_1 = Multiply()([aid_1, ti_exp])
            aid_ti_wei_mul_1 = Multiply()([aid_ti_mul_1, aid_weight_exp_1])
            aid_ti_att_1 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(aid_ti_wei_mul_1)
            aid_ti_att_1 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num]))(aid_ti_att_1)
            aid_ti_att_res_1 = Multiply()([aid_1, aid_ti_att_1])
            aid_ti_input_1 = Lambda(lambda x: K.sum(x, axis=-2))(aid_ti_att_res_1)
        
            aid_weight_exp_2 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))(self.aid_weight_2)
            aid_ti_mul_2 = Multiply()([aid_2, ti_exp])
            aid_ti_wei_mul_2 = Multiply()([aid_ti_mul_2, aid_weight_exp_2])
            aid_ti_att_2 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(aid_ti_wei_mul_2)
            aid_ti_att_2 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num]))(aid_ti_att_2)
            aid_ti_att_res_2 = Multiply()([aid_2, aid_ti_att_2])
            aid_ti_input_2 = Lambda(lambda x: K.sum(x, axis=-2))(aid_ti_att_res_2)
            
            gametype_weight_exp_2 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.gametype_weight_1)
            gametype_tg_mul_2 = Multiply()([gametype_1, tg_exp])
            gametype_tg_wei_mul_2 = Multiply()([gametype_tg_mul_2, gametype_weight_exp_2])
            gametype_tg_att_2 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(gametype_tg_wei_mul_2)
            gametype_tg_att_2 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num // 2]))(gametype_tg_att_2)
            gametype_tg_att_res_2 = Multiply()([gametype_1, gametype_tg_att_2])
            gametype_tg_input_1 = Lambda(lambda x: K.sum(x, axis=-2))(gametype_tg_att_res_2)
    
            gametype_weight_exp_3 = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))(self.gametype_weight_3)
            gametype_tg_mul_3 = Multiply()([gametype_3, tg_exp])
            gametype_tg_wei_mul_3 = Multiply()([gametype_tg_mul_3, gametype_weight_exp_3])
            gametype_tg_att_3 = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(gametype_tg_wei_mul_3)
            gametype_tg_att_3 = Lambda(lambda x: K.tile(x, [1, 1, self.hidden_num // 2]))(gametype_tg_att_3)
            gametype_tg_att_res_3 = Multiply()([gametype_3, gametype_tg_att_3])
            gametype_tg_input_3 = Lambda(lambda x: K.sum(x, axis=-2))(gametype_tg_att_res_3)

        target_id_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num]))\
            (self.target_id_weight)
        target_id_input_mul = Multiply()([ti, target_id_weight_exp])
        target_id_input = Lambda(lambda x: K.sum(x, axis=-2))(target_id_input_mul)
        
        target_gametype_weight_exp = Lambda(lambda x: K.tile(K.expand_dims(x, axis=-1), [1, 1, self.hidden_num // 2]))\
            (self.target_gametype_weight)
        target_gametype_input_mul = Multiply()([tg, target_gametype_weight_exp])
        target_gametype_input = Lambda(lambda x: K.sum(x, axis=-2))(target_gametype_input_mul)
        
        # aid_fixatt_1 = AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num)
        
        # AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num)
        # AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num // 2)
        # AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num // 2)
        # AttentionFixWeight(initializer=rn, hidden_num=self.hidden_num // 2)
        
        
        # if self.attfix:
        #     aid_fixatt_1 = AttentionFix(initializer=rn, hidden_num=self.hidden_num)
        #     aid_fixatt_2 = AttentionFix(initializer=rn, hidden_num=self.hidden_num)
        #     gametype_fixatt_1 = AttentionFix(initializer=rn, hidden_num=self.hidden_num // 2)
        #     gametype_fixatt_2 = AttentionFix(initializer=rn, hidden_num=self.hidden_num // 2)
        #     gametype_fixatt_3 = AttentionFix(initializer=rn, hidden_num=self.hidden_num // 2)
            
        #     aid_fixatt_output_1 = aid_fixatt_1([aid_1, self.aid_weight_1])
        #     aid_fixatt_output_2 = aid_fixatt_2([aid_2, self.aid_weight_2])
        #     gametype_fixatt_output_1 = gametype_fixatt_1([gametype_1, self.gametype_weight_1])
        #     gametype_fixatt_output_2 = gametype_fixatt_2([gametype_2, self.gametype_weight_2])
        #     gametype_fixatt_output_3 = gametype_fixatt_3([gametype_3, self.gametype_weight_3])

        #     model_input = Concatenate()([prov_input, city_input, aid_ti_input_1, aid_ti_input_2, 
        #                                  gametype_input_2, gametype_tg_input_1, gametype_tg_input_3, 
        #                                  aid_fixatt_output_1, aid_fixatt_output_2, gametype_fixatt_output_1,
        #                                  gametype_fixatt_output_2, gametype_fixatt_output_3,
        #                                  a_dist_input, u_dist_input, self.a_conts, self.u_conts, 
        #                                  target_id_input, target_gametype_input])
        # else:     
        model_input = Concatenate()([prov_input, city_input, aid_ti_input_1, aid_ti_input_2, 
                                     gametype_input_2, gametype_tg_input_1, gametype_tg_input_3, 
                                     a_dist_input, u_dist_input, self.a_conts, self.u_conts, 
                                     target_id_input, target_gametype_input])
        model_input = BatchNormalization()(model_input)

        if not dnn_layer:
            dense_layer_1_out = Dense(512, activation='relu')(model_input)
            dense_layer_2_out = Dense(256, activation='relu')(dense_layer_1_out)
            dense_layer_3_out = Dense(128, activation='relu')(dense_layer_2_out)

            
        i_b_id = Flatten()(ib_id_Embedding(self.target_id))
        i_b_gametype = Flatten()(ib_gametype_Embedding(self.target_gametype))
        
        dense_layer_out = Dense(1, activation='linear')(dense_layer_3_out)
        score = Add()([i_b_id, i_b_gametype, dense_layer_out])
        
        if self.fm:
            score = Add()([score, fm_output])
        if self.lr:
            score = Add()([score, lr_output])

        self.prediction = Activation('sigmoid', name='prediction')(score)

        opt = optimizers.Nadam()

        self.model = Model(
            inputs=[self.prov, self.prov_weight, self.city, self.city_weight, self.aid_1, self.aid_weight_1, self.aid_2, 
                    self.aid_weight_2, self.gametype_1, self.gametype_weight_1, self.gametype_2, self.gametype_weight_2, 
                    self.gametype_3, self.gametype_weight_3, self.a_dist, self.a_dist_weight, self.u_dist, self.u_dist_weight, 
                    self.target_id, self.target_id_weight, self.target_gametype, self.target_gametype_weight, self.a_conts, self.u_conts],\
            outputs=[self.prediction])
        
        self.model.compile(opt, loss=['binary_crossentropy'], \
                               metrics={'prediction':'accuracy'})
                
    def train_model(self, train_feature, train_label, test_feature, test_label, sample_weights=[], batch_size=128, epoch=3, record_num=1000):
        if len(sample_weights) == 0:
            sample_weights = np.array([1.]*len(train_label))
        roc = roc_callback(training_data=[train_feature, train_label], validation_data=[test_feature, test_label], record_num=record_num)
        # sample_weight=sample_weights,
        self.model.fit(train_feature, train_label, batch_size, epochs=epoch, 
                            callbacks=[roc])
        return roc

    def eval_model(self, test_feature, test_label):
        loss, acc = self.model.evaluate(test_feature, test_label)
        return loss, acc
    
    
        
        return loss, acc