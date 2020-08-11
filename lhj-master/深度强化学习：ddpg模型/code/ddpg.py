# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:45:55 2019

@author: gzs13133
"""
import os
os.chdir('C://Users//GZS13133//intern//data//data_with_hist')
from cc_din import merge_sequence

import keras
import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Concatenate, Lambda, Dense, Add, Layer, Multiply
from keras.initializers import RandomNormal
from keras.models import Model
from keras import optimizers


class roc_callback_with_update(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, soft_factor, update_time, model_type, record_num=100):
        
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]        
        self.y_val = validation_data[1]
        self.record_num = record_num
        self.soft_factor = soft_factor
        self.update_time = update_time
        self.model_type = model_type
    
    def on_train_begin(self, logs={}):
        self.acc = []
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):        
        
        if self.model_type == 'actor':
            pass
#            self.reward_val, _ = self.model.predict(self.x_val)
#            acc = np.mean([self.y_val[i] == y_pred_val[i] for i in range(len(self.y_val))])
            
#        print('acc_val: %s' % (str(round(acc, 4))), end=100*' '+'\n')
        
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        if batch % self.record_num == 0 and self.model_type == 'actor':
            pass
#            y_pred_val, _ = self.model.predict(self.x_val)
#            acc = np.mean([self.y_val[i] == y_pred_val[i] for i in range(len(self.y_val))])
#            self.acc.append(acc)
        
        if (batch+1) % self.update_time == 0:
            if self.model_type == 'actor':
                pass
                
            if self.model_type != 'actor':
                cur_actor_weight_1 = self.model.get_layer('cur_actor_dense_1').get_weights()
                cur_actor_weight_2 = self.model.get_layer('cur_actor_dense_2').get_weights()
            
                tar_actor_weight_1 = self.model.get_layer('tar_actor_dense_1').get_weights()
                tar_actor_weight_2 = self.model.get_layer('tar_actor_dense_2').get_weights()
                
                temp_actor_weight_1 = cur_actor_weight_1
                for i in range(len(temp_actor_weight_1)):
                    temp_actor_weight_1[i] = self.soft_factor * cur_actor_weight_1[i] + (1-self.soft_factor) * tar_actor_weight_1[i]
                self.model.get_layer('tar_actor_dense_1').set_weights(temp_actor_weight_1)

                temp_actor_weight_2 = cur_actor_weight_2
                for i in range(len(temp_actor_weight_2)):
                    temp_actor_weight_2[i] = self.soft_factor * cur_actor_weight_2[i] + (1-self.soft_factor) * tar_actor_weight_2[i]
                self.model.get_layer('tar_actor_dense_2').set_weights(temp_actor_weight_2)
                
                cur_critic_weight_1 = self.model.get_layer('cur_critic_dense_1').get_weights()
                cur_critic_weight_2 = self.model.get_layer('cur_critic_dense_2').get_weights()
                
                tar_critic_weight_1 = self.model.get_layer('tar_critic_dense_1').get_weights()
                tar_critic_weight_2 = self.model.get_layer('tar_critic_dense_2').get_weights()

                temp_critic_weight_1 = cur_critic_weight_1
                for i in range(len(temp_critic_weight_1)):
                    temp_critic_weight_1[i] = self.soft_factor * cur_critic_weight_1[i] + (1-self.soft_factor) * tar_critic_weight_1[i]
                self.model.get_layer('tar_critic_dense_1').set_weights(temp_critic_weight_1)
                
                temp_critic_weight_2 = cur_critic_weight_2
                for i in range(len(temp_critic_weight_2)):
                    temp_critic_weight_2[i] = self.soft_factor * cur_critic_weight_2[i] + (1-self.soft_factor) * tar_critic_weight_2[i]
                self.model.get_layer('tar_critic_dense_2').set_weights(temp_critic_weight_2)
            
        return  

#compute the mean accurately ,reward-base
#optional:applying moving_average:factor=0.9
def next_state_update(user_cate_cur, user_cate_embed_cur, next_item, reward, hidden_num, weighted=False, weight=None):

    user_cate_cur_temp = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(user_cate_cur)
    user_cate_cur_mask = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1), (1, 1, hidden_num)))(user_cate_cur_temp)
    
    if weighted:
        user_cate_cur_count = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(weight)
    else:
        user_cate_cur_count = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(user_cate_cur_temp)
    
    user_cate_cur_count_u = Add()([user_cate_cur_count, reward])
    user_cate_cur_count_u = Lambda(lambda x:1/K.tile(x, (1, hidden_num)))(user_cate_cur_count_u)
    user_cate_embed_cur_mask = Multiply()([user_cate_cur_mask, user_cate_embed_cur])
    
    if weighted:
        weight_tile =  Lambda(lambda x:K.tile(K.expand_dims(x, axis=-1), (1, 1, hidden_num)))(weight)
        user_cate_embed_cur_sum = Multiply()([user_cate_embed_cur_mask, weight_tile])
    else:
        user_cate_embed_cur_sum = user_cate_embed_cur_mask
        
    user_cate_embed_cur_sum = Lambda(lambda x:K.sum(x, axis=-2))(user_cate_embed_cur_sum)
    user_cate_embed_cur_sum = Lambda(lambda x:K.reshape(x, (-1, hidden_num)))(user_cate_embed_cur_sum)
    
    reward_tile = Lambda(lambda x:K.tile(x, (1, hidden_num)))(reward)
    next_item_temp = Multiply()([next_item, reward_tile])
    user_cate_embed_cur_sum_new = Add()([user_cate_embed_cur_sum, next_item_temp])
    user_cate_embed_cur_sum_new = Multiply()([user_cate_embed_cur_sum_new, user_cate_cur_count_u])
    
    return user_cate_embed_cur_sum_new

def time_norm(time, max_hist_len, mtype='sum'):
    
    if mtype == 'softmax':
        time_mask = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(time)
        time_exp = Lambda(lambda x:K.exp(x))(time)
        time_exp_mask = Multiply()([time_mask, time_exp])
        time_exp_mask_sum = Lambda(lambda x:K.reshape(K.sum(x, axis=-1), (-1, 1)))(time_exp_mask)
        time_exp_mask_sum = Lambda(lambda x:1/K.tile(x, (1, max_hist_len)))(time_exp_mask_sum)
        time_softmax = Multiply()([time_exp, time_exp_mask_sum])
        time_res = Multiply()([time_softmax, time_mask])
   
    elif mtype == 'sum':
        time_sum = Lambda(lambda x:K.reshape(K.sum(x, axis=-1), (-1, 1)))(time)
        time_sum = Lambda(lambda x:1/K.tile(x, (1, max_hist_len)))(time_sum)
        time_res = Multiply()([time, time_sum])
    
    elif mtype == 'None':
        time_res = time
        
    return time_res
    
class Item(Layer):     
    
    def __init__(self, item_num, item_list, item_list_len, **kwargs):  
        self.item_num = item_num
        self.item_list = item_list
        self.item_list_len = item_list_len
        super(Item, self).__init__(**kwargs)     
    
    def build(self, input_shape):                

        item_index = K.constant(np.arange(self.item_num)+2) 
        self.item_all = K.gather(self.item_list, K.cast(item_index, 'int32'))
        self.item_all = K.reshape(self.item_all, (self.item_num, self.item_list_len))
        self.built = True    
    
    def call(self, x):
        return self.item_all      
    
    def compute_output_shape(self, input_shape):
        return (self.item_num, self.item_list_len)

# 无法取出A网络最大的动作进行后续处理，因为argmax无梯度
# 发现新的tf函数可以选取前K大，但是也是没有梯度
#class ActionChoose(Layer):
#    
#    def __init__(self, hidden_num, **kwargs):
#        self.hidden_num = hidden_num
#        super(ActionChoose, self).__init__(**kwargs)
#        
#    def build(self, input_shape):
#        assert isinstance(input_shape, list)
#        self.build = True
#    
#    def call(self, x):
#        assert isinstance(x, list)
#        cur_score, item_all_embed = x
#        max_item_score, max_item_index = K.tf.nn.top_k(cur_score, k=1)
#        max_item_index = K.reshape(max_item_index, (-1, 1))
#        max_item_embed = K.reshape(K.gather(item_all_embed, K.cast(max_item_index, dtype='int32')), (-1, self.hidden_num))
#        
#        return max_item_embed
#
#    def compute_output_shape(self, input_shape):
#        assert isinstance(input_shape, list)
#        shape, _ = input_shape
#        return (shape[0], self.hidden_num)      


class RewardGet(Layer):
    
    def __init__(self, reward_factor, hidden_num, **kwargs):
        self.reward_factor = reward_factor
        self.hidden_num = hidden_num
        super(RewardGet, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.build = True
    
    def call(self, x):
        assert isinstance(x, list)
        if len(x) == 3:      
            cur_score, item_next, item_all_embed = x
        else:
            cur_score, item_next, item_all_embed, item_next_time = x
        
#        item_next_temp = K.reshape(K.gather(item_all_embed, K.cast(item_next-2, dtype='int32')), (-1, self.hidden_num))
        item_next_temp = K.tf.reduce_mean(item_next-2, axis=-1)
        reward = K.tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cur_score, labels=item_next_temp)
        reward = K.reshape(reward, (-1, 1))
        
        if len(x) == 4:
            reward = self.reward_factor * item_next_time / reward 
        else:
            reward = self.reward_factor / reward 
            
        return reward

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        
        if len(input_shape) == 3:
            shape, _, _ = input_shape
        else:
            shape, _, _, _ = input_shape
            
        return (shape[0], 1) 

class GetTopK(Layer):
    
    def __init__(self, k, **kwargs):
        self.k = k
        super(GetTopK, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.build = True
    
    def call(self, x):
        
        _, max_item_index = K.tf.nn.top_k(x, k=self.k)
        max_item_index = K.reshape(max_item_index, (-1, self.k))
        max_item_index = K.cast(max_item_index, dtype='float32')
        
        return max_item_index

    def compute_output_shape(self, input_shape):
        shape = input_shape
        return (shape[0], self.k) 
        
    
class ccDDPG(object):
    
    def __init__(self, item_num, max_hist_len, item_multi_list, item_multi_len,
                 hidden_num=16, decay_factor=0.1, update_time=300, 
                 soft_factor=0.01, reward_factor=12, use_time=False):
        
        rn = RandomNormal(mean=0, stddev=0.5)
        self.decay_factor = decay_factor
        self.update_time = update_time
        self.soft_factor = soft_factor
        self.reward_factor = reward_factor
        self.max_hist_len = max_hist_len
        self.use_time = use_time
        
#        self.user = Input(shape=(1,), dtype='int32', name='user_index')
        self.hist = Input(shape=(self.max_hist_len,), dtype='int32', name='user_hist')
        self.item_all = Item(item_num, item_multi_list, item_multi_len, name='item_index_all')(self.hist)
        self.item_next = Input(shape=(1,), dtype='int32', name='item_next')
        self.hist_time = Input(shape=(self.max_hist_len,), dtype='float32', name='user_hist_time')
        self.item_next_time = Input(shape=(1,), dtype='float32', name='item_next_time')

        item_multi_cate = \
            Lambda(lambda x:K.gather(item_multi_list,  K.cast(x, dtype='int32')))
        item_multi_cate_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=item_num+2, embeddings_initializer=rn, input_length=None, name='item_embedding')
        item_multi_cate_embed = item_multi_cate_Embedding(self.item_all)
        item_all_embed = \
            merge_sequence(self.item_all, item_multi_cate_embed, hidden_num, index_full=2)
        item_all_embed = Lambda(lambda x:K.transpose(K.reshape(x, (-1, hidden_num))))(item_all_embed)
        item_all_embed_gather = Lambda(lambda x:K.transpose(x))(item_all_embed)
        
        hist_time_softmax = time_norm(self.hist_time, self.max_hist_len)
        
        user_hist_multi_cate_cur = item_multi_cate(self.hist)
        user_hist_multi_cate_embed = item_multi_cate_Embedding(user_hist_multi_cate_cur)
        user_hist_multi_cate_embed_cur = \
            merge_sequence(user_hist_multi_cate_cur, user_hist_multi_cate_embed, hidden_num, index=False, l=self.max_hist_len)
        user_hist_multi_cate_embed_cur = Lambda(lambda x:K.reshape(x, (-1, self.max_hist_len, hidden_num)))(user_hist_multi_cate_embed_cur)
        self.cur_state = \
            merge_sequence(self.hist, user_hist_multi_cate_embed_cur, hidden_num, index_full=2, weighted=self.use_time, weight=hist_time_softmax)
        self.cur_state = Lambda(lambda x:K.reshape(x, (-1, hidden_num)))(self.cur_state)

#        user_one_cate_lambda_1 = \
#            Lambda(lambda x:K.gather(user_one_list_1,  K.cast(x, dtype='int32')))
#        user_one_cate_1 = user_one_cate_lambda_1(self.user)
#        user_one_cate_Embedding_1 = \
#            Embedding(output_dim=hidden_num, input_dim=user_one_num_1+2, embeddings_initializer=rn, input_length=1)
#        one_state_1 = user_one_cate_Embedding_1(user_one_cate_1)
#        one_state = one_state_1
        
#        all_state = Flatten()(Concatenate(axis=-1)([multi_state, one_state]))
        
        # current-actor-net(update every time)
        cur_actor_dense_1 = Dense(hidden_num, activation='sigmoid', name='cur_actor_dense_1')
        cur_actor_dense_2 = Dense(hidden_num, activation='sigmoid', name='cur_actor_dense_2')
        # target-actor-net(update sometimes)
        tar_actor_dense_1 = Dense(hidden_num, activation='sigmoid', name='tar_actor_dense_1')
        tar_actor_dense_2 = Dense(hidden_num, activation='sigmoid', name='tar_actor_dense_2')
        
        cur_actor = cur_actor_dense_1(self.cur_state)
        cur_actor = cur_actor_dense_2(cur_actor)
        self.cur_score = Lambda(lambda x:K.dot(x[0], x[1]))([cur_actor, item_all_embed])

        GetK = GetTopK(k=20)
        self.cur_item_choose = GetK(self.cur_score)
#        Actch = ActionChoose(hidden_num=hidden_num)
#        self.cur_action = Actch([self.cur_score, item_all_embed_gather])
        RGet = RewardGet(self.reward_factor, hidden_num)
        if self.use_time:
            self.reward = RGet([self.cur_score, self.item_next, item_all_embed_gather, self.item_next_time])
        else:
            self.reward = RGet([self.cur_score, self.item_next, item_all_embed_gather])
        self.cur_action = cur_actor
        
        # current-critic-net(update every time)
        cur_critic_dense_1 = Dense(hidden_num, activation='sigmoid', name='cur_critic_dense_1')
        cur_critic_dense_2 = Dense(1, activation='sigmoid', name='cur_critic_dense_2')
        
        # target-critic-net(update sometimes)
        tar_critic_dense_1 = Dense(hidden_num, activation='sigmoid', name='tar_critic_dense_1')
        tar_critic_dense_2 = Dense(1, activation='sigmoid', name='tar_critic_dense_2')
        
        q_value_cur_in = Concatenate(axis=-1)([ self.cur_state, self.cur_action])
        self.q_value_cur = cur_critic_dense_1(q_value_cur_in)
        self.q_value_cur = cur_critic_dense_2(self.q_value_cur)
#        q_value_cur_a = Activation('sigmoid')(self.q_value_cur)
#        actor loss
#        self.q_value_tar_a = tar_critic_dense_1(q_value_cur_in)
#        self.q_value_tar_a = tar_critic_dense_2(self.q_value_tar_a)
        self.actor_loss = Lambda(lambda x:-1*x)(self.q_value_cur)
        
        next_item_embed = Lambda(lambda x:K.reshape(K.gather(item_all_embed_gather, K.cast(x, dtype='int32')), (-1, hidden_num)))(self.item_next)
        self.next_state = next_state_update(self.hist, user_hist_multi_cate_embed_cur, \
                                            next_item_embed, self.reward, hidden_num, weighted=self.use_time, weight=hist_time_softmax)
        
        tar_actor = tar_actor_dense_1(self.next_state)
        tar_actor = tar_actor_dense_2(tar_actor)
#        tar_actor = cur_actor_dense_2(tar_actor)
#        self.tar_score = Lambda(lambda x:K.dot(x[0], x[1]))([tar_actor, item_all_embed])
        
#        self.tar_action = Actch([self.tar_score, item_all_embed_gather])
        self.tar_action = tar_actor
        
        q_value_tar_in = Concatenate(axis=-1)([self.next_state, self.tar_action])
        self.q_value_tar = tar_critic_dense_1(q_value_tar_in)
        self.q_value_tar = tar_critic_dense_2(self.q_value_tar)
#        self.q_value_tar = cur_critic_dense_1(q_value_tar_in)
#        self.q_value_tar = cur_critic_dense_2(self.q_value_tar)
        self.q_value_tar = Lambda(lambda x:self.decay_factor*x)(self.q_value_tar)
        self.q_value_y = Add()([self.q_value_tar, self.reward])
        
        # critic loss
        self.critic_loss = Lambda(lambda x:(x[0]-x[1])**2)([self.q_value_y, self.q_value_cur])
               
        
    def train_actor_with_update_cur(self, train_sample, eval_sample, batch_size, epoch=5, record_num=100, weight_dict=None, opt_parm=None):
            
        opt = optimizers.nadam(lr=opt_parm[0], schedule_decay=opt_parm[1])
        self.actor_model = Model(inputs=[self.hist, self.item_next, self.hist_time, self.item_next_time], outputs=[self.cur_item_choose, self.actor_loss])
        #freeze
        if weight_dict is not None:
            cur_actor_weight_1 = weight_dict['cur_actor_weight_1']
            cur_actor_weight_2 = weight_dict['cur_actor_weight_2']
                    
            self.actor_model.get_layer('cur_actor_dense_1').set_weights(cur_actor_weight_1)
            self.actor_model.get_layer('cur_actor_dense_2').set_weights(cur_actor_weight_2)
            
            cur_critic_weight_1 = weight_dict['cur_critic_weight_1']
            cur_critic_weight_2 = weight_dict['cur_critic_weight_2']
            
            self.actor_model.get_layer('cur_critic_dense_1').set_weights(cur_critic_weight_1)
            self.actor_model.get_layer('cur_critic_dense_2').set_weights(cur_critic_weight_2)
#
        for layer in self.actor_model.layers:
            layerName = str(layer.name)
            if layerName.startswith("tar_actor_") or \
            layerName.startswith("tar_critic_") or \
            layerName.startswith("cur_critic_"):
                layer.trainable = False
            if layerName.startswith("cur_actor_"):
                layer.trainable = True
                
        self.actor_model.compile(opt, loss=[lambda y_true, y_pred:K.zeros_like(y_pred)]*1+[lambda y_true, y_pred:y_pred])
        
#        roc = roc_callback_with_update(training_data=[train_sample[0], train_sample[1]], \
#                           validation_data=[eval_sample[0], eval_sample[1]], record_num=record_num, \
#                           soft_factor=self.soft_factor, update_time=self.update_time, model_type='actor')
        self.actor_model.fit([train_sample[0][0], train_sample[0][1], train_sample[0][2], train_sample[0][3]],[train_sample[1]]*2, \
                             batch_size, epochs=epoch)
        
        actor_weight_dict = {}
        
        actor_weight_dict['cur_actor_weight_1'] = self.actor_model.get_layer('cur_actor_dense_1').get_weights()
        actor_weight_dict['cur_actor_weight_2'] = self.actor_model.get_layer('cur_actor_dense_2').get_weights()
                   
        return actor_weight_dict
            
    
    def train_critic_with_update_cur(self, train_sample, eval_sample, batch_size, epoch=5, record_num=100, weight_dict=None, opt_parm=None):

        opt = optimizers.nadam(lr=opt_parm[0], schedule_decay=opt_parm[1])
        self.critic_model = Model(inputs=[self.hist, self.item_next, self.hist_time, self.item_next_time], outputs=[self.reward, self.critic_loss])
        
        if weight_dict is not None:
            if len(weight_dict) > 2:
                cur_actor_weight_1 = weight_dict['cur_actor_weight_1']
                cur_actor_weight_2 = weight_dict['cur_actor_weight_2']
                tar_actor_weight_1 = weight_dict['tar_actor_weight_1']
                tar_actor_weight_2 = weight_dict['tar_actor_weight_2']
                cur_critic_weight_1 = weight_dict['cur_critic_weight_1']
                cur_critic_weight_2 = weight_dict['cur_critic_weight_2']
                tar_critic_weight_1 = weight_dict['tar_critic_weight_1']
                tar_critic_weight_2 = weight_dict['tar_critic_weight_2']
                
                self.critic_model.get_layer('cur_actor_dense_1').set_weights(cur_actor_weight_1)
                self.critic_model.get_layer('cur_actor_dense_2').set_weights(cur_actor_weight_2)
                self.critic_model.get_layer('tar_actor_dense_1').set_weights(tar_actor_weight_1)
                self.critic_model.get_layer('tar_actor_dense_2').set_weights(tar_actor_weight_2)
                self.critic_model.get_layer('cur_critic_dense_1').set_weights(cur_critic_weight_1)
                self.critic_model.get_layer('cur_critic_dense_2').set_weights(cur_critic_weight_2)
                self.critic_model.get_layer('tar_critic_dense_1').set_weights(tar_critic_weight_1)
                self.critic_model.get_layer('tar_critic_dense_2').set_weights(tar_critic_weight_2)
            else:
                cur_actor_weight_1 = weight_dict['cur_actor_weight_1']
                cur_actor_weight_2 = weight_dict['cur_actor_weight_2']
                
                self.critic_model.get_layer('cur_actor_dense_1').set_weights(cur_actor_weight_1)
                self.critic_model.get_layer('cur_actor_dense_2').set_weights(cur_actor_weight_2)
                
        #freeze
        for layer in self.critic_model.layers:
            layerName = str(layer.name)
            if layerName.startswith("cur_actor_") or \
            layerName.startswith("tar_actor_") or \
            layerName.startswith("tar_critic_"):
                layer.trainable = False
            if layerName.startswith("cur_critic_"):
                layer.trainable = True
                
        self.critic_model.compile(opt, loss=[lambda y_true, y_pred:K.zeros_like(y_pred), lambda y_true, y_pred:y_pred])
#        roc = roc_callback_with_update(training_data=[train_sample[0], train_sample[1]], \
#                           validation_data=[eval_sample[0], eval_sample[1]], record_num=record_num, \
#                           soft_factor=self.soft_factor, update_time=self.update_time, model_type='critic')
        self.critic_model.fit([train_sample[0][0], train_sample[0][1], train_sample[0][2], train_sample[0][3]],[train_sample[1]]*2, \
                             batch_size, epochs=epoch)
        
        critic_weight_dict = {} 
        
        cur_actor_weight_1 = self.critic_model.get_layer('cur_actor_dense_1').get_weights()
        cur_actor_weight_2 = self.critic_model.get_layer('cur_actor_dense_2').get_weights()
        cur_critic_weight_1 = self.critic_model.get_layer('cur_critic_dense_1').get_weights()
        cur_critic_weight_2 = self.critic_model.get_layer('cur_critic_dense_2').get_weights()
        
        tar_actor_weight_1 = self.critic_model.get_layer('tar_actor_dense_1').get_weights()
        tar_actor_weight_2 = self.critic_model.get_layer('tar_actor_dense_2').get_weights()
        tar_critic_weight_1 = self.critic_model.get_layer('tar_critic_dense_1').get_weights()
        tar_critic_weight_2 = self.critic_model.get_layer('tar_critic_dense_2').get_weights()

        temp_actor_weight_1 = cur_actor_weight_1
        for i in range(len(temp_actor_weight_1)):
            temp_actor_weight_1[i] = self.soft_factor * cur_actor_weight_1[i] + (1-self.soft_factor) * tar_actor_weight_1[i]
        critic_weight_dict['tar_actor_weight_1'] = temp_actor_weight_1

        temp_actor_weight_2 = cur_actor_weight_2
        for i in range(len(temp_actor_weight_2)):
            temp_actor_weight_2[i] = self.soft_factor * cur_actor_weight_2[i] + (1-self.soft_factor) * tar_actor_weight_2[i]
        critic_weight_dict['tar_actor_weight_2'] = temp_actor_weight_2
        
        
        temp_critic_weight_1 = cur_critic_weight_1
        for i in range(len(temp_critic_weight_1)):
            temp_critic_weight_1[i] = self.soft_factor * cur_critic_weight_1[i] + (1-self.soft_factor) * tar_critic_weight_1[i]
        critic_weight_dict['tar_critic_weight_1'] = temp_critic_weight_1
        
        temp_critic_weight_2 = cur_critic_weight_2
        for i in range(len(temp_critic_weight_2)):
            temp_critic_weight_2[i] = self.soft_factor * cur_critic_weight_2[i] + (1-self.soft_factor) * tar_critic_weight_2[i]
        critic_weight_dict['tar_critic_weight_2'] = temp_critic_weight_2
        
        critic_weight_dict['cur_critic_weight_1'] = self.critic_model.get_layer('cur_critic_dense_1').get_weights()
        critic_weight_dict['cur_critic_weight_2'] = self.critic_model.get_layer('cur_critic_dense_2').get_weights()

        critic_weight_dict['cur_actor_weight_1'] = self.critic_model.get_layer('cur_actor_dense_1').get_weights()
        critic_weight_dict['cur_actor_weight_2'] = self.critic_model.get_layer('cur_actor_dense_2').get_weights()
        
        return critic_weight_dict
    
    
    
    
    