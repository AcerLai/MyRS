# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:34:47 2019

@author: gzs13133
"""

from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers.advanced_activations import ReLU, Dice
from keras.initializers import RandomNormal
from keras.layers import Input, Activation, Reshape, Flatten, Embedding, Dense, \
Lambda, Multiply, Add, Concatenate, BatchNormalization
from base_structure import roc_callback, merge_sequence, DINattention, attention, InterestTransformerEncoder, \
InterestTransformerDecoder, IdentityLayer, LowTriLayer, simcos, fm_merge


class ccBaseModel(object):
    
    '''baseModel used in TV recommend.
    The base deepnetwork model in References, but modified the hyper parameter and the Dice.
    
    # Input
        one hist

    # Output
        The base model with attentional label combination and transformer.

    # Arguments
        ...
        
    # References
        - [Deep Interest Network for Click-Through Rate Prediction]
        (https://arxiv.org/abs/1706.06978)
    '''
    def __init__(self, user_num, item_num, hotrank_aid_num, hotrank_uid_num, max_hist_len, 
                 user_multi_cate_num_1, user_multi_cate_num_2, user_multi_cate_num_3, 
                 user_multi_cate_list_1, user_multi_cate_list_2, user_multi_cate_list_3,
                 item_multi_cate_num, item_multi_cate_list, 
                 user_multi_cate_len_1=None, user_multi_cate_len_2=None, user_multi_cate_len_3=None, 
                 item_multi_cate_len=None, hidden_num=16, use_Activa='ReLU', use_avgauc=False,
                 weighted=False, use_Transformer=False, use_weight=['softmax','softmax','softmax'], 
                 margin=0.05, w=1):
        
        rn = RandomNormal(mean=0, stddev=0.5)
        
        self.margin = margin
        self.w = w
        self.use_Transformer = use_Transformer
        self.use_avgauc = use_avgauc
        self.user = Input(shape=(1,), dtype='int32', name='user')
        self.item = Input(shape=(1,), dtype='int32', name='item')
        self.item_hotrank_by_aid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_aid')
        self.item_hotrank_by_uid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_uid')
        self.hist = Input(shape=(max_hist_len,),dtype='int32', name='hist') 
        if self.use_Transformer:
            self.auxiliary_index_ltr = LowTriLayer(max_hist_len)(self.user) 
            self.auxiliary_index_ltr = Lambda(lambda x:K.cast(x, dtype='int32'))(self.auxiliary_index_ltr)
            self.auxiliary_index_i = IdentityLayer(max_hist_len)(self.user) 
            self.auxiliary_index_i = Lambda(lambda x:K.cast(x, dtype='int32'))(self.auxiliary_index_i)
            self.pos = Input(shape=(max_hist_len,),dtype='int32', name='pos_gametype') 
            self.neg = Input(shape=(max_hist_len,),dtype='int32', name='neg_gametype') 
            self.last_sequence = Input(shape=(max_hist_len,),dtype='int32', name='last_sequence') 
        
#        user_Embedding = \
#            Embedding(output_dim=hidden_num // 2, input_dim=user_num+2, embeddings_initializer=rn, input_length=1)
#        it_hotrank_by_aid_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=hotrank_aid_num+2, embeddings_initializer=rn, input_length=1)
#        it_hotrank_by_uid_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=hotrank_uid_num+2, embeddings_initializer=rn, input_length=1)      
            
#        u = user_Embedding(self.user)
#        if mtype != 2:
#        it_hotrank_by_aid = it_hotrank_by_aid_Embedding(self.item_hotrank_by_aid)
#        it_hotrank_by_uid = it_hotrank_by_uid_Embedding(self.item_hotrank_by_uid)
        
        #item
#        item_Embedding = \
#            Embedding(output_dim=hidden_num // 2, input_dim=item_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=item_multi_cate_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate_lambda = \
            Lambda(lambda x:K.gather(item_multi_cate_list,  K.cast(x, dtype='int32')))
        item_multi_weight_Embedding = \
            Embedding(output_dim=item_multi_cate_len, input_dim=item_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate = item_multi_cate_lambda(self.item)
        item_multi_cate_embed = item_multi_cate_Embedding(item_multi_cate)
        
        it_multi_weight = item_multi_weight_Embedding(self.item)
        it_multi_all = \
            merge_sequence(item_multi_cate, item_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=it_multi_weight)
#        it = item_Embedding(self.item)
#        it = Concatenate(axis=-1)([it, it_multi_all])
        it = it_multi_all
            
        #bias
        ib_Embedding = \
            Embedding(output_dim=1, input_dim=item_num+2, embeddings_initializer=rn, input_length=1)
        i_b = Flatten()(ib_Embedding(self.item))
        
        # multi-hot process
        #user
        user_multi_cate_lambda_1 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_1,  K.cast(x, dtype='int32')))
        user_multi_cate_1 = user_multi_cate_lambda_1(self.user)
        
        user_multi_cate_lambda_2 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_2,  K.cast(x, dtype='int32')))
        user_multi_cate_2 = user_multi_cate_lambda_2(self.user)
        
        user_multi_cate_lambda_3 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_3,  K.cast(x, dtype='int32')))
        user_multi_cate_3 = user_multi_cate_lambda_3(self.user)
        
        user_multi_cate_Embedding_1 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_1+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_1 = user_multi_cate_Embedding_1(user_multi_cate_1)
        
        user_multi_cate_Embedding_2 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_2+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_2 = user_multi_cate_Embedding_2(user_multi_cate_2)
        
        user_multi_cate_Embedding_3 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_3+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_3 = user_multi_cate_Embedding_3(user_multi_cate_3)
     
        
        if self.use_Transformer:
            user_multi_cate_encoder_1 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_1, hidden_num=hidden_num, use_weight=use_weight[0])
            user_multi_encoder_1 = user_multi_cate_encoder_1([user_multi_cate_1, user_multi_cate_embed_1])
            
            user_multi_cate_encoder_2 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_2, hidden_num=hidden_num)
            user_multi_encoder_2 = user_multi_cate_encoder_2([user_multi_cate_2, user_multi_cate_embed_2])
            
            user_multi_cate_encoder_3 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_3, hidden_num=hidden_num)
            user_multi_encoder_3 = user_multi_cate_encoder_3([user_multi_cate_3, user_multi_cate_embed_3])

        else:
            user_multi_all_1 = \
                merge_sequence(user_multi_cate_1, user_multi_cate_embed_1, hidden_num, index=True)
            u = user_multi_all_1
 
            user_multi_all_2 = \
                merge_sequence(user_multi_cate_2, user_multi_cate_embed_2, hidden_num, index=True)   
            u = Concatenate(axis=-1)([u, user_multi_all_2])
                        
            user_multi_all_3 = \
                merge_sequence(user_multi_cate_3, user_multi_cate_embed_3, hidden_num, index=True)            
            u = Concatenate(axis=-1)([u, user_multi_all_3])

       
        hist_multi_cate = item_multi_cate_lambda(self.hist)
        hist_multi_cate_embed = item_multi_cate_Embedding(hist_multi_cate)
#        if use_selfatt:
#            hist_multi_cate_embed = self_attention_dot(hist_multi_cate, hist_multi_cate_embed, hidden_num, item_multi_cate_len, ishist=False, length=max_hist_len, use_weight=use_self_weight)
        hist_multi_weight = item_multi_weight_Embedding(self.hist)
        hist_multi_all = \
            merge_sequence(hist_multi_cate, hist_multi_cate_embed, hidden_num, index=False, item=True, weighted=weighted, weight=hist_multi_weight, l=max_hist_len)
        
        hist_multi_all = Reshape((-1, hidden_num))(hist_multi_all)
#        hi = item_Embedding(self.hist)
#        hi = Concatenate(axis=-1)([hi, hist_multi_all])
        hi = hist_multi_all
        h_all = merge_sequence(self.hist, hi, hidden_num, index_full=2)
              
        #batch-norm
        h_all = BatchNormalization()(h_all)
        
        if self.use_Transformer:
            hi_decoder_1 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_1, hidden_num=hidden_num, use_weight=use_weight[1], use_self_weight=use_weight[2])
            hi_decoder_2 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_2, hidden_num=hidden_num)
            hi_decoder_3 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_3, hidden_num=hidden_num)
            
            encoder2decoder_1 = hi_decoder_1([self.hist, self.last_sequence, user_multi_cate_1, hi, user_multi_encoder_1])
            encoder2decoder_2 = hi_decoder_2([self.hist, self.last_sequence, user_multi_cate_2, hi, user_multi_encoder_2])
            encoder2decoder_3 = hi_decoder_3([self.hist, self.last_sequence, user_multi_cate_3, hi, user_multi_encoder_3])
            
            u = encoder2decoder_1
            u = Concatenate(axis=-1)([u, encoder2decoder_2])
            u = Concatenate(axis=-1)([u, encoder2decoder_3])
            
            auxiliary_hist_index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(self.hist)
            last_sequence = Lambda(lambda x:K.cast(x, dtype='float32'))(self.last_sequence)
            auxiliary_hist_index = Lambda(lambda x:x[0]-x[1])([auxiliary_hist_index, last_sequence])
            
            auxiliary_hist_all = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-2), (1, max_hist_len, 1)))(self.hist)
            auxiliary_hist_all = Lambda(lambda x:x[0]*x[1])([auxiliary_hist_all, self.auxiliary_index_ltr])
            auxiliary_hist_multi_cate = item_multi_cate_lambda(auxiliary_hist_all)
            auxiliary_hist_multi_cate_embed = item_multi_cate_Embedding(auxiliary_hist_multi_cate)
    #        if use_selfatt:
    #            hist_multi_cate_embed = self_attention_dot(hist_multi_cate, hist_multi_cate_embed, hidden_num, item_multi_cate_len, ishist=False, length=max_hist_len, use_weight=use_self_weight)
            auxiliary_hist_multi_weight = item_multi_weight_Embedding(auxiliary_hist_all)
            auxiliary_hist_multi_all = \
                merge_sequence(auxiliary_hist_multi_cate, auxiliary_hist_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=auxiliary_hist_multi_weight, index_full=1, l=max_hist_len)

            auxiliary_hist_multi_all = Reshape((-1, max_hist_len, hidden_num))(auxiliary_hist_multi_all)
#            auxiliary_hi = item_Embedding(auxiliary_hist_all)
#            auxiliary_hi = Concatenate(axis=-1)([auxiliary_hi, auxiliary_hist_multi_all])
            auxiliary_hi = auxiliary_hist_multi_all
            auxiliary_hi = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, max_hist_len, hidden_num)))(auxiliary_hi)
                
            auxiliary_encoder2decoder_1 = hi_decoder_1([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_1, auxiliary_hi, user_multi_encoder_1])
            auxiliary_encoder2decoder_2 = hi_decoder_2([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_2, auxiliary_hi, user_multi_encoder_2])
            auxiliary_encoder2decoder_3 = hi_decoder_3([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_3, auxiliary_hi, user_multi_encoder_3])

            pos_multi_cate = item_multi_cate_lambda(self.pos)
            pos_multi_cate_embed = item_multi_cate_Embedding(pos_multi_cate)
            pos_multi_weight = item_multi_weight_Embedding(self.pos)
            pos_multi_all = \
                merge_sequence(pos_multi_cate, pos_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=pos_multi_weight, l=max_hist_len)
#            pos = item_Embedding(self.pos)
#            pos = Concatenate(axis=-1)([pos, pos_multi_all])
            pos = pos_multi_all
            pos = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, hidden_num)))(pos)
            
            neg_multi_cate = item_multi_cate_lambda(self.neg)
            neg_multi_cate_embed = item_multi_cate_Embedding(neg_multi_cate)
            neg_multi_weight = item_multi_weight_Embedding(self.neg)
            neg_multi_all = \
                merge_sequence(neg_multi_cate, neg_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=neg_multi_weight, l=max_hist_len)
#            neg = item_Embedding(self.neg)
#            neg = Concatenate(axis=-1)([neg, neg_multi_all])
            neg = neg_multi_all
            neg = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, hidden_num)))(neg)
                        
            #不用dot,因为dot产生的是类似协方差矩阵的相似度，而不是其对应元素的相似度。
            pos_cos_1 = simcos(auxiliary_encoder2decoder_1, pos)
            neg_cos_1 = simcos(auxiliary_encoder2decoder_1, neg)
            
            pos_cos_2 = simcos(auxiliary_encoder2decoder_2, pos)
            neg_cos_2 = simcos(auxiliary_encoder2decoder_2, neg)
            
            pos_cos_3 = simcos(auxiliary_encoder2decoder_3, pos)
            neg_cos_3 = simcos(auxiliary_encoder2decoder_3, neg)

            triplet_loss_1 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_1, pos_cos_1])
            triplet_loss_1 = Multiply()([triplet_loss_1, auxiliary_hist_index])
            triplet_loss_1 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_1)
            
            triplet_loss_2 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_2, pos_cos_2])
            triplet_loss_2 = Multiply()([triplet_loss_2, auxiliary_hist_index])
            triplet_loss_2 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_2)
            
            triplet_loss_3 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_3, pos_cos_3])
            triplet_loss_3 = Multiply()([triplet_loss_3, auxiliary_hist_index])
            triplet_loss_3 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_3)
            
            self.triplet_loss = Add()([triplet_loss_1, triplet_loss_2])
            self.triplet_loss = Add(name='triplet_loss')([self.triplet_loss, triplet_loss_3])
            
        #拼接user和hc,uid,aid作为新的user向量
        user_all = Concatenate(axis=-1)([u, h_all])
#        user_all = u
#        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_aid])
#        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_uid])

        dense_input = Flatten()(Concatenate(axis=-1)([user_all, it]))
        dense_input = BatchNormalization()(dense_input)
        
        #(128+)128(hist)+128(i) -> 80 -> 40 -> 1
        if use_Activa == 'Sigmoid':
            dense_layer_1_out = Dense(40, activation='sigmoid')(dense_input)
#            dense_layer_2_out = Dense(20, activation='sigmoid')(dense_layer_1_out)
        elif use_Activa == 'ReLU':
            dense_layer_1_out = Dense(10)(dense_input)
            dense_layer_1_out = ReLU()(dense_layer_1_out)
#            dense_layer_2_out = Dense(20)(dense_layer_1_out)
#            dense_layer_2_out = ReLU()(dense_layer_2_out)
        else:
            dense_layer_1_out = Dense(40)(dense_input)
            dense_layer_1_out = Dice()(dense_layer_1_out)
#            dense_layer_2_out = Dense(20)(dense_layer_1_out)
#            dense_layer_2_out = Dice()(dense_layer_2_out)
          
        dense_layer_3_out = Dense(1, activation='linear')(dense_layer_1_out)
        score = Add()([i_b ,dense_layer_3_out])
      
        self.prediction = Activation('sigmoid', name='prediction')(score)
        
        opt = optimizers.nadam()
        if use_Transformer:
#            , lambda y_true, y_pred: y_pred /, loss_weights=[1., self.w]/, self.triplet_loss
            self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, self.hist, self.last_sequence, 
                                       self.pos, self.neg], \
                               outputs=[self.prediction, self.triplet_loss, it, encoder2decoder_1, encoder2decoder_2, encoder2decoder_3])
            self.model.compile(opt, loss=['binary_crossentropy', lambda y_true, y_pred: y_pred] + [lambda y_true, y_pred: y_pred*0] * 4, loss_weights=[1., self.w] + [0] * 4,\
                               metrics={'prediction':'accuracy'})

        else:
            self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, self.hist], \
                               outputs=[self.prediction, it, user_multi_all_1, user_multi_all_2, user_multi_all_3])
            self.model.compile(opt, loss=['binary_crossentropy'] + [lambda y_true, y_pred: y_pred*0] * 4, loss_weights=[1.] + [0] * 4,\
                               metrics={'prediction':'accuracy'})
      
    #, callbacks = [roc_callback(training_data=[train_sample[0], train_sample[1]], validation_data=[eval_sample[0], eval_sample[1]])]
    def train_model(self, train_sample, eval_sample, batch_size, epoch=5, record_num=100):
        roc = roc_callback(training_data=[train_sample[0], train_sample[1]], \
                           validation_data=[eval_sample[0], eval_sample[1]], record_num=record_num, use_Transformer=self.use_Transformer, use_avgauc=self.use_avgauc)
        if self.use_Transformer:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3], train_sample[0][4], \
                            train_sample[0][5], train_sample[0][6], train_sample[0][7]], \
                            [train_sample[1]] + [train_sample[2]] * 5, batch_size, epochs=epoch, callbacks = [roc])
        else:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3], \
                            train_sample[0][4]], [train_sample[1]] + [train_sample[2]] * 4, batch_size, epochs=epoch, callbacks = [roc])
        return roc
    
    def eval_model(self, eval_sample):
        loss, acc = self.model.evaluate([eval_sample[0][0], eval_sample[0][1], eval_sample[0][2], eval_sample[0][3], \
                                         eval_sample[0][4]], eval_sample[1])
        return loss, acc      
    
    
class ccDINModelwithOneHist(object):
    
    '''DINModel used in TV recommend.
    The deepnetwork model in References, but modified the hyper parameter and the Dice.
    
    # Input
        one hist

    # Output
        The DIN model with attention and Dice.

    # Arguments
        use_Activa: the type of activation, it's optional in ['Sigmoid', 'PReLU', 'Dice'].
        
    # References
        - [Deep Interest Network for Click-Through Rate Prediction]
        (https://arxiv.org/abs/1706.06978)
    '''
    def __init__(self, user_num, item_num, hotrank_aid_num, hotrank_uid_num, max_hist_len, 
                 user_multi_cate_num_1, user_multi_cate_num_2, user_multi_cate_num_3, 
                 user_multi_cate_list_1, user_multi_cate_list_2, user_multi_cate_list_3,
                 item_multi_cate_num, item_multi_cate_list, 
                 user_multi_cate_len_1=None, user_multi_cate_len_2=None, user_multi_cate_len_3=None, 
                 item_multi_cate_len=None, hidden_num=16, use_Activa='ReLU', use_avgauc=False,
                 weighted=False, use_Transformer=False, use_weight=['softmax','softmax','softmax'], 
                 margin=0.05, w=1):
        
        rn = RandomNormal(mean=0, stddev=0.5)
        
        self.margin = margin
        self.w = w
        self.use_Transformer = use_Transformer
        self.use_avgauc = use_avgauc
        self.user = Input(shape=(1,), dtype='int32', name='user')
        self.item = Input(shape=(1,), dtype='int32', name='item')
        self.item_hotrank_by_aid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_aid')
        self.item_hotrank_by_uid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_uid')
        self.hist = Input(shape=(max_hist_len,),dtype='int32', name='hist') 
        if self.use_Transformer:
            self.auxiliary_index_ltr = LowTriLayer(max_hist_len)(self.user) 
            self.auxiliary_index_ltr = Lambda(lambda x:K.cast(x, dtype='int32'))(self.auxiliary_index_ltr)
            self.auxiliary_index_i = IdentityLayer(max_hist_len)(self.user) 
            self.auxiliary_index_i = Lambda(lambda x:K.cast(x, dtype='int32'))(self.auxiliary_index_i)
            self.pos = Input(shape=(max_hist_len,),dtype='int32', name='pos_gametype') 
            self.neg = Input(shape=(max_hist_len,),dtype='int32', name='neg_gametype') 
            self.last_sequence = Input(shape=(max_hist_len,),dtype='int32', name='last_sequence') 
        
        
#        user_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=user_num+2, embeddings_initializer=rn, input_length=1)
#        it_hotrank_by_aid_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=hotrank_aid_num+2, embeddings_initializer=rn, input_length=1)
#        it_hotrank_by_uid_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=hotrank_uid_num+2, embeddings_initializer=rn, input_length=1)

#        u = user_Embedding(self.user)
#        it_hotrank_by_aid = it_hotrank_by_aid_Embedding(self.item_hotrank_by_aid)
#        it_hotrank_by_uid = it_hotrank_by_uid_Embedding(self.item_hotrank_by_uid)
        
        #item
        item_multi_cate_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=item_multi_cate_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate_lambda = \
            Lambda(lambda x:K.gather(item_multi_cate_list,  K.cast(x, dtype='int32')))
        item_multi_weight_Embedding = \
            Embedding(output_dim=item_multi_cate_len, input_dim=item_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate = item_multi_cate_lambda(self.item)
        item_multi_cate_embed = item_multi_cate_Embedding(item_multi_cate)
        
        it_multi_weight = item_multi_weight_Embedding(self.item)
        it_multi_all = \
            merge_sequence(item_multi_cate, item_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=it_multi_weight)
         
        it = it_multi_all

        #bias
        ib_Embedding = \
            Embedding(output_dim=1, input_dim=item_num+2, embeddings_initializer=rn, input_length=1)
        i_b = Flatten()(ib_Embedding(self.item))
        
        # multi-hot process
        #user
        user_multi_cate_lambda_1 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_1,  K.cast(x, dtype='int32')))
        user_multi_cate_1 = user_multi_cate_lambda_1(self.user)
        
        user_multi_cate_lambda_2 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_2,  K.cast(x, dtype='int32')))
        user_multi_cate_2 = user_multi_cate_lambda_2(self.user)
        
        user_multi_cate_lambda_3 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_3,  K.cast(x, dtype='int32')))
        user_multi_cate_3 = user_multi_cate_lambda_3(self.user)

        user_multi_cate_Embedding_1 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_1+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_1 = user_multi_cate_Embedding_1(user_multi_cate_1)
        
        user_multi_cate_Embedding_2 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_2+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_2 = user_multi_cate_Embedding_2(user_multi_cate_2)
        
        user_multi_cate_Embedding_3 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_3+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_3 = user_multi_cate_Embedding_3(user_multi_cate_3)

        if self.use_Transformer:
            user_multi_cate_encoder_1 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_1, hidden_num=hidden_num, use_weight=use_weight[0])
            user_multi_encoder_1 = user_multi_cate_encoder_1([user_multi_cate_1, user_multi_cate_embed_1])
            
            user_multi_cate_encoder_2 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_2, hidden_num=hidden_num)
            user_multi_encoder_2 = user_multi_cate_encoder_2([user_multi_cate_2, user_multi_cate_embed_2])
            
            user_multi_cate_encoder_3 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_3, hidden_num=hidden_num)
            user_multi_encoder_3 = user_multi_cate_encoder_3([user_multi_cate_3, user_multi_cate_embed_3])


        else:
            user_multi_all_1 = \
                merge_sequence(user_multi_cate_1, user_multi_cate_embed_1, hidden_num, index=True)
            u = user_multi_all_1
 
            user_multi_all_2 = \
                merge_sequence(user_multi_cate_2, user_multi_cate_embed_2, hidden_num, index=True)   
            u = Concatenate(axis=-1)([u, user_multi_all_2])
                        
            user_multi_all_3 = \
                merge_sequence(user_multi_cate_3, user_multi_cate_embed_3, hidden_num, index=True)            
            u = Concatenate(axis=-1)([u, user_multi_all_3])

        hist_multi_cate = item_multi_cate_lambda(self.hist)
        hist_multi_cate_embed = item_multi_cate_Embedding(hist_multi_cate)
#        if use_selfatt:
#            hist_multi_cate_embed = self_attention_dot(hist_multi_cate, hist_multi_cate_embed, hidden_num, item_multi_cate_len, ishist=False, length=max_hist_len, use_weight=use_self_weight)
        
        hist_multi_weight = item_multi_weight_Embedding(self.hist)
        hist_multi_all = \
            merge_sequence(hist_multi_cate, hist_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=hist_multi_weight)
        
        hist_multi_all = Reshape((-1, hidden_num))(hist_multi_all)
        hi = hist_multi_all

        h_all = merge_sequence(self.hist, hi, hidden_num, index_full=2)

        h_all = attention(self.hist, it, hi, hidden_num, max_hist_len, use_weight='softmax')
        
        #batch-norm
        h_all = BatchNormalization()(h_all)
        
        if self.use_Transformer:
            hi_decoder_1 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_1, hidden_num=hidden_num, use_weight=use_weight[1], use_self_weight=use_weight[2])
            hi_decoder_2 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_2, hidden_num=hidden_num)
            hi_decoder_3 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_3, hidden_num=hidden_num)
        
            encoder2decoder_1 = hi_decoder_1([self.hist, self.last_sequence, user_multi_cate_1, hi, user_multi_encoder_1])
            encoder2decoder_2 = hi_decoder_2([self.hist, self.last_sequence, user_multi_cate_2, hi, user_multi_encoder_2])
            encoder2decoder_3 = hi_decoder_3([self.hist, self.last_sequence, user_multi_cate_3, hi, user_multi_encoder_3])
            
            u = encoder2decoder_1
            u = Concatenate(axis=-1)([u, encoder2decoder_2])
            u = Concatenate(axis=-1)([u, encoder2decoder_3])
            
            auxiliary_hist_index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(self.hist)
            last_sequence = Lambda(lambda x:K.cast(x, dtype='float32'))(self.last_sequence)
            auxiliary_hist_index = Lambda(lambda x:x[0]-x[1])([auxiliary_hist_index, last_sequence])
            
            auxiliary_hist_all = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-2), (1, max_hist_len, 1)))(self.hist)
            auxiliary_hist_all = Lambda(lambda x:x[0]*x[1])([auxiliary_hist_all, self.auxiliary_index_ltr])
            auxiliary_hist_multi_cate = item_multi_cate_lambda(auxiliary_hist_all)
            auxiliary_hist_multi_cate_embed = item_multi_cate_Embedding(auxiliary_hist_multi_cate)
    #        if use_selfatt:
    #            hist_multi_cate_embed = self_attention_dot(hist_multi_cate, hist_multi_cate_embed, hidden_num, item_multi_cate_len, ishist=False, length=max_hist_len, use_weight=use_self_weight)
            auxiliary_hist_multi_weight = item_multi_weight_Embedding(auxiliary_hist_all)
            auxiliary_hist_multi_all = \
                merge_sequence(auxiliary_hist_multi_cate, auxiliary_hist_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=auxiliary_hist_multi_weight, index_full=1, l=max_hist_len)

            auxiliary_hist_multi_all = Reshape((-1, max_hist_len, hidden_num))(auxiliary_hist_multi_all)
            auxiliary_hi = auxiliary_hist_multi_all
            auxiliary_hi = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, max_hist_len, hidden_num)))(auxiliary_hi)
                
            auxiliary_encoder2decoder_1 = hi_decoder_1([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_1, auxiliary_hi, user_multi_encoder_1])
            auxiliary_encoder2decoder_2 = hi_decoder_2([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_2, auxiliary_hi, user_multi_encoder_2])
            auxiliary_encoder2decoder_3 = hi_decoder_3([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_3, auxiliary_hi, user_multi_encoder_3])

            pos_multi_cate = item_multi_cate_lambda(self.pos)
            pos_multi_cate_embed = item_multi_cate_Embedding(pos_multi_cate)
            pos_multi_weight = item_multi_weight_Embedding(self.pos)
            pos_multi_all = \
                merge_sequence(pos_multi_cate, pos_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=pos_multi_weight, l=max_hist_len)
            pos = pos_multi_all
            pos = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, hidden_num)))(pos)
            
            neg_multi_cate = item_multi_cate_lambda(self.neg)
            neg_multi_cate_embed = item_multi_cate_Embedding(neg_multi_cate)
            neg_multi_weight = item_multi_weight_Embedding(self.neg)
            neg_multi_all = \
                merge_sequence(neg_multi_cate, neg_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=neg_multi_weight, l=max_hist_len)
            neg = neg_multi_all
            neg = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, hidden_num)))(neg)
                        
            #不用dot,因为dot产生的是类似协方差矩阵的相似度，而不是其对应元素的相似度。
            pos_cos_1 = simcos(auxiliary_encoder2decoder_1, pos)
            neg_cos_1 = simcos(auxiliary_encoder2decoder_1, neg)
            
            pos_cos_2 = simcos(auxiliary_encoder2decoder_2, pos)
            neg_cos_2 = simcos(auxiliary_encoder2decoder_2, neg)
            
            pos_cos_3 = simcos(auxiliary_encoder2decoder_3, pos)
            neg_cos_3 = simcos(auxiliary_encoder2decoder_3, neg)

            triplet_loss_1 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_1, pos_cos_1])
            triplet_loss_1 = Multiply()([triplet_loss_1, auxiliary_hist_index])
            triplet_loss_1 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_1)
            
            triplet_loss_2 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_2, pos_cos_2])
            triplet_loss_2 = Multiply()([triplet_loss_2, auxiliary_hist_index])
            triplet_loss_2 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_2)
            
            triplet_loss_3 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_3, pos_cos_3])
            triplet_loss_3 = Multiply()([triplet_loss_3, auxiliary_hist_index])
            triplet_loss_3 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_3)
            
            self.triplet_loss = Add()([triplet_loss_1, triplet_loss_2])
            self.triplet_loss = Add(name='triplet_loss')([self.triplet_loss, triplet_loss_3])
        
        #拼接user和hc,uid,aid作为新的user向量
        user_all = Concatenate(axis=-1)([u, h_all])
#        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_aid])
#        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_uid])

        dense_input = Flatten()(Concatenate(axis=-1)([user_all, it]))
        dense_input = BatchNormalization()(dense_input)
        
        #(128+)128(hist)+128(i) -> 80 -> 40 -> 1
        if use_Activa == 'Sigmoid':
            dense_layer_1_out = Dense(40, activation='sigmoid')(dense_input)
            dense_layer_2_out = Dense(20, activation='sigmoid')(dense_layer_1_out)
        elif use_Activa == 'ReLU':
            dense_layer_1_out = Dense(10)(dense_input)
            dense_layer_1_out = ReLU()(dense_layer_1_out)
        else:
            dense_layer_1_out = Dense(40)(dense_input)
            dense_layer_1_out = Dice()(dense_layer_1_out)
            dense_layer_2_out = Dense(20)(dense_layer_1_out)
            dense_layer_2_out = Dice()(dense_layer_2_out)
          
        dense_layer_3_out = Dense(1, activation='linear')(dense_layer_1_out)
        score = Add()([i_b ,dense_layer_3_out])
      
        self.prediction = Activation('sigmoid', name='prediction')(score)

        opt = optimizers.nadam()
        if use_Transformer:
#            , lambda y_true, y_pred: y_pred /, loss_weights=[1., self.w]/, self.triplet_loss
            self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, self.hist, self.last_sequence, 
                                       self.pos, self.neg], \
                               outputs=[self.prediction, self.triplet_loss, it, encoder2decoder_1, encoder2decoder_2, encoder2decoder_3])
            self.model.compile(opt, loss=['binary_crossentropy', lambda y_true, y_pred: y_pred] + [lambda y_true, y_pred: y_pred*0] * 4, loss_weights=[1., self.w] + [0] * 4,\
                               metrics={'prediction':'accuracy'})

        else:
            self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, self.hist], \
                               outputs=[self.prediction, it, user_multi_all_1, user_multi_all_2, user_multi_all_3])
            self.model.compile(opt, loss=['binary_crossentropy'] + [lambda y_true, y_pred: y_pred*0] * 4, loss_weights=[1.] + [0] * 4,\
                               metrics={'prediction':'accuracy'})
      
    #, callbacks = [roc_callback(training_data=[train_sample[0], train_sample[1]], validation_data=[eval_sample[0], eval_sample[1]])]
    def train_model(self, train_sample, eval_sample, batch_size, epoch=5, record_num=100):
        roc = roc_callback(training_data=[train_sample[0], train_sample[1]], \
                           validation_data=[eval_sample[0], eval_sample[1]], record_num=record_num, use_Transformer=self.use_Transformer, use_avgauc=self.use_avgauc)
        if self.use_Transformer:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3], train_sample[0][4], \
                            train_sample[0][5], train_sample[0][6], train_sample[0][7]], \
                            [train_sample[1]] + [train_sample[2]] * 5, batch_size, epochs=epoch, callbacks = [roc])
        else:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3], \
                            train_sample[0][4]], [train_sample[1]] + [train_sample[2]] * 4, batch_size, epochs=epoch, callbacks = [roc])
        return roc
    
    def eval_model(self, eval_sample):
        loss, acc = self.model.evaluate([eval_sample[0][0], eval_sample[0][1], eval_sample[0][2], eval_sample[0][3], \
                                         eval_sample[0][4]], eval_sample[1])
        return loss, acc      
   

class ccDINModelwithMultiHist(object):
    
    '''DINModel used in TV recommend.
    The deepnetwork model in References, but modified the hyper parameter and the Dice.
    
    # Input
        four hist

    # Output
        The DIN model with attention and Dice.

    # Arguments
        use_Activa: the type of activation, it's optional in ['Sigmoid', 'PReLU', 'Dice'].
        
    # References
        - [Deep Interest Network for Click-Through Rate Prediction]
        (https://arxiv.org/abs/1706.06978)
    '''
    def __init__(self, user_num, item_num, hotrank_aid_num, hotrank_uid_num, max_hist_len_1, 
                 max_hist_len_2, max_hist_len_3, max_hist_len_4,
                 hist_num_2, hist_num_3, hist_num_4, item_multi_cate_num, item_multi_cate_list, 
                 word2vec_list=None, hidden_num=16, use_Activa='Sigmoid', use_DINattention=True, num=1,
                 use_weight=None):
        
        rn = RandomNormal(mean=0, stddev=0.5)
        
        self.user = Input(shape=(1,), dtype='int32', name='user')
        self.item = Input(shape=(1,), dtype='int32', name='item')
        self.item_hotrank_by_aid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_aid')
        self.item_hotrank_by_uid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_uid')
        #与tensorflow一样，None代表未知长度序列，但必须保证各个batch下的长度一致
        self.hist_1 = Input(shape=(max_hist_len_1,),dtype='int32', name='hist_interest')
        self.hist_2 = Input(shape=(max_hist_len_2,),dtype='int32', name='hist_android')
        self.hist_3 = Input(shape=(max_hist_len_3,),dtype='int32', name='hist_ios')
        self.hist_4 = Input(shape=(max_hist_len_4,),dtype='int32', name='hist_mobile')

        
        user_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=user_num+1, embeddings_initializer=rn, input_length=1)
        it_Embedding =  \
            Embedding(output_dim=hidden_num // 2, input_dim=item_num+1, embeddings_initializer=rn, input_length=None)
        it_hotrank_by_aid_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=hotrank_aid_num+1, embeddings_initializer=rn, input_length=1)
        it_hotrank_by_uid_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=hotrank_uid_num+1, embeddings_initializer=rn, input_length=1)
        android_Embedding =  \
            Embedding(output_dim=hidden_num, input_dim=hist_num_2+1, embeddings_initializer=rn, input_length=None)
        ios_Embedding =  \
            Embedding(output_dim=hidden_num, input_dim=hist_num_3+1, embeddings_initializer=rn, input_length=None)
        mobile_Embedding =  \
            Embedding(output_dim=hidden_num, input_dim=hist_num_4+1, embeddings_initializer=rn, input_length=None)

        

        u = user_Embedding(self.user)
        it = it_Embedding(self.item)
        it_hotrank_by_aid = it_hotrank_by_aid_Embedding(self.item_hotrank_by_aid)
        it_hotrank_by_uid = it_hotrank_by_uid_Embedding(self.item_hotrank_by_uid)

        #item
        item_multi_cate_Embedding = \
            Embedding(output_dim=hidden_num // 2, input_dim=item_multi_cate_num+1, embeddings_initializer=rn, input_length=None)
        item_multi_cate_lambda = \
            Lambda(lambda x:K.gather(item_multi_cate_list,  K.cast(x, dtype='int32')))
        item_multi_cate = item_multi_cate_lambda(self.item)
        item_multi_cate_embed = item_multi_cate_Embedding(item_multi_cate)
        it_multi_all = \
            merge_sequence(item_multi_cate, item_multi_cate_embed, hidden_num // 2, index=True)
        
        it = Concatenate(axis=-1)([it, it_multi_all])
        
        same = False
        #item_nlp
        if word2vec_list is not None:
            item_word2vec_lambda = \
                Lambda(lambda x:K.gather(word2vec_list,  K.cast(x, dtype='int32')))
            item_word2vec = item_word2vec_lambda(self.item)
            item_word2vec = Reshape((1, -1))(item_word2vec)
            item_word2vec_dense = Dense(hidden_num*num, activation='sigmoid')
            item_word2vec_all = item_word2vec_dense(item_word2vec)
            
            it = Concatenate(axis=-1)([it, item_word2vec_all])

          
        #bias
        ib_Embedding = \
            Embedding(output_dim=1, input_dim=item_num+1, embeddings_initializer=rn, input_length=1)
        i_b = Flatten()(ib_Embedding(self.item))
        
        #四个hist
        ## interest_gametype
        hi_1 = it_Embedding(self.hist_1)

        hist_multi_cate = item_multi_cate_lambda(self.hist_1)
        hist_multi_cate_embed = item_multi_cate_Embedding(hist_multi_cate)
        hist_multi_all = \
            merge_sequence(hist_multi_cate, hist_multi_cate_embed, hidden_num // 2, index=True)
        
        hi_1 = Concatenate(axis=-1)([hi_1, hist_multi_all])
        
        if word2vec_list is not None:
            hist_word2vec = item_word2vec_lambda(self.hist_1)
            hist_word2vec = Reshape((max_hist_len_1, -1))(hist_word2vec)
            hist_word2vec_all = item_word2vec_dense(hist_word2vec)
            
            hi_1 = Concatenate(axis=-1)([hi_1, hist_word2vec_all])
#            if use_selfatt:
#                hi_1 = self_attention_dot(self.hist_1, hi_1, hidden_num*3, max_hist_len_1, length=max_hist_len_1, expand=False, ishist=True)
            #attention
            if use_DINattention:
                h_all_1 = DINattention(self.hist_1, it, hi_1, hidden_num*(2+num), max_hist_len_1, use_Activa)
            else:
                h_all_1 = attention(self.hist_1, it, hi_1, hidden_num*(2+num), max_hist_len_1, use_weight)
        else:
#            if use_selfatt:
#                hi_1 = self_attention_dot(self.hist_1, hi_1, hidden_num*2, max_hist_len_1, length=max_hist_len_1, expand=False, ishist=True)
            if use_DINattention:
                h_all_1 = DINattention(self.hist_1, it, hi_1, hidden_num, max_hist_len_1, use_Activa)
            else:
                h_all_1 = attention(self.hist_1, it, hi_1, hidden_num, max_hist_len_1, use_weight)
        
        #batch-norm
        h_all_1 = BatchNormalization()(h_all_1)
        
        ##android, ios, mobile
        hi_2 = android_Embedding(self.hist_2)
#        if use_selfatt:
#            hi_2 = self_attention_dot(self.hist_2, hi_2, hidden_num, max_hist_len_2, length=max_hist_len_2, expand=False, ishist=True)
        if use_DINattention:
            h_all_2 = DINattention(self.hist_2, it, hi_2, hidden_num, max_hist_len_2, use_Activa, same)
        else:
            h_all_2 = attention(self.hist_2, it, hi_2, hidden_num, max_hist_len_2, use_weight)
        h_all_2 = BatchNormalization()(h_all_2)
        
        hi_3 = ios_Embedding(self.hist_3)
#        if use_selfatt:
#            hi_3 = self_attention_dot(self.hist_3, hi_3, hidden_num, max_hist_len_3, length=max_hist_len_3, expand=False, ishist=True)
        if use_DINattention:
            h_all_3 = DINattention(self.hist_3, it, hi_3, hidden_num, max_hist_len_3, use_Activa, same)
        else:
            h_all_3 = attention(self.hist_3, it, hi_3, hidden_num, max_hist_len_3, use_weight)
        h_all_3 = BatchNormalization()(h_all_3)
        
        hi_4 = mobile_Embedding(self.hist_4)
#        if use_selfatt:
#            hi_4 = self_attention_dot(self.hist_4, hi_4, hidden_num, max_hist_len_4, length=max_hist_len_4, expand=False, ishist=True)
#        if use_DINattention:
#            h_all_4 = DINattention(self.hist_4, it, hi_4, hidden_num, max_hist_len_4, use_Activa, same)
#        else:
#            h_all_4 = attention(self.hist_4, it, fix(hi_4), hidden_num*3, max_hist_len_4, use_weight)
        h_all_4 = BatchNormalization()(hi_4)
        
        #拼接user和四个hc,uid,aid作为新的user向量
        user_all = Concatenate(axis=-1)([u, h_all_1])
        user_all = Concatenate(axis=-1)([user_all, h_all_2])
        user_all = Concatenate(axis=-1)([user_all, h_all_3])
        user_all = Concatenate(axis=-1)([user_all, h_all_4])
        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_aid])
        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_uid])

        
        dense_input = Flatten()(Concatenate(axis=-1)([user_all, it]))
        dense_input = BatchNormalization()(dense_input)
        
        #(128+)128(hist)+128(i) -> 80 -> 40 -> 1
        if use_Activa == 'Sigmoid':
            dense_layer_1_out = Dense(40, activation='sigmoid')(dense_input)
            dense_layer_2_out = Dense(20, activation='sigmoid')(dense_layer_1_out)
        elif use_Activa == 'ReLU':
            dense_layer_1_out = Dense(40)(dense_input)
            dense_layer_1_out = ReLU()(dense_layer_1_out)
            dense_layer_2_out = Dense(20)(dense_layer_1_out)
            dense_layer_2_out = ReLU()(dense_layer_2_out)
        elif use_Activa == 'Dice':
            dense_layer_1_out = Dense(40)(dense_input)
            dense_layer_1_out = Dice()(dense_layer_1_out)
            dense_layer_2_out = Dense(20)(dense_layer_1_out)
            dense_layer_2_out = Dice()(dense_layer_2_out)
        else:
            dense_layer_1_out = Dense(40, activation='tanh')(dense_input)
            dense_layer_2_out = Dense(20, activation='tanh')(dense_layer_1_out)
            
        dense_layer_3_out = Dense(1, activation='linear')(dense_layer_2_out)
        score = Add()([i_b ,dense_layer_3_out])
      
        self.prediction = Activation('sigmoid')(score)
        
        self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, \
                                   self.hist_1, self.hist_2, self.hist_3, self.hist_4], outputs=[self.prediction])
#        lr=0.0001, clipnorm=0.001
        opt = optimizers.rmsprop()
        self.model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
      
    #, callbacks = [roc_callback(training_data=[train_sample[0], train_sample[1]], validation_data=[eval_sample[0], eval_sample[1]])]
    def train_model(self, train_sample, eval_sample, batch_size, epoch=5, record_num=100, sample_weight=None):
        roc = roc_callback(training_data=[train_sample[0], train_sample[1]], \
                           validation_data=[eval_sample[0], eval_sample[1]], record_num=record_num)
        if sample_weight is not None:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3],\
                            train_sample[0][4],train_sample[0][5],train_sample[0][6],train_sample[0][7]], \
                            train_sample[1], batch_size, epochs=epoch, callbacks = [roc], sample_weight=sample_weight)
        else:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3],\
                            train_sample[0][4],train_sample[0][5],train_sample[0][6],train_sample[0][7]], \
                            train_sample[1], batch_size, epochs=epoch, callbacks = [roc])
        return roc
    
    def eval_model(self, eval_sample):
        loss, acc = self.model.evaluate([eval_sample[0][0], eval_sample[0][1], eval_sample[0][2], \
                                         eval_sample[0][3], eval_sample[0][4], eval_sample[0][5], \
                                         eval_sample[0][6], eval_sample[0][7]], eval_sample[1])
        return loss, acc      
   
    
class ccdeepFM(object):
    
    def __init__(self, user_num, item_num, hotrank_aid_num, hotrank_uid_num, max_hist_len, 
                 user_multi_cate_num_1, user_multi_cate_num_2, user_multi_cate_num_3, 
                 user_multi_cate_list_1, user_multi_cate_list_2, user_multi_cate_list_3,
                 item_multi_cate_num, item_multi_cate_list, 
                 user_multi_cate_len_1, user_multi_cate_len_2, user_multi_cate_len_3, 
                 item_multi_cate_len, hidden_num=16, use_Activa='ReLU', weighted=False, use_avgauc=False,
                 use_Transformer=False, use_weight=['softmax','softmax','softmax'], 
                 margin=0.05, w=1):
        
        rn = RandomNormal(mean=0, stddev=0.5)
        
        self.margin = margin
        self.w = w
        self.use_Transformer = use_Transformer
        self.use_avgauc = use_avgauc
        self.user = Input(shape=(1,), dtype='int32', name='user')
        self.item = Input(shape=(1,), dtype='int32', name='item')
        self.item_hotrank_by_aid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_aid')
        self.item_hotrank_by_uid = Input(shape=(1,), dtype='int32', name='item_hotrank_by_uid')
        self.hist = Input(shape=(max_hist_len,),dtype='int32', name='hist') 
        if self.use_Transformer:
            self.auxiliary_index_ltr = LowTriLayer(max_hist_len)(self.user) 
            self.auxiliary_index_ltr = Lambda(lambda x:K.cast(x, dtype='int32'))(self.auxiliary_index_ltr)
            self.auxiliary_index_i = IdentityLayer(max_hist_len)(self.user) 
            self.auxiliary_index_i = Lambda(lambda x:K.cast(x, dtype='int32'))(self.auxiliary_index_i)
            self.pos = Input(shape=(max_hist_len,),dtype='int32', name='pos_gametype') 
            self.neg = Input(shape=(max_hist_len,),dtype='int32', name='neg_gametype') 
            self.last_sequence = Input(shape=(max_hist_len,),dtype='int32', name='last_sequence') 
        
#        it_hotrank_by_aid_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=hotrank_aid_num+2, embeddings_initializer=rn, input_length=1)
#        it_hotrank_by_uid_Embedding = \
#            Embedding(output_dim=hidden_num, input_dim=hotrank_uid_num+2, embeddings_initializer=rn, input_length=1)      
#        it_hotrank_by_aid_bias = \
#            Embedding(output_dim=1, input_dim=hotrank_aid_num+2, embeddings_initializer=rn, input_length=1)
#        it_hotrank_by_uid_bias = \
#            Embedding(output_dim=1, input_dim=hotrank_uid_num+2, embeddings_initializer=rn, input_length=1)    
    
#        it_hotrank_by_aid = it_hotrank_by_aid_Embedding(self.item_hotrank_by_aid)
#        it_hotrank_by_uid = it_hotrank_by_uid_Embedding(self.item_hotrank_by_uid)
#        it_hotrank_by_aid_b = it_hotrank_by_aid_bias(self.item_hotrank_by_aid)
#        it_hotrank_by_uid_b = it_hotrank_by_uid_bias(self.item_hotrank_by_uid)

        #item
        item_bias = \
            Embedding(output_dim=1, input_dim=item_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate_Embedding = \
            Embedding(output_dim=hidden_num, input_dim=item_multi_cate_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate_lambda = \
            Lambda(lambda x:K.gather(item_multi_cate_list,  K.cast(x, dtype='int32')))
        item_multi_weight_Embedding = \
            Embedding(output_dim=item_multi_cate_len, input_dim=item_num+2, embeddings_initializer=rn, input_length=None)
        item_multi_cate = item_multi_cate_lambda(self.item)
        item_multi_cate_embed = item_multi_cate_Embedding(item_multi_cate)
        
        it_multi_weight = item_multi_weight_Embedding(self.item)
        it_multi_all = \
            merge_sequence(item_multi_cate, item_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=it_multi_weight)
         
        it = it_multi_all
        
        ib_Embedding = \
            Embedding(output_dim=1, input_dim=item_num+2, embeddings_initializer=rn, input_length=1)
        i_b = Flatten()(ib_Embedding(self.item))
        
        # multi-hot process
        #user
        user_multi_cate_lambda_1 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_1,  K.cast(x, dtype='int32')))
        user_multi_cate_1 = user_multi_cate_lambda_1(self.user)
        
        user_multi_cate_lambda_2 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_2,  K.cast(x, dtype='int32')))
        user_multi_cate_2 = user_multi_cate_lambda_2(self.user)
        
        user_multi_cate_lambda_3 = \
            Lambda(lambda x:K.gather(user_multi_cate_list_3,  K.cast(x, dtype='int32')))
        user_multi_cate_3 = user_multi_cate_lambda_3(self.user)
        
        user_multi_cate_Embedding_1 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_1+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_1 = user_multi_cate_Embedding_1(user_multi_cate_1)
        
        user_multi_cate_Embedding_2 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_2+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_2 = user_multi_cate_Embedding_2(user_multi_cate_2)
        
        user_multi_cate_Embedding_3 = \
            Embedding(output_dim=hidden_num, input_dim=user_multi_cate_num_3+2, embeddings_initializer=rn, input_length=None)
        user_multi_cate_embed_3 = user_multi_cate_Embedding_3(user_multi_cate_3)
        
        user_multi_cate_bias_1 = \
            Embedding(output_dim=1, input_dim=user_multi_cate_num_1+2, embeddings_initializer=rn, input_length=None)
    
        user_multi_cate_bias_2 = \
            Embedding(output_dim=1, input_dim=user_multi_cate_num_2+2, embeddings_initializer=rn, input_length=None)        
        
        user_multi_cate_bias_3 = \
            Embedding(output_dim=1, input_dim=user_multi_cate_num_3+2, embeddings_initializer=rn, input_length=None)


        user_multi_cate_index_1 = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(user_multi_cate_1)
        user_multi_cate_b_1 = Reshape((-1, 1, user_multi_cate_len_1))(user_multi_cate_bias_1(user_multi_cate_1))
        user_multi_cate_b_1 = Multiply()([user_multi_cate_b_1, user_multi_cate_index_1])
        user_multi_cate_sum_bias_1 = Lambda(lambda x:K.sum(K.reshape(x, (-1, 1, user_multi_cate_len_1)), axis=-1))
        user_multi_cate_b_1 = user_multi_cate_sum_bias_1(user_multi_cate_b_1)

        user_multi_cate_index_2 = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(user_multi_cate_2)
        user_multi_cate_b_2 = Reshape((-1, 1, user_multi_cate_len_2))(user_multi_cate_bias_2(user_multi_cate_2))
        user_multi_cate_b_2 = Multiply()([user_multi_cate_b_2, user_multi_cate_index_2])
        user_multi_cate_sum_bias_2 = Lambda(lambda x:K.sum(K.reshape(x, (-1, 1, user_multi_cate_len_2)), axis=-1))
        user_multi_cate_b_2 = user_multi_cate_sum_bias_2(user_multi_cate_b_2)

        user_multi_cate_index_3 = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(user_multi_cate_3)
        user_multi_cate_b_3 = Reshape((-1, 1, user_multi_cate_len_3))(user_multi_cate_bias_3(user_multi_cate_3))
        user_multi_cate_b_3 = Multiply()([user_multi_cate_b_3, user_multi_cate_index_3])
        user_multi_cate_sum_bias_3 = Lambda(lambda x:K.sum(K.reshape(x, (-1, 1, user_multi_cate_len_3)), axis=-1))
        user_multi_cate_b_3 = user_multi_cate_sum_bias_3(user_multi_cate_b_3)


        if self.use_Transformer:
            
            user_multi_cate_encoder_1 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_1, hidden_num=hidden_num, use_weight=use_weight[0])
            user_multi_encoder_1 = user_multi_cate_encoder_1([user_multi_cate_1, user_multi_cate_embed_1])
            
            user_multi_cate_encoder_2 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_2, hidden_num=hidden_num)
            user_multi_encoder_2 = user_multi_cate_encoder_2([user_multi_cate_2, user_multi_cate_embed_2])
            
            user_multi_cate_encoder_3 = InterestTransformerEncoder(initializer=rn, max_hist_len=user_multi_cate_len_3, hidden_num=hidden_num)
            user_multi_encoder_3 = user_multi_cate_encoder_3([user_multi_cate_3, user_multi_cate_embed_3])
        
        else:
            u1 = \
                merge_sequence(user_multi_cate_1, user_multi_cate_embed_1, hidden_num, index=True)
            u = u1
            u2 = \
                merge_sequence(user_multi_cate_2, user_multi_cate_embed_2, hidden_num, index=True)   
            u = Concatenate(axis=-1)([u, u2])                     
            u3 = \
                merge_sequence(user_multi_cate_3, user_multi_cate_embed_3, hidden_num, index=True)            
            u = Concatenate(axis=-1)([u, u3])

        
        hist_multi_cate = item_multi_cate_lambda(self.hist)
        hist_multi_cate_embed = item_multi_cate_Embedding(hist_multi_cate)
#        if use_selfatt:
#            hist_multi_cate_embed = self_attention_dot(hist_multi_cate, hist_multi_cate_embed, hidden_num, item_multi_cate_len, ishist=False, length=max_hist_len, use_weight=use_self_weight)
        hist_multi_weight = item_multi_weight_Embedding(self.hist)
        hist_multi_all = \
            merge_sequence(hist_multi_cate, hist_multi_cate_embed, hidden_num, index=False, item=True, weighted=weighted, weight=hist_multi_weight, l=max_hist_len)
        
        hist_multi_all = Reshape((-1, hidden_num))(hist_multi_all)

        hi_b = Reshape((-1, max_hist_len))(item_bias(self.hist))

        hi_index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(self.hist)
        hi_b = Multiply()([hi_b, hi_index])
        hi_sum_bias = Lambda(lambda x:K.sum(K.reshape(x, (-1, 1, max_hist_len)), axis=-1))
        hi_b = hi_sum_bias(hi_b)


        hi = hist_multi_all
        h_all = merge_sequence(self.hist, hi, hidden_num, index_full=2)
        h_all = BatchNormalization()(h_all)
        
#        multi_feature = [[user_multi_cate_1, user_multi_cate_embed_1, user_multi_cate_len_1], \
#                         [self.hist, hi, max_hist_len]]
        if self.use_Transformer:
            hi_decoder_1 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_1, hidden_num=hidden_num, use_weight=use_weight[1], use_self_weight=use_weight[2])
            hi_decoder_2 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_2, hidden_num=hidden_num)
            hi_decoder_3 = InterestTransformerDecoder(initializer=rn, max_hist_len=max_hist_len, max_hist_len_1=user_multi_cate_len_3, hidden_num=hidden_num)
            
            encoder2decoder_1 = hi_decoder_1([self.hist, self.last_sequence, user_multi_cate_1, hi, user_multi_encoder_1])
            encoder2decoder_2 = hi_decoder_2([self.hist, self.last_sequence, user_multi_cate_2, hi, user_multi_encoder_2])
            encoder2decoder_3 = hi_decoder_3([self.hist, self.last_sequence, user_multi_cate_3, hi, user_multi_encoder_3])
            
            u = u1 = encoder2decoder_1
            u2 = encoder2decoder_2
            u = Concatenate(axis=-1)([u, encoder2decoder_2])
            u3 = encoder2decoder_3
            u = Concatenate(axis=-1)([u, encoder2decoder_3])
            
            auxiliary_hist_index = Lambda(lambda x:K.cast(K.cast(x, dtype='bool'), dtype='float32'))(self.hist)
            last_sequence = Lambda(lambda x:K.cast(x, dtype='float32'))(self.last_sequence)
            auxiliary_hist_index = Lambda(lambda x:x[0]-x[1])([auxiliary_hist_index, last_sequence])
            
            auxiliary_hist_all = Lambda(lambda x:K.tile(K.expand_dims(x, axis=-2), (1, max_hist_len, 1)))(self.hist)
            auxiliary_hist_all = Lambda(lambda x:x[0]*x[1])([auxiliary_hist_all, self.auxiliary_index_ltr])
            auxiliary_hist_multi_cate = item_multi_cate_lambda(auxiliary_hist_all)
            auxiliary_hist_multi_cate_embed = item_multi_cate_Embedding(auxiliary_hist_multi_cate)
    #        if use_selfatt:
    #            hist_multi_cate_embed = self_attention_dot(hist_multi_cate, hist_multi_cate_embed, hidden_num, item_multi_cate_len, ishist=False, length=max_hist_len, use_weight=use_self_weight)
            auxiliary_hist_multi_weight = item_multi_weight_Embedding(auxiliary_hist_all)
            auxiliary_hist_multi_all = \
                merge_sequence(auxiliary_hist_multi_cate, auxiliary_hist_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=auxiliary_hist_multi_weight, index_full=1, l=max_hist_len)

            auxiliary_hist_multi_all = Reshape((-1, max_hist_len, hidden_num))(auxiliary_hist_multi_all)
            auxiliary_hi = auxiliary_hist_multi_all
            auxiliary_hi = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, max_hist_len, hidden_num)))(auxiliary_hi)
                
            auxiliary_encoder2decoder_1 = hi_decoder_1([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_1, auxiliary_hi, user_multi_encoder_1])
            auxiliary_encoder2decoder_2 = hi_decoder_2([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_2, auxiliary_hi, user_multi_encoder_2])
            auxiliary_encoder2decoder_3 = hi_decoder_3([auxiliary_hist_all, self.auxiliary_index_i, user_multi_cate_3, auxiliary_hi, user_multi_encoder_3])

            pos_multi_cate = item_multi_cate_lambda(self.pos)
            pos_multi_cate_embed = item_multi_cate_Embedding(pos_multi_cate)
            pos_multi_weight = item_multi_weight_Embedding(self.pos)
            pos_multi_all = \
                merge_sequence(pos_multi_cate, pos_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=pos_multi_weight, l=max_hist_len)
            pos = pos_multi_all
            pos = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, hidden_num)))(pos)
            
            neg_multi_cate = item_multi_cate_lambda(self.neg)
            neg_multi_cate_embed = item_multi_cate_Embedding(neg_multi_cate)
            neg_multi_weight = item_multi_weight_Embedding(self.neg)
            neg_multi_all = \
                merge_sequence(neg_multi_cate, neg_multi_cate_embed, hidden_num, index=True, item=True, weighted=weighted, weight=neg_multi_weight, l=max_hist_len)
            neg = neg_multi_all
            neg = Lambda(lambda x:K.reshape(x, (-1, max_hist_len, hidden_num)))(neg)
                        
            #不用dot,因为dot产生的是类似协方差矩阵的相似度，而不是其对应元素的相似度。
            pos_cos_1 = simcos(auxiliary_encoder2decoder_1, pos)
            neg_cos_1 = simcos(auxiliary_encoder2decoder_1, neg)
            
            pos_cos_2 = simcos(auxiliary_encoder2decoder_2, pos)
            neg_cos_2 = simcos(auxiliary_encoder2decoder_2, neg)
            
            pos_cos_3 = simcos(auxiliary_encoder2decoder_3, pos)
            neg_cos_3 = simcos(auxiliary_encoder2decoder_3, neg)

            triplet_loss_1 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_1, pos_cos_1])
            triplet_loss_1 = Multiply()([triplet_loss_1, auxiliary_hist_index])
            triplet_loss_1 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_1)
            
            triplet_loss_2 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_2, pos_cos_2])
            triplet_loss_2 = Multiply()([triplet_loss_2, auxiliary_hist_index])
            triplet_loss_2 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_2)
            
            triplet_loss_3 = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([neg_cos_3, pos_cos_3])
            triplet_loss_3 = Multiply()([triplet_loss_3, auxiliary_hist_index])
            triplet_loss_3 = Lambda(lambda x:K.reshape(K.sum(x, axis=-1),(-1, 1)))(triplet_loss_3)
            
            self.triplet_loss = Add()([triplet_loss_1, triplet_loss_2])
            self.triplet_loss = Add(name='triplet_loss')([self.triplet_loss, triplet_loss_3])
        
        one_feature = [it, h_all, u1, u2, u3]
#        one_feature = [it, it_hotrank_by_aid, it_hotrank_by_uid, h_all, u1, u2, u3]
        
        fm_score = fm_merge(one_feature, hidden_num)
                            
        #拼接user和hc,uid,aid作为新的user向量
        user_all = Concatenate(axis=-1)([u, h_all])
#        user_all = u
#        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_aid])
#        user_all = Concatenate(axis=-1)([user_all, it_hotrank_by_uid])

        dense_input = Flatten()(Concatenate(axis=-1)([user_all, it]))
        dense_input = BatchNormalization()(dense_input)
        #(128+)128(hist)+128(i) -> 80 -> 40 -> 1
        if use_Activa == 'Sigmoid':
            dense_layer_1_out = Dense(40, activation='sigmoid')(dense_input)
#            dense_layer_2_out = Dense(20, activation='sigmoid')(dense_layer_1_out)
        elif use_Activa == 'ReLU':
            dense_layer_1_out = Dense(10)(dense_input)
            dense_layer_1_out = ReLU()(dense_layer_1_out)
#            dense_layer_2_out = Dense(20)(dense_layer_1_out)
#            dense_layer_2_out = ReLU()(dense_layer_2_out)
        else:
            dense_layer_1_out = Dense(40)(dense_input)
            dense_layer_1_out = Dice()(dense_layer_1_out)
#            dense_layer_2_out = Dense(20)(dense_layer_1_out)
#            dense_layer_2_out = Dice()(dense_layer_2_out)
          
        dense_layer_3_out = Dense(1, activation='linear')(dense_layer_1_out)
#        score = Add()([it_hotrank_by_aid_b ,dense_layer_3_out])
#        score = Add()([it_hotrank_by_uid_b ,score])
        score = Add()([i_b ,dense_layer_3_out])
        score = Add()([user_multi_cate_b_1 ,score])
        score = Add()([user_multi_cate_b_2 ,score])
        score = Add()([user_multi_cate_b_3 ,score])
        score = Add()([hi_b ,score])
        score = Add()([fm_score ,score])
        score = Lambda(lambda x:K.reshape(x, (-1, 1)))(score)
      
        self.prediction = Activation('sigmoid', name='prediction')(score)
        
        opt = optimizers.nadam()
        if use_Transformer:
#            , lambda y_true, y_pred: y_pred /, loss_weights=[1., self.w]/, self.triplet_loss
            self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, self.hist, self.last_sequence, 
                                       self.pos, self.neg], \
                               outputs=[self.prediction, self.triplet_loss, it, encoder2decoder_1, encoder2decoder_2, encoder2decoder_3])
            self.model.compile(opt, loss=['binary_crossentropy', lambda y_true, y_pred: y_pred] + [lambda y_true, y_pred: y_pred*0] * 4, loss_weights=[1., self.w] + [0] * 4,\
                               metrics={'prediction':'accuracy'})

        else:
            self.model = Model(inputs=[self.user, self.item, self.item_hotrank_by_aid, self.item_hotrank_by_uid, self.hist], \
                               outputs=[self.prediction, it, u1, u2, u3])
            self.model.compile(opt, loss=['binary_crossentropy'] + [lambda y_true, y_pred: y_pred*0] * 4, loss_weights=[1.] + [0] * 4,\
                               metrics={'prediction':'accuracy'})
      
    #, callbacks = [roc_callback(training_data=[train_sample[0], train_sample[1]], validation_data=[eval_sample[0], eval_sample[1]])]
    def train_model(self, train_sample, eval_sample, batch_size, epoch=5, record_num=100):
        roc = roc_callback(training_data=[train_sample[0], train_sample[1]], \
                           validation_data=[eval_sample[0], eval_sample[1]], record_num=record_num, use_Transformer=self.use_Transformer, use_avgauc=self.use_avgauc)
        if self.use_Transformer:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3], train_sample[0][4], \
                            train_sample[0][5], train_sample[0][6], train_sample[0][7]], \
                            [train_sample[1]] + [train_sample[2]] * 5, batch_size, epochs=epoch, callbacks = [roc])
        else:
            self.model.fit([train_sample[0][0],train_sample[0][1],train_sample[0][2],train_sample[0][3], \
                            train_sample[0][4]], [train_sample[1]] + [train_sample[2]] * 4, batch_size, epochs=epoch, callbacks = [roc])
        return roc
    
    def eval_model(self, eval_sample):
        loss, acc = self.model.evaluate([eval_sample[0][0], eval_sample[0][1], eval_sample[0][2], eval_sample[0][3], \
                                         eval_sample[0][4]], eval_sample[1])
        return loss, acc      