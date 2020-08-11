# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:39:39 2019

@author: gzs13133
"""

import tensorflow as tf
import numpy as np
from data_process import emoj_sub, url_sub

class biLSTMwithCRF(object):
    
    def __init__(self, sequence_length=20, hidden_num=32, learning_rate=0.001,
                 optimizer='Adam', batch_size=16, epoch=10, ckpt_dir='checkpoint/'):
        
        self.sequence_length = sequence_length
        self.hidden_num = hidden_num
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = epoch
        self.ckpt_dir = ckpt_dir
        self.vocab = 'vocab'
        self.save_path = self.ckpt_dir + "model.ckpt"

    def load_data_one(self):
        
        char_list = []
        with open('data', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('\n', '').replace('null', '[]')
                line = eval(line)
                for i in line['text']:
                    if i not in char_list:
                        char_list.append(i)
                
        char_list = ['UNK'] + char_list        
        with open(self.vocab, 'w', encoding='utf8') as f:
            for i in char_list:
                f.write(i + '\n')
        
    def load_data(self):
        
        all_list = []
        anchor_list = []
        with open('data', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('\n', '').replace('null', '[]')
                line = eval(line)
                label = line['annotations'][0]['label']
                if label == 'all':
                    all_list.append(line['text'])
                else:
                    s = int(line['annotations'][0]['start_offset'])
                    e = int(line['annotations'][0]['end_offset'])
                    anchor_list.append([line['text'], s, e])
                    
        char_list = []
        with open(self.vocab, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace('\n', '')
                char_list.append(line)
                
        self.UNK_ID = 0
        self.word2index={}
        for i, char in enumerate(char_list):
            self.word2index[char]=i
        self.vocab_size = len(char_list)
        
#        all_list_encode = [[self.word2index.get(j, self.UNK_ID) for j in i] for i in all_list] 
#        all_list_encode = [i[:self.sequence_length] if len(i) > self.sequence_length else i+[0]*(self.sequence_length-len(i)) for i in all_list_encode] 
#        all_list_length = [len(i) for i in all_list] 
#        all_list_label = [[0 for _ in i] for i in all_list]
#        all_list_label = [i[:self.sequence_length] if len(i) > self.sequence_length else i+[0]*(self.sequence_length-len(i)) for i in all_list_label] 
        
        anchor_list_encode = [[self.word2index.get(j, self.UNK_ID) for j in i[0]] for i in anchor_list] 
        anchor_list_encode = [i[:self.sequence_length] if len(i) > self.sequence_length else i+[0]*(self.sequence_length-len(i)) for i in anchor_list_encode] 
        anchor_list_length = [len(i) for i in anchor_list] 
        anchor_list_label = [[0 for _ in i] for i in anchor_list]
        anchor_list_label = [i[:self.sequence_length] if len(i) > self.sequence_length else i+[0]*(self.sequence_length-len(i)) for i in anchor_list_label] 
        for i in range(len(anchor_list_label)):
            e = min(anchor_list[i][2], self.sequence_length)
            s = min(anchor_list[i][1], self.sequence_length)
            anchor_list_label[i][s:e] = [1] * (e - s)
            
        
#        self.X_train = all_list_encode + anchor_list_encode
#        self.length_train = all_list_length + anchor_list_length
#        self.Y_train = all_list_label + anchor_list_label
        self.X_train = anchor_list_encode
        self.length_train = anchor_list_length
        self.Y_train = anchor_list_label
        self.label_num = 2
        
    def build(self):
        
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length], name='danmu')
        self.seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
        self.labels = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name="labels")
        
        self.embedding_table = tf.get_variable(name="char_embeddings", dtype=tf.float32,
            shape=[self.vocab_size, self.hidden_num])
        char_embeddings = tf.nn.embedding_lookup(self.embedding_table, self.x)
        word_lengths = tf.reshape(self.seq_length, shape=[-1])
        
        # 3. bi lstm on chars
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_num, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_num, state_is_tuple=True)
        
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
            cell_bw, char_embeddings, sequence_length=word_lengths,
            dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
#        print(output.shape)
        
        char_rep = tf.reshape(output, shape=[-1, self.sequence_length, 2*self.hidden_num])
        
        self.W = tf.get_variable("W", shape=[2*self.hidden_num, self.label_num],
                        dtype=tf.float32)
        self.b = tf.get_variable("b", shape=[self.label_num], dtype=tf.float32,
                        initializer=tf.zeros_initializer())
#        tf.reshape(tf.einsum('ijk,kh->ijh', inputs, tensor), shape=[-1, sentence_length])
        pred = tf.einsum('ijk,kh->ijh', char_rep, self.W) + self.b
        self.scores = tf.reshape(pred, [-1, self.sequence_length, self.label_num])        
        
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
        self.scores, self.labels, self.seq_length)
        print(self.scores.shape)
        print(self.labels.shape)
        print(self.seq_length.shape)
        self.loss = tf.reduce_mean(-log_likelihood)   

        global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=global_step, learning_rate=self.learning_rate,
        #                                           optimizer = self.optimizer, clip_gradients=self.clip_gradients)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss,  global_step=global_step)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self):
        number_of_training_data = len(self.X_train)
        for epoch in range(self.epoch):
            batch = number_of_training_data // self.batch_size
            for i in range(batch):
                if i == batch-1:
                    batch_x = self.X_train[self.batch_size*i:]
                    batch_y = self.Y_train[self.batch_size*i:]
                    batch_length = self.length_train[self.batch_size*i:]
                else:
                    batch_x = self.X_train[self.batch_size*i:self.batch_size*(i+1)]
                    batch_y = self.Y_train[self.batch_size*i:self.batch_size*(i+1)]
                    batch_length = self.length_train[self.batch_size*i:self.batch_size*(i+1)]
                # print(batch_y)
                feed_dict = {self.x:np.array(batch_x).reshape((-1, self.sequence_length)),
                             self.labels:np.array(batch_y).reshape((-1, self.sequence_length)),
                             self.seq_length:np.array(batch_length).reshape((-1))}
                curr_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict)
              
#            feed_dict_test = {self.x:self.X_test, self.y:self.Y_test}
#            _, self.prob_all = self.sess.run([self.loss, self.probability], feed_dict_test)
#            
#            self.print_acc()

            print("Going to save model..")
            self.saver.save(self.sess, self.save_path, global_step=epoch)

    def load_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.ckpt_dir))

    def predict(self, x):
        y_pred = np.zeros((1, self.sequence_length))
        x_pred = emoj_sub(x)
        x_pred = url_sub(x_pred)
        x_pred = [self.word2index.get(i, self.UNK_ID) for i in x_pred]
        x_pred = x_pred[:self.sequence_length] if len(x_pred) > self.sequence_length else x_pred + [0] * (self.sequence_length - len(x_pred))
        x_pred = np.array(x_pred).reshape((1, -1))
        feed_dict = {self.x:x_pred, self.labels:y_pred, self.seq_length:np.array([len(x)])}
        score, transition = self.sess.run([self.scores, self.transition_params], feed_dict)
        score = np.reshape(score, (self.sequence_length, 2))
        self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.viterbi_decode(
                                score, transition)
        
        labels, scores = self.viterbi_sequence[:len(x)], self.viterbi_score
        return labels, scores


