import tensorflow as tf
import numpy as np
import time
import shutil


def expert_unit(hidden, expert_num, expert_layer, gate_layer, gate_norm, bn, is_training):

    # hidden:(-1, h), expert_net
    for i in range(expert_num):
        for j in range(len(expert_layer)):
            if j == 0:
                hidden_temp = hidden
            hidden_temp = tf.layers.dense(name='expert_' + str(i) + '_layer_' + str(j), inputs=hidden_temp,
                            units=expert_layer[j], activation=None, use_bias=False,
                            kernel_initializer=tf.glorot_normal_initializer())
            if bn:
                hidden_temp = tf.layers.batch_normalization(hidden_temp, training=is_training)
            hidden_temp = tf.nn.relu(hidden_temp)

        hidden_temp = tf.reshape(hidden_temp, shape=[-1, 1, expert_layer[-1]]) # (-1, 1, ?)
        if i == 0:
            expert_output = hidden_temp
        else:
            expert_output = tf.concat([expert_output, hidden_temp], axis=-2) # (-1, num, ?)

    # gate_net
    for i in range(len(gate_layer)):
        if i == 0:
            gate_temp = hidden
        gate_temp = tf.layers.dense(name='gate_layer_' + str(i), inputs=gate_temp,
                        units=gate_layer[i], activation=None, use_bias=False,
                        kernel_initializer=tf.glorot_normal_initializer())
        if bn:
            gate_temp = tf.layers.batch_normalization(gate_temp, training=is_training)
        gate_temp = tf.nn.relu(gate_temp)

    gate_temp = tf.layers.dense(name='gate_layer_last', inputs=gate_temp,
                    units=expert_num, activation=None, use_bias=False,
                    kernel_initializer=tf.glorot_normal_initializer())
    if bn:
        gate_temp = tf.layers.batch_normalization(gate_temp, training=is_training)
    gate_temp = tf.nn.relu(gate_temp) #(-1, num)

    if gate_norm == "softmax":
        gate_temp = tf.nn.softmax(gate_temp, axis=-1)
    elif gate_norm == "sum":
        gate_temp_sum = tf.reduce_sum(gate_temp, axis=-1, keep_dims=True) + 1e-7
        gate_temp_sum = tf.tile(gate_temp_sum, [-1, expert_num])
        gate_temp = gate_temp / gate_temp_sum

    gate_temp = tf.tile(tf.expand_dims(gate_temp, axis=-1), [1, 1, expert_layer[-1]])

    expert_output = tf.reduce_sum(expert_output * gate_temp, axis=-2) # (-1, ?)

    return expert_output



class ccDeep(object):

    def __init__(self, config):

        self.feature_dict = config['feature']
        self.label_dict = config['label']
        self.l1_reg_index = config['l1_reg']
        self.l2_reg_index = config['l2_reg']
        self.bn = config['bn']
        self.dense_size = config['dense_size']
        self.l1_reg = tf.contrib.layers.l1_regularizer(self.l1_reg_index)
        self.l2_reg = tf.contrib.layers.l2_regularizer(self.l2_reg_index)
        self.expert_use = True if config['expert'] else False
        if self.expert_use:
            self.expert_num = config['expert']['expert_num']
            self.expert_layer = config['expert']['expert_layer']
            self.gate_layer = config['expert']['gate_layer']
            self.gate_norm = config['expert']['gate_norm']

        self.input_all_dense, self.input_all_kv_k, self.input_all_kv_v, self.kv_share, \
        self.kv_class, self.kv_embedsize, self.kv_comb = [], [], [], [], [], [], []

    def build(self):
        for i, jj in enumerate(self.feature_dict.items()):
            j = jj[1]
            if j['type'] == 'FixLenDense':
                self.input_all_dense.append(tf.placeholder(tf.float32, shape=(None, j['length']), name=jj[0]))
            elif j['type'] == 'FixLenKeyValue':
                self.input_all_kv_k.append(tf.placeholder(tf.int32, shape=(None, j['length']), name=jj[0]))
                self.input_all_kv_v.append(tf.placeholder(tf.float32, shape=(None, j['length']), name=jj[0]+'_weight'))
                self.kv_embedsize.append(j['embedding_dim'])
                self.kv_class.append(j['size'])
                self.kv_share.append(j['share'])
                self.kv_comb.append(j['combine'])

        self.is_training = tf.placeholder(tf.bool, (1,1), name='bn_index')


        self.label = tf.placeholder(tf.float32, shape=(None, 1), name='label')
        self.label_weight = tf.placeholder(tf.float32, shape=(None, 1), name='label_weight')

        self.l_kv = len(self.kv_share)
        for i in range(self.l_kv):
            name_kv = self.kv_share[i]
            with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
                w = tf.get_variable(name='Embedding_' + name_kv, shape=[self.kv_class[i], self.kv_embedsize[i]],
                            initializer=tf.glorot_normal_initializer(),
                            dtype=tf.float32,
                            regularizer=self.l1_reg)
            embed_kv = tf.nn.embedding_lookup(w, self.input_all_kv_k[i]) # (-1, l, h)
            kv_v = tf.expand_dims(self.input_all_kv_v[i], axis=-1)
            kv_v = tf.tile(kv_v, [1, 1, self.kv_embedsize[i]])
            kv_res = tf.multiply(embed_kv, kv_v)

            # (-1, h)
            if self.kv_comb[i] == 'sum':
                temp = tf.reduce_sum(kv_res, axis=-2)
            # (-1, **)
            elif self.kv_comb[i] == 'concat':
                temp = tf.layers.flatten(kv_res)

            if i == 0:
                dense_input = temp
            else:
                dense_input = tf.concat([dense_input, temp], axis=-1)

        if self.input_all_dense:
            for input in self.input_all_dense:
                dense_input = tf.concat([dense_input, input], axis=-1)

        hidden = dense_input


        if self.expert_use:
            hidden = expert_unit(hidden, self.expert_num, self.expert_layer, self.gate_layer, self.gate_norm, self.bn, self.is_training)
        else:
            for i in range(len(self.dense_size)):
                hidden = tf.layers.dense(name='dense_layer_' + str(i), inputs=hidden, units=self.dense_size[i], activation=None,
                                         use_bias=False,
                                         kernel_initializer=tf.glorot_normal_initializer())
                if self.bn:
                    hidden = tf.layers.batch_normalization(hidden, training=self.is_training)
                hidden = tf.nn.relu(hidden)

        dense_out = tf.layers.dense(name='deep_out', inputs=hidden,
                                   units=1, activation=None,
                                   kernel_initializer=tf.glorot_normal_initializer())  # shape=[-1, 1]

        bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

        logit = dense_out + bias
        # logit = tf.reshape(logit, shape=[-1])
        self.predict = tf.sigmoid(logit)

        # # 中间若想要summary某些变量的histgram或者distribution，需要使用以下语句
        # # 若是想要指定某些变量，可以使用var.name.startswith('*')等字符串操作来限制变量
        # vars = tf.trainable_variables()
        # for var in vars:
        #     tf.summary.histogram(var.name, var)
        sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=self.label)
        if self.label_dict['weight']:
            sample_loss = sample_loss * self.label_weight

        self.loss = tf.reduce_mean(sample_loss)
        # self.acc = tf.metrics.accuracy(self.label, self.predict)
        self.auc = tf.metrics.auc(self.label, self.predict)

        self.loss1 = tf.reduce_mean(sample_loss)
        self.auc1 = tf.metrics.auc(self.label, self.predict)

        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.loss)

        tf.summary.scalar('loss_train', self.loss)
        tf.summary.scalar('auc_train', self.auc[1])
        tf.summary.scalar('loss_test', self.loss1)
        tf.summary.scalar('auc_test', self.auc1[1])
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        try:
            shutil.rmtree('log/train/')
            shutil.rmtree('log/test/')
        except Exception as e:
            print(e, "at clear_model")
        else:
            print("start new learning task -> existing model cleaned")

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter('log/train/', self.sess.graph)
        self.writer1 = tf.summary.FileWriter('log/test/', self.sess.graph)
        self.merged = tf.summary.merge_all()

        self.sess.run(init)

    def train_gen(self, data, s, e, is_bn, is_training):

        index_dense, index_kv, training_data_dict = False, False, {}

        dense_feature, kv_feature, label = data

        if dense_feature:
            index_dense = True

        if kv_feature:
            index_kv = True

        if index_dense:
            for i in range(len(dense_feature)):
                training_data_dict[self.input_all_dense[i]] = dense_feature[i][s:e]
        if index_kv:
            for i in range(len(kv_feature)):
                kv_feature_temp = kv_feature[i][s:e]
                training_data_dict[self.input_all_kv_k[i]] = [[k[0] for k in j] for j in kv_feature_temp]
                training_data_dict[self.input_all_kv_v[i]] = [[k[1] for k in j] for j in kv_feature_temp]

        label_temp = label[s:e]
        training_data_dict[self.label] = np.array([j[0] for j in label_temp]).reshape((-1,1))
        training_data_dict[self.label_weight] = np.array([j[1] for j in label_temp]).reshape((-1,1))
        
        if is_training:
            training_data_dict[self.is_training] = [[True]]
        else:
            training_data_dict[self.is_training] = [[False]]

        return training_data_dict

    def train_and_eval(self, training_datas, testing_datas, epochs=6, batch_size=128):

        training_num = len(training_datas[-1])
        testing_num = len(testing_datas[-1])
        self.loss_test, self.auc_test = [], []
        for epoch in range(epochs):
            s = time.time()
            batch = training_num // batch_size
            for i in range(batch):
                training_data_dict = self.train_gen(training_datas, i*batch_size, min((i+1)*batch_size, training_num), self.bn, True)
                _, loss_train, auc_train, summary_train = self.sess.run([self.train_op, self.loss, self.auc, self.merged], feed_dict=training_data_dict)
                self.writer.add_summary(summary_train, epoch * training_num + i)
                if i % 10000 == 0:
                    testing_data_dict = self.train_gen(testing_datas, 0, testing_num, self.bn, False)
                    loss_test, auc_test, summary_test = self.sess.run([self.loss1, self.auc1, self.merged], feed_dict=testing_data_dict)
                    self.writer1.add_summary(summary_train, epoch * training_num + i)
                    self.loss_test.append(loss_test)
                    self.auc_test.append(auc_test)
                    print('epoch %s batch %s ,loss_train is %s,auc_train is %s, loss_test is %s,auc_test is %s.' % (epoch, i, loss_train, auc_train[1], loss_test, auc_test[1]))
            e = time.time()
            print('use time %s' % str(e-s))
        
        np.savetxt("expert"+str(self.expert_use)+".txt", self.auc_test)
        self.writer.close()
        self.writer1.close()

class ccDeepAtt(object):

    def __init__(self, config, config_att, config_lrfm):

        self.feature_dict = config['feature']
        self.label_dict = config['label']
        self.l1_reg_index = config['l1_reg']
        self.l2_reg_index = config['l2_reg']
        self.bn = config['bn']
        self.dense_size = config['dense_size']
        self.att_config = config_att
        self.lrfm_config = config_lrfm
        self.l1_reg = tf.contrib.layers.l1_regularizer(self.l1_reg_index)
        self.l2_reg = tf.contrib.layers.l2_regularizer(self.l2_reg_index)

        self.input_all_dense, self.input_all_kv_k, self.input_all_kv_v, self.d_column, self.kv_column_num, self.kv_share, \
        self.kv_class, self.kv_embedsize, self.kv_comb, self.kv_column = [], [], [], [], [], [], [], [], [], []

    def build(self):
        for i, jj in enumerate(self.feature_dict.items()):
            j = jj[1]
            if j['type'] == 'FixLenDense':
                self.input_all_dense.append(tf.placeholder(tf.float32, shape=(None, j['length']), name=jj[0]))
                self.d_column_num.append(i)
            elif j['type'] == 'FixLenKeyValue':
                self.kv_column_num.append(i)
                self.kv_column.append(jj[0])
                self.input_all_kv_k.append(tf.placeholder(tf.int32, shape=(None, j['length']), name=jj[0]))
                self.input_all_kv_v.append(tf.placeholder(tf.float32, shape=(None, j['length']), name=jj[0]+'_weight'))
                self.kv_embedsize.append(j['embedding_dim'])
                self.kv_class.append(j['size'])
                self.kv_share.append(j['share'])
                self.kv_comb.append(j['combine'])

        if self.bn:
            self.is_training = tf.placeholder(tf.bool, (1,1),name='bn_index')


        self.label = tf.placeholder(tf.int32, shape=(None, 1), name='label')
        self.label_weight = tf.placeholder(tf.float32, shape=(None, 1), name='label_weight')

        self.l_kv = len(self.kv_share)
        self.l_d = len(self.d_column)
        ### LR
        lr_index = self.lrfm_config['lr']['use']
        if lr_index:
            pass
            # for i in range(self.l_kv):
            #     name_kv = self.kv_share[i]
            #     column_kv = self.kv_column[i]
            #     w = tf.get_variable(name='Embedding_' + name_kv, shape=[self.kv_class[i], self.kv_embedsize[i]],
            #                         initializer=tf.glorot_normal_initializer(),
            #                         dtype=tf.float32,
            #                         regularizer=self.l1_reg)
            #     embed_kv = tf.nn.embedding_lookup(w, self.input_all_kv_k[i])  # (-1, l, n)
            #     kv_v = tf.expand_dims(self.input_all_kv_v[i], axis=-1)
            #     kv_v = tf.tile(kv_v, [1, 1, self.kv_embedsize[i]])
            #
            #     index_att = False
            #     for j in range(len(weighted)):
            #         if column_kv in weighted[j]:
            #             index_att = True
            #             att_target = self.kv_column.index(attention[j])
            #             att_target_k = self.input_all_kv_k[att_target]
            #
            #     if index_att:
            #         embed_target = tf.nn.embedding_lookup(w, att_target_k)
            #         embed_target = tf.expand_dims(embed_target, axis=-2)
            #         embed_target = tf.tile(embed_target, [1, 1, self.kv_embedsize[i]])
            #         mask = tf.cast(tf.cast(kv_v, tf.bool), tf.float32)
            #         att_weight = tf.reduce_sum(embed_kv * embed_target, axis=-1, keepdims=True)
            #         att_weight = tf.tile(att_weight, [1, 1, self.kv_embedsize[i]]) * kv_v
            #         att_weight = tf.exp(att_weight) * mask  # (-1, l, h)
            #         att_weight_sum = tf.reduce_sum(att_weight, axis=-2) + 1e-7  # (-1, h)
            #         temp = att_weight * embed_kv
            #         temp = tf.reduce_sum(temp, axis=-2) / att_weight_sum
            #
            #     else:
            #         kv_res = tf.multiply(embed_kv, kv_v)
            #         if self.kv_comb[i] == 'sum':
            #             temp = tf.reduce_sum(kv_res, axis=-2)
            #         elif self.kv_comb[i] == 'concat':
            #             temp = tf.layers.flatten(kv_res)
            #
            #     if i == 0:
            #         dense_input = temp
            #     else:
            #         dense_input = tf.concat([dense_input, temp], axis=-1)
            #
            # if len(self.input_all_dense) > 0:
            #     for input in self.input_all_dense:
            #         dense_input = tf.concat([dense_input, input], axis=-1)
            #
            # hidden = dense_input



        self.l_kv = len(self.kv_share)

        attention = []
        weighted = []
        for i in self.att_config:
            attention.append(i['attention'])
            weighted.append(i['weighted'])

        for i in range(self.l_kv):
            name_kv = self.kv_share[i]
            column_kv = self.kv_column[i]
            w = tf.get_variable(name='Embedding_' + name_kv, shape=[self.kv_class[i], self.kv_embedsize[i]],
                        initializer=tf.glorot_normal_initializer(),
                        dtype=tf.float32,
                        regularizer=self.l1_reg)
            embed_kv = tf.nn.embedding_lookup(w, self.input_all_kv_k[i]) # (-1, l, n)
            kv_v = tf.expand_dims(self.input_all_kv_v[i], axis=-1)
            kv_v = tf.tile(kv_v, [1, 1, self.kv_embedsize[i]])

            index_att = False
            for j in range(len(weighted)):
                if column_kv in weighted[j]:
                    index_att = True
                    att_target = self.kv_column.index(attention[j])
                    att_target_k = self.input_all_kv_k[att_target]

            if index_att:
                embed_target = tf.nn.embedding_lookup(w, att_target_k)
                embed_target = tf.expand_dims(embed_target, axis=-2)
                embed_target = tf.tile(embed_target, [1, 1, self.kv_embedsize[i]])
                mask = tf.cast(tf.cast(kv_v, tf.bool), tf.float32)
                att_weight = tf.reduce_sum(embed_kv * embed_target, axis=-1, keepdims=True)
                att_weight = tf.tile(att_weight, [1, 1, self.kv_embedsize[i]]) * kv_v
                att_weight = tf.exp(att_weight) * mask # (-1, l, h)
                att_weight_sum = tf.reduce_sum(att_weight, axis=-2) + 1e-7 # (-1, h)
                temp = att_weight * embed_kv
                temp = tf.reduce_sum(temp, axis=-2) / att_weight_sum

            else:
                kv_res = tf.multiply(embed_kv, kv_v)
                if self.kv_comb[i] == 'sum':
                    temp = tf.reduce_sum(kv_res, axis=-2)
                elif self.kv_comb[i] == 'concat':
                    temp = tf.layers.flatten(kv_res)

            if i == 0:
                dense_input = temp
            else:
                dense_input = tf.concat([dense_input, temp], axis=-1)

        if len(self.input_all_dense) > 0:
            for input in self.input_all_dense:
                dense_input = tf.concat([dense_input, input], axis=-1)

        hidden = dense_input

        for i in range(self.dense_size):
            hidden = tf.layers.dense(name='dense_layer_' + str(i), inputs=hidden, units=self.dense_size[i], activation=None,
                                     use_bias=False,
                                     kernel_initializer=tf.glorot_normal_initializer())
            hidden = tf.layers.batch_normalization(hidden, training=self.is_training)
            hidden = tf.nn.relu(hidden)

        dense_out = tf.layers.dense(name='deep_out', inputs=hidden,
                                   units=1, activation=None,
                                   kernel_initializer=tf.glorot_normal_initializer())  # shape=[-1, 1]

        bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

        logit = dense_out + bias
        logit = tf.reshape(logit, shape=[-1])
        self.predict = tf.sigmoid(logit)

        # # 中间若想要summary某些变量的histgram或者distribution，需要使用以下语句
        # # 若是想要指定某些变量，可以使用var.name.startswith('*')等字符串操作来限制变量
        # vars = tf.trainable_variables()
        # for var in vars:
        #     tf.summary.histogram(var.name, var)
        sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=self.label)
        if self.label_dict['weight']:
            sample_loss = sample_loss * self.label_weight

        self.loss = tf.reduce_mean(sample_loss)
        # self.acc = tf.metrics.accuracy(self.label, self.predict)
        self.auc = tf.metrics.auc(self.label, self.predict)

    def train_gen(self, data, s, e, is_bn, is_training):
        training_data_dict = {}
        training_data = data[s:e]
        for i in range(len(training_data[0])):
            training_data_dict[self.input_all_dense[i]] = training_data[0][i]
        for i in range(len(training_data[1])):
            training_data_dict[self.input_all_kv_k[i]] = training_data[1][i][0]
            training_data_dict[self.input_all_kv_v[i]] = training_data[1][i][1]
        training_data_dict[self.label] = training_data[2][0]
        training_data_dict[self.label_weight] = training_data[2][1]
        if is_bn:
            if is_training:
                training_data_dict[self.is_training] = True
            else:
                training_data_dict[self.is_training] = False

        return training_data_dict


    def train_and_eval(self, training_datas, testing_datas, epoch=6, batch_size=128):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()

        training_num = len(training_datas)
        testing_num = len(testing_datas)
        self.loss_test, self.auc_test = [], []
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epoch):
                batch = training_num // batch_size
                for i in range(batch):
                    training_data_dict = self.train_gen(training_datas, i*batch_size, min((i+1)*batch_size, training_num), self.bn, True)
                    _, loss_train, auc_train = sess.run([self.train_op, self.loss, self.auc], feed_dict=training_data_dict)
                    if i % 10000 == 0:
                        testing_data_dict = self.train_gen(testing_datas, 0, testing_num, self.bn, False)
                        loss_test, auc_test = sess.run([self.loss, self.auc], feed_dict=testing_data_dict)
                        self.loss_test.append(loss_test)
                        self.auc_test.append(auc_test)
                        print('epoch %s batch %s ,loss_train is %s,auc_train is %s, loss_test is %s,auc_test is %s.' % (loss_train, auc_train, loss_test, auc_test))




