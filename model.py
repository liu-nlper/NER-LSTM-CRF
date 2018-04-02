#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    模型: bi-lstm + crf
"""
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import uniform_tensor, get_sequence_actual_length, \
    zero_nil_slot, shuffle_matrix


def get_activation(activation=None):
    """
    Get activation function accord to the parameter 'activation'
    Args:
        activation: str: 激活函数的名称
    Return:
        激活函数
    """
    if activation is None:
        return None
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'softmax':
        return tf.nn.softmax
    elif activation == 'sigmoid':
        return tf.sigmoid
    else:
        raise Exception('Unknow activation function: %s' % activation)


class MultiConvolutional3D(object):

    def __init__(self, input_data, filter_length_list, nb_filter_list, padding='VALID',
                 activation='relu', pooling='max', name='Convolutional3D'):
        """3D卷积层
        Args:
            input_data: 4D tensor of shape=[batch_size, sent_len, word_len, char_dim]
                in_channels is set to 1 when use Convolutional3D.
            filter_length_list: list of int, 卷积核的长度，用于构造卷积核，在
                Convolutional1D中，卷积核shape=[filter_length, in_width, in_channels, nb_filters]
            nb_filter_list: list of int, 卷积核数量
            padding: 默认'VALID'，暂时不支持设成'SAME'
        """
        assert padding in ('VALID'), 'Unknow padding %s' % padding
        # assert padding in ('VALID', 'SAME'), 'Unknow padding %s' % padding

        # expand dim
        char_dim = int(input_data.get_shape()[-1])  # char的维度
        self._input_data = tf.expand_dims(input_data, -1)  # shape=[x, x, x, 1]
        self._filter_length_list = filter_length_list
        self._nb_filter_list = nb_filter_list
        self._padding = padding
        self._activation = get_activation(activation)
        self._name = name

        pooling_outpouts = []
        for i in range(len(self._filter_length_list)):
            filter_length = self._filter_length_list[i]
            nb_filter = self._nb_filter_list[i]
            with tf.variable_scope('%s_%d' % (name, filter_length)) as scope:
                # shape= [batch_size, sent_len-filter_length+1, word_len, 1, nb_filters]
                conv_output = tf.contrib.layers.conv3d(
                    inputs=self._input_data,
                    num_outputs=nb_filter,
                    kernel_size=[1, filter_length, char_dim],
                    padding=self._padding)
                # output's shape=[batch_size, new_height, 1, nb_filters]
                act_output = (
                    conv_output if activation is None
                    else self._activation(conv_output))
                # max pooling，shape = [batch_size, sent_len, nb_filters]
                if pooling == 'max':
                    pooling_output = tf.reduce_max(tf.squeeze(act_output, [-2]), 2)
                elif pooling == 'mean':
                    pooling_output = tf.reduce_mean(tf.squeeze(act_output, [-2]), 2)
                else:
                    raise Exception('pooling must in (max, mean)!')
                pooling_outpouts.append(pooling_output)

                scope.reuse_variables()
        # [batch_size, sent_len, sum(nb_filter_list]
        self._output = tf.concat(pooling_outpouts, axis=-1)

    @property
    def output(self):
        return self._output

    @property
    def output_dim(self):
        return sum(self._nb_filter_list)


class SequenceLabelingModel(object):

    def __init__(self, sequence_length, nb_classes, nb_hidden=512, num_layers=1,
                 rnn_dropout=0., feature_names=None, feature_init_weight_dict=None,
                 feature_weight_shape_dict=None, feature_weight_dropout_dict=None,
                 dropout_rate=0., use_crf=True, path_model=None, nb_epoch=200,
                 batch_size=128, train_max_patience=10, l2_rate=0.01, rnn_unit='lstm',
                 learning_rate=0.001, clip=None, use_char_feature=False, word_length=None,
                 conv_filter_size_list=None, conv_filter_len_list=None, cnn_dropout_rate=0.):
        """
        Args:
          sequence_length: int, 输入序列的padding后的长度
          nb_classes: int, 标签类别数量
          nb_hidden: int, lstm/gru层的结点数
          num_layers: int, lstm/gru层数
          rnn_dropout: lstm层的dropout值

          feature_names: list of str, 特征名称集合
          feature_init_weight_dict: dict, 键:特征名称, 值:np,array, 特征的初始化权重字典
          feature_weight_shape_dict: dict，特征embedding权重的shape，键:特征名称, 值: shape(tuple)。
          feature_weight_dropout_dict: feature name to float, feature weights dropout rate

          dropout: float, dropout rate
          use_crf: bool, 标示是否使用crf层
          path_model: str, 模型保存的路径
          nb_epoch: int, 训练最大迭代次数
          batch_size: int
          train_max_patience: int, 在dev上的loss对于train_max_patience次没有提升，则early stopping

          l2_rate: float

          rnn_unit: str, lstm or gru
          learning_rate: float, default is 0.001
          clip: None or float, gradients clip

          use_char_feature: bool,是否使用字符特征
          word_length: int, 单词长度
        """
        self._sequence_length = sequence_length
        self._nb_classes = nb_classes
        self._nb_hidden = nb_hidden
        self._num_layers = num_layers
        self._rnn_dropout = rnn_dropout

        self._feature_names = feature_names
        self._feature_init_weight_dict = feature_init_weight_dict if \
            feature_init_weight_dict else dict()
        self._feature_weight_shape_dict = feature_weight_shape_dict
        self._feature_weight_dropout_dict = feature_weight_dropout_dict

        self._dropout_rate = dropout_rate
        self._use_crf = use_crf

        self._path_model = path_model
        self._nb_epoch = nb_epoch
        self._batch_size = batch_size
        self._train_max_patience = train_max_patience

        self._l2_rate = l2_rate
        self._rnn_unit = rnn_unit
        self._learning_rate = learning_rate
        self._clip = clip

        self._use_char_feature = use_char_feature
        self._word_length = word_length
        self._conv_filter_len_list = conv_filter_len_list
        self._conv_filter_size_list = conv_filter_size_list
        self._cnn_dropout_rate = cnn_dropout_rate

        assert len(feature_names) == len(list(set(feature_names))), \
            'duplication of feature names!'

        # init ph, weights and dropout rate
        self.input_feature_ph_dict = dict()
        self.weight_dropout_ph_dict = dict()
        self.feature_weight_dict = dict()
        self.nil_vars = set()
        self.dropout_rate_ph = tf.placeholder(tf.float32, name='dropout_rate_ph')
        self.rnn_dropout_rate_ph = tf.placeholder(tf.float32, name='rnn_dropout_rate_ph')
        # label ph
        self.input_label_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, self._sequence_length], name='input_label_ph')
        if self._use_char_feature:
            self.cnn_dropout_rate_ph = tf.placeholder(tf.float32, name='cnn_dropout_rate_ph')

        self.build_model()

    def build_model(self):
        for feature_name in self._feature_names:

            # input ph
            self.input_feature_ph_dict[feature_name] = tf.placeholder(
                dtype=tf.int32, shape=[None, self._sequence_length],
                name='input_feature_ph_%s' % feature_name)

            # dropout rate ph
            self.weight_dropout_ph_dict[feature_name] = tf.placeholder(
                tf.float32, name='dropout_ph_%s' % feature_name)

            # init feature weights, 初始化未指定的
            if feature_name not in self._feature_init_weight_dict:
                feature_weight = uniform_tensor(
                    shape=self._feature_weight_shape_dict[feature_name],
                    name='f_w_%s' % feature_name)
                self.feature_weight_dict[feature_name] = tf.Variable(
                    initial_value=feature_weight, name='feature_weigth_%s' % feature_name)
            else:
                self.feature_weight_dict[feature_name] = tf.Variable(
                    initial_value=self._feature_init_weight_dict[feature_name],
                    name='feature_weight_%s' % feature_name)
                self.nil_vars.add(self.feature_weight_dict[feature_name].name)

            # init dropout rate, 初始化未指定的
            if feature_name not in self._feature_weight_dropout_dict:
                self._feature_weight_dropout_dict[feature_name] = 0.
        # char feature
        if self._use_char_feature:
            # char feature weights
            feature_weight = uniform_tensor(
                shape=self._feature_weight_shape_dict['char'], name='f_w_%s' % 'char')
            self.feature_weight_dict['char'] = tf.Variable(
                initial_value=feature_weight, name='feature_weigth_%s' % 'char')
            self.nil_vars.add(self.feature_weight_dict['char'].name)
            self.nil_vars.add(self.feature_weight_dict['char'].name)
            self.input_feature_ph_dict['char'] = tf.placeholder(
                dtype=tf.int32, shape=[None, self._sequence_length, self._word_length],
                name='input_feature_ph_%s' % 'char')

        # init embeddings
        self.embedding_features = []
        for feature_name in self._feature_names:
            embedding_feature = tf.nn.dropout(tf.nn.embedding_lookup(
                self.feature_weight_dict[feature_name],
                ids=self.input_feature_ph_dict[feature_name],
                name='embedding_feature_%s' % feature_name),
                keep_prob=1.-self.weight_dropout_ph_dict[feature_name],
                name='embedding_feature_dropout_%s' % feature_name)
            self.embedding_features.append(embedding_feature)
        # char embedding
        if self._use_char_feature:
            char_embedding_feature = tf.nn.embedding_lookup(
                self.feature_weight_dict['char'],
                ids=self.input_feature_ph_dict['char'],
                name='embedding_feature_%s' % 'char')
            # conv
            couv_feature_char = MultiConvolutional3D(
                char_embedding_feature, filter_length_list=self._conv_filter_len_list,
                nb_filter_list=self._conv_filter_size_list).output
            couv_feature_char = tf.nn.dropout(
                couv_feature_char, keep_prob=1-self.cnn_dropout_rate_ph)

        # concat all features
        input_features = self.embedding_features[0] if len(self.embedding_features) == 1 \
            else tf.concat(values=self.embedding_features, axis=2, name='input_features')
        if self._use_char_feature:
            input_features = tf.concat([input_features, couv_feature_char], axis=-1)

        # multi bi-lstm layer
        _fw_cells = []
        _bw_cells = []
        for _ in range(self._num_layers):
            fw, bw = self._get_rnn_unit(self._rnn_unit)
            _fw_cells.append(tf.nn.rnn_cell.DropoutWrapper(fw, output_keep_prob=1-self.rnn_dropout_rate_ph))
            _bw_cells.append(tf.nn.rnn_cell.DropoutWrapper(bw, output_keep_prob=1-self.rnn_dropout_rate_ph))
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(_fw_cells)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(_bw_cells)

        # 计算self.input_features[feature_names[0]]的实际长度(0为padding值)
        self.sequence_actual_length = get_sequence_actual_length(  # 每个句子的实际长度
            self.input_feature_ph_dict[self._feature_names[0]])
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, input_features, scope='bi-lstm',
            dtype=tf.float32, sequence_length=self.sequence_actual_length)
        # shape = [batch_size, max_len, nb_hidden*2]
        lstm_output = tf.nn.dropout(
            tf.concat(rnn_outputs, axis=2, name='lstm_output'),
            keep_prob=1.-self.dropout_rate_ph, name='lstm_output_dropout')

        # softmax
        hidden_size = int(lstm_output.shape[-1])
        self.outputs = tf.reshape(lstm_output, [-1, hidden_size], name='outputs')
        self.softmax_w = tf.get_variable('softmax_w', [hidden_size, self._nb_classes])
        self.softmax_b = tf.get_variable('softmax_b', [self._nb_classes])
        self.logits = tf.reshape(
            tf.matmul(self.outputs, self.softmax_w) + self.softmax_b,
            shape=[-1, self._sequence_length, self._nb_classes], name='logits')

        # 计算loss
        self.loss = self.compute_loss()
        self.l2_loss = self._l2_rate * (tf.nn.l2_loss(self.softmax_w) + tf.nn.l2_loss(self.softmax_b))

        self.total_loss = self.loss + self.l2_loss

        # train op
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.total_loss)
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._clip:
            # clip by global norm
            gradients, variables = zip(*nil_grads_and_vars)
            gradients, _ = tf.clip_by_global_norm(gradients, self._clip)
            self.train_op = optimizer.apply_gradients(
                zip(gradients, variables), name='train_op', global_step=global_step)
        else:
            self.train_op = optimizer.apply_gradients(
                nil_grads_and_vars, name='train_op', global_step=global_step)

        # TODO sess, visible_device_list待修改
        gpu_options = tf.GPUOptions(visible_device_list='0', allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # init all variable
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _get_rnn_unit(self, rnn_unit):
        if rnn_unit == 'lstm':
            fw_cell = rnn.BasicLSTMCell(self._nb_hidden, forget_bias=1., state_is_tuple=True)
            bw_cell = rnn.BasicLSTMCell(self._nb_hidden, forget_bias=1., state_is_tuple=True)
        elif rnn_unit == 'gru':
            fw_cell = rnn.GRUCell(self._nb_hidden)
            bw_cell = rnn.GRUCell(self._nb_hidden)
        else:
            raise ValueError('rnn_unit must in (lstm, gru)!')
        return fw_cell, bw_cell

    def fit(self, data_dict, dev_size=0.2, seed=1337):
        """
        训练
        Args:
            data_dict: dict, 键: 特征名(or 'label'), 值: np.array
            dev_size: float, 开发集所占的比例，default is 0.2

            batch_size: int
            seed: int, for shuffle data
        """
        data_train_dict, data_dev_dict = self.split_train_dev(data_dict, dev_size=dev_size)
        self.saver = tf.train.Saver()  # save model
        train_data_count = data_train_dict['label'].shape[0]
        nb_train = int(math.ceil(train_data_count / float(self._batch_size)))
        min_dev_loss = 1000  # 全局最小dev loss, for early stopping)
        current_patience = 0  # for early stopping
        for step in range(self._nb_epoch):
            print('Epoch %d / %d:' % (step+1, self._nb_epoch))

            # shuffle train data
            data_list = [data_train_dict['label']]
            [data_list.append(data_train_dict[name]) for name in self._feature_names]
            shuffle_matrix(*data_list, seed=seed)

            # train
            train_loss, l2_loss = 0., 0.
            for i in tqdm(range(nb_train)):
                feed_dict = dict()
                batch_indices = np.arange(i * self._batch_size, (i + 1) * self._batch_size) \
                    if (i+1)*self._batch_size <= train_data_count else \
                    np.arange(i * self._batch_size, train_data_count)
                # feature feed and dropout feed
                for feature_name in self._feature_names:  # features
                    # feature
                    batch_data = data_train_dict[feature_name][batch_indices]
                    item = {self.input_feature_ph_dict[feature_name]: batch_data}
                    feed_dict.update(item)
                    # dropout
                    dropout_rate = self._feature_weight_dropout_dict[feature_name]
                    item = {self.weight_dropout_ph_dict[feature_name]: dropout_rate}
                    feed_dict.update(item)
                if self._use_char_feature:
                    batch_data = data_train_dict['char'][batch_indices]
                    item = {self.input_feature_ph_dict['char']: batch_data}
                    feed_dict.update(item)
                    item = {self.cnn_dropout_rate_ph: self._cnn_dropout_rate}
                    feed_dict.update(item)
                feed_dict.update(
                    {
                        self.dropout_rate_ph: self._dropout_rate,
                        self.rnn_dropout_rate_ph: self._rnn_dropout,
                    })
                # label feed
                batch_label = data_train_dict['label'][batch_indices]
                feed_dict.update({self.input_label_ph: batch_label})

                _, loss, l2_loss = self.sess.run([self.train_op, self.loss, self.l2_loss], feed_dict=feed_dict)
                train_loss += loss
            train_loss /= float(nb_train)

            # 计算在开发集上的loss
            dev_loss = self.evaluate(data_dev_dict)

            print('train loss: %f, dev loss: %f, l2 loss: %f' % (train_loss, dev_loss, l2_loss))

            # 根据dev上的表现保存模型
            if not self._path_model:
                continue
            if dev_loss < min_dev_loss:
                min_dev_loss = dev_loss
                current_patience = 0
                # save model
                self.saver.save(self.sess, self._path_model)
                print('model has saved to %s!' % self._path_model)
            else:
                current_patience += 1
                print('no improvement, current patience: %d / %d' %
                      (current_patience, self._train_max_patience))
                if self._train_max_patience and current_patience >= self._train_max_patience:
                    print('\nfinished training! (early stopping, max patience: %d)'
                          % self._train_max_patience)
                    return
        print('\nfinished training!')
        return

    def split_train_dev(self, data_dict, dev_size=0.2):
        """
        划分为开发集和测试集
        Args:
            data_dict: dict, 键: 特征名(or 'label'), 值: np.array
            dev_size: float, 开发集所占的比例，default is 0.2
        Returns:
            data_train_dict, data_dev_dict: same type as data_dict
        """
        data_train_dict, data_dev_dict = dict(), dict()
        for name in data_dict.keys():
            boundary = int((1.-dev_size) * data_dict[name].shape[0])
            data_train_dict[name] = data_dict[name][:boundary]
            data_dev_dict[name] = data_dict[name][boundary:]
        return data_train_dict, data_dev_dict

    def evaluate(self, data_dict):
        """
        计算loss
        Args:
            data_dict: dict
        Return:
            loss: float
        """
        data_count = data_dict['label'].shape[0]
        nb_eval = int(math.ceil(data_count / float(self._batch_size)))
        eval_loss = 0.
        for i in range(nb_eval):
            feed_dict = dict()
            batch_indices = np.arange(i * self._batch_size, (i + 1) * self._batch_size) \
                if (i+1)*self._batch_size <= data_count else \
                np.arange(i * self._batch_size, data_count)
            for feature_name in self._feature_names:  # features and dropout
                batch_data = data_dict[feature_name][batch_indices]
                item = {self.input_feature_ph_dict[feature_name]: batch_data}
                feed_dict.update(item)
                # dropout
                item = {self.weight_dropout_ph_dict[feature_name]: 0.}
                feed_dict.update(item)
            if self._use_char_feature:
                batch_data = data_dict['char'][batch_indices]
                item = {self.input_feature_ph_dict['char']: batch_data}
                feed_dict.update(item)
                item = {self.cnn_dropout_rate_ph: 0.}
                feed_dict.update(item)
            feed_dict.update({self.dropout_rate_ph: 0., self.rnn_dropout_rate_ph: 0.})
            # label feed
            batch_label = data_dict['label'][batch_indices]
            feed_dict.update({self.input_label_ph: batch_label})

            loss = self.sess.run(self.loss, feed_dict=feed_dict)
            eval_loss += loss
        eval_loss /= float(nb_eval)
        return eval_loss

    def predict(self, data_test_dict):
        """
        根据训练好的模型标记数据
        Args:
            data_test_dict: dict
        Return:
            pass
        """
        print('predicting...')
        data_count = data_test_dict[self._feature_names[0]].shape[0]
        nb_test = int(math.ceil(data_count / float(self._batch_size)))
        result_sequences = []  # 标记结果
        for i in tqdm(range(nb_test)):
            feed_dict = dict()
            batch_indices = np.arange(i * self._batch_size, (i + 1) * self._batch_size) \
                if (i+1)*self._batch_size <= data_count else \
                np.arange(i * self._batch_size, data_count)
            for feature_name in self._feature_names:  # features and dropout
                batch_data = data_test_dict[feature_name][batch_indices]
                item = {self.input_feature_ph_dict[feature_name]: batch_data}
                feed_dict.update(item)
                # dropout
                item = {self.weight_dropout_ph_dict[feature_name]: 0.}
                feed_dict.update(item)
            if self._use_char_feature:
                batch_data = data_test_dict['char'][batch_indices]
                item = {self.input_feature_ph_dict['char']: batch_data}
                feed_dict.update(item)
                item = {self.cnn_dropout_rate_ph: 0.}
                feed_dict.update(item)
            feed_dict.update({self.dropout_rate_ph: 0., self.rnn_dropout_rate_ph: 0.})

            if self._use_crf:
                logits, sequence_actual_length, transition_params = self.sess.run(
                    [self.logits, self.sequence_actual_length, self.transition_params], feed_dict=feed_dict)
                for logit, seq_len in zip(logits, sequence_actual_length):
                    logit_actual = logit[:seq_len]
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                        logit_actual, transition_params)
                    result_sequences.append(viterbi_sequence)
            else:
                logits, sequence_actual_length = self.sess.run(
                    [self.logits, self.sequence_actual_length], feed_dict=feed_dict)
                for logit, seq_len in zip(logits, sequence_actual_length):
                    logit_actual = logit[:seq_len]
                    sequence = np.argmax(logit_actual, axis=-1)
                    result_sequences.append(sequence)
        print('共标记句子数: %d' % data_count)
        return result_sequences

    def compute_loss(self):
        """
        计算loss

        Return:
            loss: scalar
        """
        if not self._use_crf:
            labels = tf.reshape(
                tf.contrib.layers.one_hot_encoding(
                    tf.reshape(self.input_label_ph, [-1]), num_classes=self._nb_classes),
                shape=[-1, self._sequence_length, self._nb_classes])
            logits = tf.nn.softmax(self.logits, dim=-1)
            cross_entropy = -tf.reduce_sum(labels * tf.log(logits), axis=2)
            mask = tf.sign(tf.reduce_max(tf.abs(labels), axis=2))
            cross_entropy_masked = tf.reduce_sum(
                cross_entropy*mask, axis=1) / tf.cast(self.sequence_actual_length, tf.float32)
            return tf.reduce_mean(cross_entropy_masked)
        else:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.input_label_ph, self.sequence_actual_length)
            return tf.reduce_mean(-log_likelihood)
