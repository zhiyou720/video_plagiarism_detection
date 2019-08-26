#!/usr/bin/env python
# coding: utf-8
"""
@File     :model.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/26
@Desc     : 
"""
import keras
from keras import Input, Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Embedding, Dense, Dropout, Bidirectional, CuDNNLSTM, LSTM, Conv1D, GlobalMaxPooling1D, \
    Concatenate


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class TextAttBiRNN(object):
    def __init__(self, max_features, embedding_dims,
                 maxlen=100,
                 class_num=10,
                 last_activation='softmax',
                 gpu=True
                 ):

        self.hidden_units = 100
        self.max_len = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

        self.GPU = gpu

    def get_model(self):
        _input = Input((self.max_len,))
        embedding = Embedding(self.max_features + 1, self.embedding_dims, trainable=True)(_input)

        bn = keras.layers.BatchNormalization()(embedding)

        drop = keras.layers.Dropout(0.5)(bn)

        # lstm1 = CuDNNLSTM(self.hidden_units, return_sequences=True)  # LSTM or GRU
        #
        # lstm2 = CuDNNLSTM(self.hidden_units, return_sequences=True)(drop)  # LSTM or GRU

        if self.GPU:
            lstm3 = CuDNNLSTM(self.hidden_units)(drop)  # LSTM or GRU
        else:
            lstm3 = LSTM(self.hidden_units)(drop)  # LSTM or GRU

        # attn = Attention(int(self.maxlen))
        # model.add(attn)
        # convs = []
        # for kernel_size in [3, 4, 5]:
        #     c = Conv1D(128, kernel_size, activation='relu')(lstm2)
        #     c = GlobalMaxPooling1D()(c)
        #     c = keras.layers.Dropout(0.25)(c)
        #
        #     convs.append(c)
        # x = Concatenate()(convs)

        dense = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1))(lstm3)

        d = keras.layers.Dropout(0.5)(dense)

        output = Dense(self.class_num, activation=self.last_activation)(d)

        model = Model(inputs=_input, outputs=output)

        return model
