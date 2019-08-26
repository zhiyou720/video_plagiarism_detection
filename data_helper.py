#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/26
@Desc     : 
"""
import pickle
import itertools
import numpy as np
from collections import Counter
from tools.dataio import load_txt_data


class DataLoader:
    def __init__(self, data_path, seq_len=10, class_num=10, train=False, vocab_path='./model/vocab_config.pkl'):
        self.data_path = data_path
        self.vocab_path = vocab_path

        self.seq_len = seq_len
        self.class_num = class_num

        self.train = train
        self.x_train, self.y_train, self.vocabulary = self.load_data()

    def build_seq_data_set(self):
        """
        Load raw data and give x_train and y_train
        :return:
        """
        raw_data = load_txt_data(self.data_path)
        print('Origin data: {}'.format(len(raw_data)))

        ptr = 0
        x_train = []
        y_train = []

        while ptr + self.seq_len < len(raw_data):
            delta = 0
            tmp_x = []
            tmp_y = [0 for x in range(self.class_num)]
            while len(tmp_x) < self.seq_len:
                stack = raw_data[ptr + delta].split(',')
                tmp_x.append(stack[0] + stack[1])
                delta += 1

            x_train.append(tmp_x)

            if int(raw_data[ptr + self.seq_len].split(',')[1]) in [0, 8, 9]:
                tmp_y[0] = 1
            else:
                # tmp_y[1] = 1
                tmp_y[int(raw_data[ptr + self.seq_len].split(',')[1])] = 1  # 10 classification

            # tmp_y[int(raw_data[ptr + self.seq_len].split(',')[1])] = 1  # 10 classification
            y_train.append(tmp_y)
            ptr += 1
        return x_train, y_train

    def load_data(self):
        """
        Loads and preprocessed data for the dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and pre-process data
        sentences, labels = self.build_seq_data_set()

        if self.train:
            vocabulary, vocabulary_inv = self.build_vocab(sentences)
            with open(self.vocab_path, 'wb') as out_p:
                pickle.dump(vocabulary, out_p)
        else:
            with open(self.vocab_path, 'rb') as inp:
                vocabulary = pickle.load(inp)

        x, y = self.build_input_data(sentences, labels, vocabulary)
        return [x, y, vocabulary]

    @staticmethod
    def build_vocab(sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return vocabulary, vocabulary_inv

    @staticmethod
    def build_input_data(sentences, labels, vocabulary):
        """
        Maps sentences and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)

        return [x, y]


if __name__ == '__main__':
    _d = DataLoader('./data/new_train_a+week.csv')
    _, c, v, = _d.x_train, _d.y_train, _d.vocabulary
    print(_)
