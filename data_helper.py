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
    def __init__(self, all_path, copied_path, class_num=10, train=False,
                 vocab_path='./model/vocab_config.pkl'):
        self.all_path = all_path
        self.copied_path = copied_path
        self.vocab_path = vocab_path

        self.class_num = class_num

        self.train = train
        self.x_train, self.y_train, self.vocabulary = self.load_data()

    def build_seq_data_set(self):
        """
        Load raw data and give x_train and y_train
        :return:
        """
        raw_data = load_txt_data(self.all_path, encoding='gbk')

        copied_data = load_txt_data(self.copied_path, encoding='gbk')

        copied_stamp_book = [(x.split('\t')[0] + x.split('\t')[1]) for x in copied_data]

        x_train = []
        y_train = []  # [0, 1] copied, [1, 0] not copied
        count = 0
        for item in raw_data:
            stack = item.split('\t')
            topic = stack[0]  # 标题
            source = stack[2]  # 源网站的名字
            terminal = stack[3]  # 终端
            length = [stack[4] if stack[4] else 'unk'][0]  # 时长
            address = stack[1]
            label = [0 for x in range(self.class_num)]
            if (topic + address) in copied_stamp_book:
                count += 1
                label[1] = 1
            else:
                label[0] = 1
            y_train.append(label)
            x_train.append('{}, 视频来源网站: {}, 终端: {} 时长: {}'.format(topic, source, terminal, length))
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
    _d = DataLoader('./data/all.csv', './data/copied.csv', train=True)
    _, c, v, = _d.x_train, _d.y_train, _d.vocabulary
    print(_)
