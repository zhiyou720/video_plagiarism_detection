#!/usr/bin/env python
# coding: utf-8
"""
@File     :train.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/26
@Desc     : 
"""
import keras
from model import TextAttBiRNN
from data_helper import DataLoader
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split


def predict(x_open_test, y_open_test):
    _model = keras.models.load_model('./model/res.model')

    score = _model.evaluate(x_open_test, y_open_test, batch_size=16)
    print(score)
    print('Test...')

    result = _model.predict(x_open_test)

    def bubble_sort(arr):
        n = len(arr)
        for _i in range(n):
            for j in range(0, n - _i - 1):

                if arr[j] < arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

    res = []
    for item in result:
        item = list(item)
        tmp = item[:]
        bubble_sort(tmp)
        __ = []
        for i in range(len(tmp) - 5):
            __.append(item.index(tmp[i]))
        res.append(__)

    y_true = []

    for _y in y_open_test:
        _ = list(_y).index(1)
        y_true.append(_)

    total = len(x_open_test)

    for rng in range(5):
        score = 0
        for i in range(len(y_true)):
            if y_true[i] in res[i][:rng + 1]:
                score += 1
            if rng == 4:
                print('真实值: {}, 预测值: {}'.format(y_true[i], res[i][:5]))
        print('预测概率前 {} 个数: {}'.format(rng + 1, score / total))
    res_one = []
    for item in result:
        item = list(item)
        res_one.append(item.index(max(item)))

    for i in range(5, 21):
        y_true_x = y_true[:-i]
        _total = len(y_true_x)
        rrd_score = 0
        for j in range(len(y_true_x)):
            if y_true[j] in res_one[j:j + i]:
                rrd_score += 1
        print('如果此次预测的数字出现在了未来{}个就算正确的概率: {}'.format(i, rrd_score / _total))

    return score / total


if __name__ == '__main__':
    train = True
    GPU = False
    batch_size = 32
    embedding_dims = 200
    epochs = 12

    class_num = 2
    all_data_path = './data/all.csv'
    copied_data_path = './data/copied.csv'
    print('Loading data...')

    if train:
        data = DataLoader(all_data_path, copied_data_path, train=True, class_num=class_num)
    else:
        data = DataLoader(all_data_path, copied_data_path, train=False, class_num=class_num)
    x, y, vocab = data.x_train, data.y_train, data.vocabulary

    print('Pad sequences (samples x time)...')

    x_train = keras.preprocessing.sequence.pad_sequences(x, maxlen=None, dtype='int32', padding='pre',
                                                         truncating='pre', value=0.0)

    print('x_train shape:', x_train.shape, x_train.shape[1])

    cx = x_train[:-100]
    cy = y[:-100]

    ox = x_train[-100:]
    oy = y[-100:]

    x_train, x_test, y_train, y_test = train_test_split(cx, cy, test_size=0.01, shuffle=False, random_state=123)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Build model...')
    model = TextAttBiRNN(len(vocab), embedding_dims, maxlen=x_train.shape[1], class_num=class_num, gpu=GPU).get_model()
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    if train:
        print('Train...')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=False,
                  validation_data=(x_test, y_test))

        model.save('./model/res.model')
    else:
        s = predict(ox, oy)
