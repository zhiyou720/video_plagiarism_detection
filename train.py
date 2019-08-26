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


def predict(x_open_test, y_open_test, _vocab):
    _model = keras.models.load_model('./model/res.model')

    i2w_dict = dict(zip(_vocab.values(), _vocab.keys()))

    score = _model.evaluate(x_open_test, y_open_test, batch_size=16)
    print(score)
    print('Test...')
    label_dict = {0: 'origin', 1: 'copied'}
    result = _model.predict(x_open_test)
    score = 0
    total = len(x_open_test)
    for i in range(len(result)):
        true_label = list(y_open_test[i]).index(1)
        predict_label = list(result[i]).index(max(result[i]))
        sent = ''.join([i2w_dict[w] for w in x_open_test[i]])
        print("{}\n真实标签: {}, 预测标签: {}".format(sent, label_dict[true_label], label_dict[predict_label]))
        if true_label == predict_label:
            score += 1

    return score / total


if __name__ == '__main__':
    train = False
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
        s = predict(ox, oy, vocab)
