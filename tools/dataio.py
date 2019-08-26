#!/usr/bin/env python
# coding: utf-8
"""
@File     :dataio.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""


def load_txt_data(path):
    """
    This func is used to reading txt file
    :param path: path where file stored
    :type path: str
    :return: string lines in file in a list
    :rtype: list
    """
    if type(path) != str:
        raise TypeError
    res = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res.append(line.strip())

    return res


def save_txt_file(data, path, end='\n'):
    """
    This func is used to saving data to txt file
    support data type:
    list: Fully support
    dict: Only save dict key
    str: will save single char to each line
    tuple: Fully support
    set: Fully support
    :param data: data
    :param path: path to save
    :type path: str
    :param end:
    :type end: str
    :return: None
    """
    if type(data) not in [list, dict, str, tuple, set] or type(path) != str:
        raise TypeError

    with open(path, 'a', encoding='utf-8') as f:
        for item in data:
            if not item:
                f.write(end)
            else:
                f.write(item + end)
