#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    预处理
"""
import sys
import yaml
import pickle
import codecs
import numpy as np
from collections import defaultdict
from utils import create_dictionary, load_embed_from_txt


def build_vocabulary(path_data, path_vocs_dict, min_counts_dict, columns,
                     sequence_len_pt=98, use_char_featrue=False, word_len_pt=98):
    """
    构建字典
    Args:
        path_data: str, 数据路径
        path_vocs_dict: dict, 字典存放路径
        min_counts_dict: dict, item最少出现次数
        columns: list of str, 每一列的名称
        sequence_len_pt: int，句子长度百分位
        use_char_featrue: bool，是否使用字符特征(针对英文)
        word_len_pt: int，单词长度百分位
    Returns:
        voc_size_1, voc_size_2, ...: int
        sequence_length: 序列最大长度
    """
    print('building vocs...')
    file_data = codecs.open(path_data, 'r', encoding='utf-8')
    line = file_data.readline()

    sequence_length_list = []  # 句子长度
    # 计数items
    feature_item_dict_list = []
    for i in range(len(columns)):
        feature_item_dict_list.append(defaultdict(int))
    # char feature
    if use_char_featrue:
        char_dict = defaultdict(int)
        word_length_list = []  # 单词长度
    sequence_length = 0
    sentence_count = 0  # 句子数
    while line:
        line = line.rstrip()
        if not line:
            sentence_count += 1
            sys.stdout.write('当前处理句子数: %d\r' % sentence_count)
            sys.stdout.flush()
            line = file_data.readline()
            sequence_length_list.append(sequence_length)
            sequence_length = 0
            continue
        items = line.split('\t')
        sequence_length += 1
        # print(items)
        for i in range(len(columns)-1):
            feature_item_dict_list[i][items[i]] += 1
        # label
        feature_item_dict_list[-1][items[-1]] += 1
        # char feature
        if use_char_featrue:
            for c in items[0]:
                char_dict[c] += 1
            word_length_list.append(len(items[0]))
        line = file_data.readline()
    file_data.close()
    # last instance
    if sequence_length != 0:
        sentence_count += 1
        sys.stdout.write('当前处理句子数: %d\r' % sentence_count)
        sequence_length_list.append(sequence_length)
    print()

    # 写入文件
    voc_sizes = []
    if use_char_featrue:  # char feature
        size = create_dictionary(
            char_dict, path_vocs_dict['char'], start=2,
            sort=True, min_count=min_counts_dict['char'], overwrite=True)
        voc_sizes.append(size)
    for i, name in enumerate(columns):
        start = 1 if i == len(columns) - 1 else 2
        size = create_dictionary(
            feature_item_dict_list[i], path_vocs_dict[name], start=start,
            sort=True, min_count=min_counts_dict[name], overwrite=True)
        print('voc: %s, size: %d' % (path_vocs_dict[name], size))
        voc_sizes.append(size)

    print('句子长度分布:')
    sentence_length = -1
    option_len_pt = [90, 95, 98, 100]
    if sequence_len_pt not in option_len_pt:
        option_len_pt.append(sequence_len_pt)
    for per in sorted(option_len_pt):
        tmp = int(np.percentile(sequence_length_list, per))
        if per == sequence_len_pt:
            sentence_length = tmp
            print('%3d percentile: %d (default)' % (per, tmp))
        else:
            print('%3d percentile: %d' % (per, tmp))
    if use_char_featrue:
        print('单词长度分布:')
        word_length = -1
        option_len_pt = [90, 95, 98, 100]
        if word_len_pt not in option_len_pt:
            option_len_pt.append(word_len_pt)
        for per in sorted(option_len_pt):
            tmp = int(np.percentile(word_length_list, per))
            if per == word_len_pt:
                word_length = tmp
                print('%3d percentile: %d (default)' % (per, tmp))
            else:
                print('%3d percentile: %d' % (per, tmp))

    print('done!')
    lengths = [sentence_length]
    if use_char_featrue:
        lengths.append(word_length)
    return voc_sizes, lengths


def main():
    print('preprocessing...')

    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    # 构建字典(同时获取词表size，序列最大长度)
    columns = config['model_params']['feature_names'] + ['label']
    min_counts_dict, path_vocs_dict = defaultdict(int), dict()
    feature_names = config['model_params']['feature_names']
    for feature_name in feature_names:
        min_counts_dict[feature_name] = \
            config['data_params']['voc_params'][feature_name]['min_count']
        path_vocs_dict[feature_name] = \
            config['data_params']['voc_params'][feature_name]['path']
    path_vocs_dict['label'] = \
        config['data_params']['voc_params']['label']['path']

    # char feature
    min_counts_dict['char'] = config['data_params']['voc_params']['char']['min_count']
    path_vocs_dict['char'] = config['data_params']['voc_params']['char']['path']

    sequence_len_pt = config['model_params']['sequence_len_pt']
    use_char_feature = config['model_params']['use_char_feature']
    word_len_pt = config['model_params']['word_len_pt']
    voc_sizes, lengths = build_vocabulary(
        path_data=config['data_params']['path_train'], columns=columns,
        min_counts_dict=min_counts_dict, path_vocs_dict=path_vocs_dict,
        sequence_len_pt=sequence_len_pt, use_char_featrue=use_char_feature,
        word_len_pt=word_len_pt)
    if not use_char_feature:
        sequence_length = lengths[0]
    else:
        sequence_length, word_length = lengths[:]

    # 构建embedding表
    feature_dim_dict = dict()  # 存储每个feature的dim
    for i, feature_name in enumerate(feature_names):
        path_pre_train = config['model_params']['embed_params'][feature_name]['path_pre_train']
        if not path_pre_train:
            if i == 0:
                feature_dim_dict[feature_name] = 64
            else:
                feature_dim_dict[feature_name] = 32
            continue
        path_pkl = config['model_params']['embed_params'][feature_name]['path']
        path_voc = config['data_params']['voc_params'][feature_name]['path']
        with open(path_voc, 'rb') as file_r:
            voc = pickle.load(file_r)
        embedding_dict, vec_dim = load_embed_from_txt(path_pre_train)
        feature_dim_dict[feature_name] = vec_dim
        embedding_matrix = np.zeros((len(voc.keys())+2, vec_dim), dtype='float32')
        for item in voc:
            if item in embedding_dict:
                embedding_matrix[voc[item], :] = embedding_dict[item]
            else:
                embedding_matrix[voc[item], :] = np.random.uniform(-0.25, 0.25, size=(vec_dim))
        with open(path_pkl, 'wb') as file_w:
            pickle.dump(embedding_matrix, file_w)

    # 修改config中各个特征的shape，embedding大小默认为[64, 32, 32, ...]
    if use_char_feature:
        char_voc_size = voc_sizes.pop(0)
    label_size = voc_sizes[-1]
    voc_sizes = voc_sizes[:-1]
    # 修改nb_classes
    config['model_params']['nb_classes'] = label_size
    # 修改embedding表的shape
    for i, feature_name in enumerate(feature_names):
        if i == 0:
            config['model_params']['embed_params'][feature_name]['shape'] = \
                [voc_sizes[i], feature_dim_dict[feature_name]]
        else:
            config['model_params']['embed_params'][feature_name]['shape'] = \
                [voc_sizes[i], feature_dim_dict[feature_name]]
    # 修改char表的embedding
    if use_char_feature:
        # 默认16维，根据任务调整
        config['model_params']['embed_params']['char']['shape'] = \
            [char_voc_size, 16]
        config['model_params']['word_length'] = word_length
    # 修改句子长度
    config['model_params']['sequence_length'] = sequence_length
    # 写入文件
    with codecs.open('./config.yml', 'w', encoding='utf-8') as file_w:
        yaml.dump(config, file_w)

    print('all done!')


if __name__ == '__main__':
    main()
