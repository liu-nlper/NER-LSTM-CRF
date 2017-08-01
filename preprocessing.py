#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    预处理
"""
import yaml
import pickle
import codecs
import numpy as np
from collections import defaultdict
from utils import create_dictionary, load_embed_from_txt


def build_vocabulary(path_data, path_vocs_dict, min_counts_dict, columns):
    """
    构建字典
    Args:
        path_data: str, 数据路径
        path_vocs_dict: dict, 字典存放路径
        min_counts_dict: dict, item最少出现次数
        columns: list of str, 每一列的名称
    Returns:
        voc_size_1, voc_size_2, ...: int
        sequence_length: 序列最大长度
    """
    print('building vocs...')
    file_data = codecs.open(path_data, 'r', encoding='utf-8')
    line = file_data.readline()

    sequence_length_dict = defaultdict(int)  # 句子最大长度
    # 计数items
    feature_item_dict_list = []
    for i in range(len(columns)):
        feature_item_dict_list.append(defaultdict(int))
    sequence_length = 0
    while line:
        line = line.rstrip()
        if not line:
            line = file_data.readline()
            sequence_length_dict[sequence_length] += 1
            sequence_length = 0
            continue
        items = line.split('\t')
        sequence_length += 1
        print(items)
        for i in range(len(items)):
            feature_item_dict_list[i][items[i]] += 1
        line = file_data.readline()
    file_data.close()
    # last instance
    if sequence_length != 0:
        sequence_length_dict[sequence_length] += 1

    # 写入文件
    voc_sizes = []
    for i, name in enumerate(columns):
        size = create_dictionary(
            feature_item_dict_list[i], path_vocs_dict[name], start=1,
            sort=True, min_count=min_counts_dict[name], overwrite=True)
        print('voc: %s, size: %d' % (path_vocs_dict[name], size))
        voc_sizes.append(size)
    print('句子长度分布:')
    print(sorted(sequence_length_dict.items()))
    print('done!')

    return voc_sizes, max(sequence_length_dict.keys())


def main():
    print('proprecessing...')

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
    voc_sizes, sequence_length = build_vocabulary(
        path_data=config['data_params']['path_train'], columns=columns,
        min_counts_dict=min_counts_dict, path_vocs_dict=path_vocs_dict)

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
        embedding_matrix = np.zeros((len(voc.keys())+1, vec_dim), dtype='float32')
        for item in voc:
            if item in embedding_dict:
                embedding_matrix[voc[item], :] = embedding_dict[item]
            else:
                embedding_matrix[voc[item], :] = np.random.uniform(-0.25, 0.25, size=(vec_dim))
        with open(path_pkl, 'wb') as file_w:
            pickle.dump(embedding_matrix, file_w)

    # 修改config中各个特征的shape，embedding大小默认为[64, 32, 32, ...]
    label_size = voc_sizes[-1]
    voc_sizes = voc_sizes[:-1]
    # 修改nb_classes
    config['model_params']['nb_classes'] = label_size + 1
    for i, feature_name in enumerate(feature_names):
        if i == 0:
            config['model_params']['embed_params'][feature_name]['shape'] = \
                [voc_sizes[i]+1, feature_dim_dict[feature_name]]
        else:
            config['model_params']['embed_params'][feature_name]['shape'] = \
                [voc_sizes[i]+1, feature_dim_dict[feature_name]]
    # 写入文件
    with codecs.open('./config.yml', 'w', encoding='utf-8') as file_w:
        yaml.dump(config, file_w)

    print('all done!')


if __name__ == '__main__':
    main()
