#!/usr/bin/env python
# -*- encoding: utf-8 -*-
__author__ = 'jxliu.nlper@gmail.com'
"""
    训练NER模型
"""
import yaml
import pickle
from load_data import load_vocs, init_data
from model import SequenceLabelingModel


def main():
    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    feature_names = config['model_params']['feature_names']
    use_char_feature = config['model_params']['use_char_feature']

    # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)
    feature_weight_shape_dict, feature_weight_dropout_dict, \
        feature_init_weight_dict = dict(), dict(), dict()
    for feature_name in feature_names:
        feature_weight_shape_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['shape']
        feature_weight_dropout_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['dropout_rate']
        path_pre_train = config['model_params']['embed_params'][feature_name]['path']
        if path_pre_train:
            with open(path_pre_train, 'rb') as file_r:
                feature_init_weight_dict[feature_name] = pickle.load(file_r)
    # char embedding shape
    if use_char_feature:
        feature_weight_shape_dict['char'] = \
            config['model_params']['embed_params']['char']['shape']
        conv_filter_len_list = config['model_params']['conv_filter_len_list']
        conv_filter_size_list = config['model_params']['conv_filter_size_list']
    else:
        conv_filter_len_list = None
        conv_filter_size_list = None

    # 加载数据

    # 加载vocs
    path_vocs = []
    if use_char_feature:
        path_vocs.append(config['data_params']['voc_params']['char']['path'])
    for feature_name in feature_names:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    vocs = load_vocs(path_vocs)

    # 加载训练数据
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    max_len = config['model_params']['sequence_length']
    word_len = config['model_params']['word_length']
    data_dict = init_data(
        path=config['data_params']['path_train'], feature_names=feature_names, sep=sep,
        vocs=vocs, max_len=max_len, model='train', use_char_feature=use_char_feature,
        word_len=word_len)

    # 训练模型
    model = SequenceLabelingModel(
        sequence_length=config['model_params']['sequence_length'],
        nb_classes=config['model_params']['nb_classes'],
        nb_hidden=config['model_params']['bilstm_params']['num_units'],
        num_layers=config['model_params']['bilstm_params']['num_layers'],
        rnn_dropout=config['model_params']['bilstm_params']['rnn_dropout'],
        feature_weight_shape_dict=feature_weight_shape_dict,
        feature_init_weight_dict=feature_init_weight_dict,
        feature_weight_dropout_dict=feature_weight_dropout_dict,
        dropout_rate=config['model_params']['dropout_rate'],
        nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
        batch_size=config['model_params']['batch_size'],
        train_max_patience=config['model_params']['max_patience'],
        use_crf=config['model_params']['use_crf'],
        l2_rate=config['model_params']['l2_rate'],
        rnn_unit=config['model_params']['rnn_unit'],
        learning_rate=config['model_params']['learning_rate'],
        clip=config['model_params']['clip'],
        use_char_feature=use_char_feature,
        conv_filter_size_list=conv_filter_size_list,
        conv_filter_len_list=conv_filter_len_list,
        cnn_dropout_rate=config['model_params']['conv_dropout'],
        word_length=word_len,
        path_model=config['model_params']['path_model'])

    model.fit(
        data_dict=data_dict, dev_size=config['model_params']['dev_size'])


if __name__ == '__main__':
    main()
