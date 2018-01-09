# NER-LSTM-CRF
An easy-to-use named entity recognition (NER) toolkit, implemented the LSTM+\[CNN\]+CRF model in tensorflow.

## 1. Model
Bi-LSTM/Bi-GRU + \[CNN\] + CRF，其中CNN层针对英文，捕获字符层面特征，通过参数`use_char_feature`控制self.nil_vars.add(self.feature_weight_dict[feature_name].name)。

## 2. Usage
### 2.1 数据准备
训练数据处理成下列形式，特征之间用制表符(或空格)隔开，每行共n列，1至n-1列为特征，最后一列为label。

    苏   NR   B-ORG
    州   NR   I-ORG
    大   NN   I-ORG
    学   NN   E-ORG
    位   VV   O
    于   VV   O
    江   NR   B-GPE
    苏   NR   I-GPE
    省   NR   E-GPE
    苏   NR   B-GPE
    州   NR   I-GPE
    市   NR   E-GPE
### 2.2 修改配置文件
Step 1: 将上述训练文件的路径写入配置文件`config.yml`中的`data_params/path_train`参数里；

Step 2: 以上样例数据中每行包含三列，分别称为`f1`、`f2`和`label`，首先需要将需要将`model_params/feature_names`设置为`['f1', 'f2']`，并将`embed_params`下的名称改为相应的feature name，其中的`shape`参数需要通过预处理之后才能得到(Step 3)，`path_pre_train`为预训练的词向量路径，格式同gensim生成的txt文件格式；

Step 3: 修改`data_params`下的参数：该参数存放特征和label的voc(即名称到编号id的映射字典)，改为相应的路径。

**注**：处理中文时，将`char_feature`参数设为`false`；处理英文时，设为`true`。

### 2.3 预处理
    $ python/python3 preprocessing.py
预处理后，会得到各个特征的item数以及label数，并自动修改`config.yml`文件中各个feature的`shape`参数，以及`nb_classes`参数；

句子的最大长度参数`sequence_length`由参数`sequence_len_pt`控制，默认为`98`，即计算所得的句子长度`sequence_length`覆盖了98%的实例，可根据实际情况作调整；

需要注意的是，若提供了预训练的embedding向量，则特征embedding的维度以预训练的向量维度为准，若没有提供预训练的向量，则第一列的特征向量维度默认为64，其余特征为32，这里可以根据实际情况进行调整。

### 2.４ 训练模型

训练模型：根据需要调整其余参数，其中dev_size表示开发集占训练集的比例（默认值为0.1），并运行：

    $ python/python3 train.py

### 2.５ 标记数据
标记数据：`config.yml`中修改相应的`path_test`和`path_result`，并运行：

    $ python/python3 test.py

### 2.6 参数说明

|  | 参数 |说明  |
| ------------ | ------------ | ------------ |
|1|rnn_unit| str，\['lstm', 'gru'\]，模型中使用哪种单元，用户设置，默认值`lstm`|
|2|num_units| int，bilstm/bigru单元数，用户设置，默认值`256`|
|3|num_layers| int，bilstm/bigru层数，用户设置，默认值`1`|
|4|rnn_dropout| float，lstm/gru层的dropout值，用户设置，默认值`0.2`|
|5|use_crf| bool，是否使用crf层，用户设置，默认值`true`|
|6|use_char_feature| bool，是否使用字符层面特征（针对英文），用户设置，默认值`false`|
|7|learning_rate| float，学习率，用户设置，默认值`0.001`|
|8|dropout_rate| float，bilstm/bigru输出与全连接层之间，用户设置，默认值`0.5`|
|9|l2_rate| float，加在全连接层权重上，用户设置，默认值`0.001`|
|10|clip:| None or int, 梯度裁剪，用户设置，默认值`10`|
|11|dev_size| float between (0, 1)，训练集中划分出的开发集的比例，shuffle之后再划分，用户设置，默认值`0.2`|
|12|sequence_len_pt|int，句子长度百分数，用于设定句子最大长度，用户设置，默认值`98`|
|13|sequence_length| int，句子最大长度，由参数`sequence_len_pt`计算出，无需设置|
|14|word_len_pt|int，单词长度百分数，用于设定单词最大长度，只有在`use_char_feature`设为`true`时才会使用，用户设置，默认值`95`|
|15|word_length| int，单词最大长度，由参数`word_len_pt`计算出
|16|nb_classes| int，标签数，自动计算，无需设置|
|17|batch_size| int，batch size，用户设置，默认值为`64`|
|18|nb_epoch| int，迭代次数，用户设置|
|19|max_patience| int，最大耐心值，即在开发集上的表现累计max_patience次没有提升时，训练即终止，用户设置，默认值`5`|
|20|path_model| str，模型存放路径，用户设置，默认值值`./Model/best_model`|
|21|sep| str，\['table', 'space'\]，表示特征之间的分隔符，用户设置，默认值`table`|
|22|conv_filter_size_list|list，当使用`char feature`时，卷积核的数量，用户设定，默认值`[8, 8, 8, 8, 8]`|
|23|conv_filter_len_list|list，当使用`char feature`时，卷积核的尺寸，用户设定，默认值`[1, 2, 3, 4, 5]`|
|24|conv_dropout|卷积层的dropout rate|
|25|Other parameters|......|

## 3. Utils
一些小工具，包括：
- train_word2vec_model.py: 利用gensim训练词向量；
- ~~trietree.py: 构建Trie树，并实现查找（待优化），可用于构建字典特征；~~
- [KeywordExtractor](https://github.com/liu-nlper/KeywordExtractor)，trietree.py用KeywordExtractor库代替，提供更多接口；
- updating...

## 4. Requirements
- numpy
- tensorflow 1.4
- pickle
- tqdm
- yaml

## 5. References
- 参考论文：[http://www.aclweb.org/anthology/N16-1030](http://www.aclweb.org/anthology/N16-1030 "http://www.aclweb.org/anthology/N16-1030")
- 参考项目：[https://github.com/koth/kcws](https://github.com/koth/kcws "https://github.com/koth/kcws") ; [https://github.com/chilynn/sequence-labeling](https://github.com/chilynn/sequence-labeling "https://github.com/chilynn/sequence-labeling")

待更新:
  (1) 添加计算转移矩阵的部分；
  (2) 分布式训练。
