# NER-LSTM-CRF
An easy-to-use named entity recognition (NER) toolkit, implemented the LSTM+CRF model in tensorflow.

## 1. Model
Bi-LSTM/Bi-GRU + CRF.

## 2. Usage
### 2.1 数据准备
训练数据处理成下列形式，特征之间用制表符'\t'隔开，每行共n列，1至n-1列为特征，最后一列为label。

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
- Step 1: 将上述训练文件的路径写入配置文件`config.yml`中的`data_params／path_train`参数里；
- Step 2: 以上样例数据中每行包含三列，分别称为`f1`、`f2`和`label`，首先需要将需要将`model_params/feature_names`设置为`['f1', 'f2']`，并将`embed_params`下的名称改为相应的feature name，其中的`shape`参数需要通过预处理之后才能得带(Step 3)，`path_pre_train`为预训练的词向量路径，格式同gensim生成的txt文件格式；
- Step 3: 修改`data_params`下的参数：该参数存放特征和label的voc(即名称到编号id的映射字典)，改为相应的路径；

### 2.3 预处理
- 运行`preprocessing.py`文件进行预处理，会得到各个特征的item数以及label数，并修改步骤(2)中的`shape`参数，要注意的是表的大小需要比计算得出的值大1，因为id:0在模型中表示的是padding值。

### 2.４ 训练模型
- 训练模型：调整其余参数，其中dev_size表示开发集占训练集的比例，并运行：`python3 train.py`。

### 2.５ 标记数据
- 标记数据：`config.yml`中修改相应的`path_test`和`path_result`，并运行`python3 test.py`。

### 2.6 参数说明
- model_params/bilstm_params:


```python
    num_units: bilstm/bigru单元数，默认256;
    num_layers: bilstm/bigru层数；
    use_crf: 是否使用crf层；
    rnn_unit: lstm or gru，模型中使用哪种单元；
    learning_rate: 学习率；
    dev_size: 训练集中划分出的开发集的比例，shuffle之后再划分；
    dropout_rate: bilstm/bigru输出与全连接层之间；
    l2_rate: 加在全连接层权重上；
    nb_classes: 标签数（preprocessing.py输出结果上+1）;
    sequence_length: 句子最大长度（preprocessing.py输出结果中有句子分布）；
    batch_size: batch size；
    nb_epoch: 迭代次数；
    max_patience: 最大耐心值，即在开发集上的表现累计max_patience次没有提升时，训练即终止；
    path_model: 模型存放路径；
    some other parameters...
```

## 3. Requirements
- python3
- numpy
- tensorflow 1.2
- pickle
- tqdm
- yaml

## 4. Reference
- 参考论文：[http://www.aclweb.org/anthology/N16-1030](http://www.aclweb.org/anthology/N16-1030 "http://www.aclweb.org/anthology/N16-1030")
- 参考项目：[https://github.com/koth/kcws](https://github.com/koth/kcws "https://github.com/koth/kcws") ; [https://github.com/chilynn/sequence-labeling](https://github.com/chilynn/sequence-labeling "https://github.com/chilynn/sequence-labeling")

Updating...
