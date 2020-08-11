# 深度推荐模型

## Bandit算法

**Bandit算法.docx**文档整理了Bandit算法的一些基本情况，同时我也将Thompson sampling算法、Epsilon-Greedy算法以及UCB算法三种算法实现在微服务recommendonline的[lhj_test分支](https://git-cc.nie.netease.com/bigdatams/recommendonline/tree/lhj_test)上。

## DIN算法

### 算法细节

**doc**文件夹下的文档分别记录了深度推荐模型的一些细节以及实现方案。

- **DIN细节.docx**：记录DIN模型的论文技术细节以及计算公式。
- **DIN实现.docx**：记录DIN模型的个人实施方案、技术难点以及相应的实验结果。
- **DIEN细节.docx**：记录DIN模型进化版本——DIEN模型的论文技术细节以及计算公式。

### 代码结构

实验环境基于**python3**进行，需要用到以下依赖。

```
numpy >= 1.16.2
pandas >= 0.24.2
keras >= 2.2.4
scikit-learn >= 0.20.3
lightgbm >= 2.2.1
```

**code**文件夹下各个.py文件的功能如下所示。

- **data_process.py**：存放数据预处理的代码
- **base_structure.py**：存放模型的一些基本结构类以及基本函数
- **model.py**：存放使用的DIN模型类、deepFM模型类以及其改进版本
- **train.py**：存放数据预处理过程与模型训练过程代码

