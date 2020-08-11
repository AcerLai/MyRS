# 输入数据单个样本格式实例：[[[.2,.3],[.1,.1..1]], [[[2,1.],[4,.3],[5,0]],[[3,.2],[5,0.],[5,0.]],[[1,.3],[2,.4],[5,0]]],[1,.9]]
#                             FixLenDense,                              FixLenKeyValue                              label
# 上格式因效率过低弃用，直接将各个字段的值放到一个list中，变成[[feature1],[feature2],...,[label]]，kvfeat的格式是[[k1,v1],[k2,v2],...]
import time
from collections import deque

class Feature_Foramt(object):

    def __init__(self, file_path, config):
        self.file_path = file_path
        self.config = config
        self.data = []

    def feature_build(self):

        f_name = list(self.config.keys())
        count_dense, count_kv = 0, 0

        for i in f_name:
            if self.config[i]['type'] == 'FixLenDense':
                count_dense += 1
            elif self.config[i]['type'] == 'FixLenKeyValue':
                count_kv += 1

        dense_feature, kv_feature, label_feature = deque([[] for i in range(count_dense)]), deque([[] for i in range(count_kv)]), []

        with open(self.file_path, 'r+') as f:
            lines = f.readlines()
            count = 1
            s = time.time()
            for line in lines:
                label_temp, features = line.strip().split('|')
                label_temp = label_temp.split(':')
                label_temp = [float(label_temp[0]), float(label_temp[1])]
                label_feature.append(label_temp)

                features = features.split(';')
                index, index_dense, index_kv = 0, 0, 0
                for feature in features:
                    feature_list = feature.split(',')
                    if self.config[f_name[index]]['type'] == 'FixLenDense':
                        feature_list = [float(i) for i in feature_list]
                        dense_feature[index_dense].append(feature_list)
                        index_dense += 1
                    if self.config[f_name[index]]['type'] == 'FixLenKeyValue':
                        feature_list = [i.split(':') for i in feature_list]
                        feature_list = [[int(i[0]), float(i[1])] for i in feature_list]
                        kv_feature[index_kv].append(feature_list)
                        index_kv += 1
                    index += 1
                if count % 100000 == 0:
                    e = time.time()
                    print(count, e-s)
                    s = time.time()
                count += 1

        # 少用zip，耗时严重
        # dense_feature = list(zip(*dense_feature))
        # kv_feature = list(zip(*kv_feature))
        # label_feature = list(zip(*label_feature))


        return [dense_feature, kv_feature, label_feature]




