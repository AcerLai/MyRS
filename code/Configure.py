import json

class Configure(object):

    def __init__(self, file_column, file_layers):
        self.file_column = file_column
        self.file_layers = file_layers
        self.configure = {}
        self.model_configure = {}

    # feature情况：类型、长度、种类数...
    def feature_parse(self):
        with open(self.file_column, 'r+') as f:
            cons_c = f.read()
        cons_c = json.loads(cons_c.strip()).get('inputs')

        for con in cons_c:
            column = con['column']
            con.pop('column')
            self.configure[column] = con


    def attention_parse(self):
        pass

    # 模型情况：feature情况、特征共享情况、pooling方式、是否bn...
    def model_parse(self, bn=False, label_weighted=False, expert_use=False, norm="softamx"):
        with open(self.file_layers, 'r+') as f:
            cons_l = f.read()
        cons_l = json.loads(cons_l.strip())

        for con in cons_l['deep']['embedding']:
            self.configure[con['column']]['embedding_dim'] = con['embedding_dim']
            self.configure[con['column']]['combine'] = con['combine']
            if con.get('share'):
                self.configure[con['column']]['share'] = con['share']
            else:
                self.configure[con['column']]['share'] = con['column']

        for i in self.configure.keys():
            if not self.configure[i].get('embedding_dim'):
                self.configure[i]['embedding_dim'] = 8
                self.configure[i]['combine'] = 'sum'
                self.configure[i]['share'] = i

        self.model_configure['feature'] = self.configure
        self.model_configure['label'] = {'weight':label_weighted}

        # Expert Network
        # norm方式：无、softmax、求和
        self.model_configure['expert'] = {}
        if expert_use:
            self.model_configure['expert']['expert_num'] = 5
            self.model_configure['expert']['expert_layer'] = [256, 128, 64]
            self.model_configure['expert']['gate_layer'] = [256, 64]
            self.model_configure['expert']['gate_norm'] = norm

        # LR



        # FM


        # bn
        self.model_configure['bn'] = bn
        self.model_configure['dense_size'] = [512, 256, 128]
        self.model_configure['l1_reg'] = 0.0001
        self.model_configure['l2_reg'] = 0.0001




