from Configure import Configure
from model_tf import ccDeep
from Feature import Feature_Foramt
import time

root_config_path = 'D:/我的日常/new_rec/tensorflow1.14/data/din/configure_v1/'
# root_data_path = 'D:/我的日常/deep-ctr-algorithms/tensorflow1.14/data/dataset/enc/'
root_data_path = 'D:/我的日常/new_rec/data/'
file_column_path, file_layers_path = root_config_path + 'input_columns.json', root_config_path + 'input_layers.json'
train_path, test_path = root_data_path + 'train.txt', root_data_path + 'test.txt'


# def read_file():
#     config = Configure(file_column_path, file_layers_path)
#     config.feature_parse()
#     config.model_parse()
    
#     s = time.time()
#     train_format = Feature_Foramt(train_path, config.configure)
#     feature_train = train_format.feature_build()
#     test_format = Feature_Foramt(test_path, config.configure)
#     feature_test = test_format.feature_build()
#     e = time.time()
#     print(e-s)

def experiment():
    config = Configure(file_column_path, file_layers_path)
    config.feature_parse()
    config.model_parse(expert_use=False)

    s = time.time()
    train_format = Feature_Foramt(train_path, config.configure)
    feature_train = train_format.feature_build()
    test_format = Feature_Foramt(test_path, config.configure)
    feature_test = test_format.feature_build()
    e = time.time()
    print(e-s)

    model = ccDeep(config.model_configure)
    model.build()
    model.train_and_eval(feature_train, feature_test)

if __name__ == '__main__':
# tf.reset_default_graph()
    experiment()

