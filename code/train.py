from data_rec import data_pro
from modeltry import ccDeep, ccDeepAtt
import numpy as np
import matplotlib.pylab as plt

route_json = "D://我的日常//我的推荐算法//input.json"
route_train = "D://我的日常//我的推荐算法//enc//train.txt"
route_test = "D://我的日常//我的推荐算法//enc//test.txt"

train_feature, train_label, weights_train = data_pro(route_json, route_train)
test_feature, test_label, weights_test = data_pro(route_json, route_test)

### 清空模型
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# from keras import backend as K
 
# K.clear_session()
# tf.reset_default_graph()
###


### AttDNN 未归一化
model = ccDeepAtt(hidden_num=16)
res2 = model.train_model(train_feature, train_label, test_feature, test_label, epoch=10, record_num=3000)
res2.auc_val

### AttDNN 归一化
model = ccDeepAtt(hidden_num=16, norm=True)
res3 = model.train_model(train_feature, train_label, test_feature, test_label, epoch=5, record_num=4000)
res3.auc_val

### AttDNN 归一化, fix
model = ccDeepAtt(hidden_num=16, norm=True, attfix=True)
res4 = model.train_model(train_feature, train_label, test_feature, test_label, epoch=5, record_num=3000)
res4.auc_val

### AttDNN 归一化, nofix，fm
model = ccDeepAtt(hidden_num=16, norm=True, fm=True)
res5 = model.train_model(train_feature, train_label, test_feature, test_label, epoch=5, record_num=5000)
res5.auc_val

### AttDNN 归一化, nofix，lr
model = ccDeepAtt(hidden_num=16, norm=True, lr=True)
res6 = model.train_model(train_feature, train_label, test_feature, test_label, epoch=8, record_num=6000)
res6.auc_val

### AttDNN 归一化, nofix，fm+lr
model = ccDeepAtt(hidden_num=16, norm=True, fm=True, lr=True)
res7 = model.train_model(train_feature, train_label, test_feature, test_label, epoch=6, record_num=9000)
res7.auc_val


train_feature, train_label, weights_train = data_pro(route_json, route_train, dnn=True)
test_feature, test_label, weights_test = data_pro(route_json, route_test, dnn=True)

### DNN vs bnDNN
for bn in [True, False]:
    for i in range(5):
        model = ccDeep(hidden_num=16, bn=bn)
        res = model.train_model(train_feature, train_label, test_feature, test_label, epoch=7, record_num=15000)
        np.savetxt('DNN_BN_' + str(bn) + '-' + str(i) + '.txt', res.auc_val)
        

train_feature, train_label, weights_train = data_pro(route_json, route_train)
test_feature, test_label, weights_test = data_pro(route_json, route_test)

### attDNN vs normattDNN
for bn in [True, False]:
    for i in range(5):
        model = ccDeepAtt(hidden_num=16, norm=bn)
        res = model.train_model(train_feature, train_label, test_feature, test_label, epoch=7, record_num=15000)
        np.savetxt('DNN_normatt_' + str(bn) + '-' + str(i) + '.txt', res.auc_val)

### normattDNN+LR vs normattDNN+LRreg
for bn in [True, False]:
    for i in range(3):
        model = ccDeepAtt(hidden_num=16, norm=True, lr=True, reg=bn)
        res = model.train_model(train_feature, train_label, test_feature, test_label, epoch=7, record_num=15000)
        np.savetxt('DNN_LRReg_' + str(bn) + '-' + str(i) + '.txt', res.auc_val)

### normattDNN+FM vs normattDNN+FMreg
for bn in [True, False]:
    for i in range(3):
        model = ccDeepAtt(hidden_num=16, norm=True, fm=True, reg=bn)
        res = model.train_model(train_feature, train_label, test_feature, test_label, epoch=7, record_num=15000)
        np.savetxt('DNN_FMReg_' + str(bn) + '-' + str(i) + '.txt', res.auc_val)
        
### normattDNN+FM+LR vs normattDNN+FM+LR+reg
for bn in [True, False]:
    for i in range(3):
        model = ccDeepAtt(hidden_num=16, norm=True, lr=True, fm=True, reg=bn)
        res = model.train_model(train_feature, train_label, test_feature, test_label, epoch=7, record_num=15000)
        np.savetxt('DNN_FMLRReg_' + str(bn) + '-' + str(i) + '.txt', res.auc_val)
        

res = [np.zeros(14), np.zeros(14)]
for bn in [True, False]:
    for i in range(3):
        res[int(bn)] += np.loadtxt('DNN_FMLRReg_' + str(bn) + '-' + str(i) + '.txt')
        
res = [i / 3 for i in res]

plt.plot(range(10), res[0][4:], label='without FMLRREG') 
plt.plot(range(10), res[1][4:], label='with FMLRREG')
plt.xlabel('epochs')
plt.ylabel('auc')
plt.legend()
plt.title('The training process of attDNN with FMLRREG or not')        
         
res = [np.zeros(20), np.zeros(14), np.zeros(14), np.zeros(14)]
for i in range(3):
    res[0] += np.loadtxt('DNN_normatt_' + str(True) + '-' + str(i) + '.txt')
    res[1] += np.loadtxt('DNN_LRREG_' + str(False) + '-' + str(i) + '.txt')
    res[2] += np.loadtxt('DNN_FMREG_' + str(False) + '-' + str(i) + '.txt')
    res[3] += np.loadtxt('DNN_FMLRREG_' + str(False) + '-' + str(i) + '.txt')
    
res = [i / 3 for i in res]

plt.plot(range(10), res[0][4:14], label='without LRFM') 
plt.plot(range(10), res[1][4:], label='with LR')
plt.plot(range(10), res[2][4:], label='with FM') 
plt.plot(range(10), res[3][4:], label='with FMLR')
plt.xlabel('epochs')
plt.ylabel('auc')
plt.legend()
plt.title('The training process of attDNN with FMREG/LRREG or not')        
          
   
res = [np.zeros(20), np.zeros(20)]
for i in range(3):
    res[0] += np.loadtxt('DNN_FMRegLR_' + str(i) + '.txt')
    res[1] += np.loadtxt('DNN_FMRegLR1_' + str(i) + '.txt')
res = [i / 3 for i in res]

plt.plot(range(20), res[0], label='without FM') 
plt.plot(range(20), res[1], label='with FM')
plt.xlabel('epochs')
plt.ylabel('auc')
plt.legend()
plt.title('The training process of attDNN with FMREG or not')        
    

res_tmp = [[], []]
for i in [True, False]:
    model = ccDeepAtt(hidden_num=16, norm=True, lr=True, fm=i, reg=True)
    res = model.train_model(train_feature, train_label, test_feature, test_label, epoch=6, record_num=15000)
    res_tmp[int(i)] = res.auc_val

plt.plot(range(len(res_tmp[0][3:])), res_tmp[0][3:], label='no split') 
plt.plot(range(len(res_tmp[0][3:])), res_tmp[1][3:], label='split')
plt.xlabel('epochs')
plt.ylabel('auc')
plt.legend()
plt.title('The training process of attDNN with split or not')        


