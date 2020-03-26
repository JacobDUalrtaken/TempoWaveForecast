# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:05:09 2020

artifical data 粗尝试 xgboost 进行预测

@author: Okan D Lab1
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import xlrd
import xlwt
'''===========================================artificial data==========================================='''



#t = np.arange(1,150,1)
t = np.linspace(1,150,150)

trend_t = 10*(1-np.exp(-0.01*t))          # t= 1:150
x1_t = 0.1*np.multiply(t,np.sin(t))  # t = 1:150
x2_t = 0.01*np.power(t,2) - 0.5*t    # t = 1:150
x3_t = 0.3*np.multiply(t,np.cos(t/5))
x4_t = 1*np.multiply(np.power(t,0.5),np.cos(t/8))

#assume x(t) = trend_x(t-1) + delta_x1(t-1) + delta_x2(t-1)
delta_x1_t = np.diff((x1_t),1)
delta_x1_t_1 =  delta_x1_t[:-1]

delta_x2_t = np.diff((x2_t),1)
delta_x2_t_1 = delta_x2_t[:-1]

delta_x3_t = np.diff((x3_t),1)
delta_x3_t_1 = delta_x3_t[:-1]

trend_t_1 = trend_t[1:-1]

x_t = trend_t_1 + 0.5* delta_x1_t_1 -0.3* delta_x2_t_1 + 0.7*delta_x3_t_1 
+ np.random.normal(0,np.linalg.norm(trend_t),size=np.shape(trend_t_1))  #对应t = 3:150

plt.close('all')
plt.figure( )
plt.subplot(511)
plt.plot(trend_t)
plt.subplot(512)
plt.plot(x1_t)
plt.subplot(513)
plt.plot(x2_t)
plt.subplot(514)
plt.plot(x3_t)
plt.subplot(515)
plt.plot(x_t)

target_ts = x_t
ts1 = x1_t[2:]
ts2 = x2_t[2:]
ts3 = x3_t[2:]
ts4 = x4_t[2:]

#task is to predict target_ts given ts1,ts2,ts3,ts4
# import sys
# sys.path
from JacobTSFlib import temporal_split 
#[x1,y1,x2,y2,x3,y3] = JacobTSFlib.temporal_split(x, 3, 80, 10 ,10 , step = 6)
target_ts.size

train_ls = 124
valid_ls = 12
test_ls = 12

order = 3   #uniform order for all ts
step = 1 

[X_train,y_train,X_valid,y_valid,X_test,y_test] =  temporal_split(target_ts,order,124,12,12)
[X1_train,y1_train,X1_valid,y1_valid,X1_test,y1_test] =  temporal_split(ts1,order,124,12,12)
[X2_train,y2_train,X2_valid,y2_valid,X2_test,y2_test] =  temporal_split(ts2,order,124,12,12)
[X3_train,y3_train,X3_valid,y3_valid,X3_test,y3_test] =  temporal_split(ts3,order,124,12,12)
[X4_train,y4_train,X4_valid,y4_valid,X4_test,y4_test] =  temporal_split(ts4,order,124,12,12)

Feature_train = np.concatenate((X_train, X1_train, X2_train,X3_train,X4_train),axis =1)  #水平拼接
Feature_valid = np.concatenate((X_valid, X1_valid, X2_valid,X3_valid,X4_valid),axis =1)  #水平拼接
Feature_test = np.concatenate((X_test, X1_test, X2_test,X3_test,X4_test),axis =1)  #水平拼接

Response_train = y_train
Response_valid = y_valid
Response_test = y_test


## Preprocessing
from sklearn import preprocessing 

Feature_scaler = preprocessing.StandardScaler().fit(Feature_train)
#Feature_scaler = preprocessing.MinMaxScaler().fit(Feature_train)   #容易受outlier影响
Response_scaler = preprocessing.StandardScaler().fit(np.reshape(Response_train,(-1,1)))

Feature_train_normalized = Feature_scaler.transform(Feature_train)   
#Feature_train_inverse = Feature_scaler.inverse_transform(Feature_train_normalized)
Response_train_normalized = Response_scaler.transform(np.reshape(Response_train,(-1,1)))   

Feature_valid_normalized = Feature_scaler.transform(Feature_valid)   
Response_valid_normalized = Response_scaler.transform(np.reshape(Response_valid,(-1,1)))   

Feature_test_normalized = Feature_scaler.transform(Feature_test)   



## modeling
params = {
    'n_estimators':500,
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'gamma': 0.1,
    'max_depth': 7,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000
}

dtrain = xgb.DMatrix(Feature_train_normalized, Response_train_normalized)
dvalid = xgb.DMatrix(Feature_valid_normalized, Response_valid_normalized)
num_rounds = 1000
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

model_xgb = xgb.train(params, dtrain, num_rounds ,evals =  watchlist, 
                      verbose_eval=10, early_stopping_rounds=50)


plt.figure()
xgb.plot_importance(model_xgb)
plt.show()


'''=====================================tuning and hyperparameter searching======================'''

from itertools import product as product
import time

time_start = time.time()
n_estimators = np.linspace(100,500,5)
gamma = np.linspace(0,0.5,5)
eta = np.linspace(0.01,0.1,5)
max_depth = [3,5,7]
subsample = [0.5,0.75,1.0]
colsample = [0.4,0.6,0.8]
hparam_space = product(n_estimators, gamma,eta,max_depth, subsample, colsample)

Valid_eval = np.array([])
Test_eval = np.array([])
count = 1
total_count = 5*5*5*3*4*3
for i in hparam_space:
    params_gsearch = {
    'n_estimators':i[0],
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'gamma': i[1],
    'eta': i[2],
    'max_depth': i[3],
    'lambda': 1,
    'subsample': i[4],
    'colsample_bytree': i[5],
    'min_child_weight': 3,
    'silent': 1,
    'seed': 1000
    }    
    
    model_gsearch = xgb.train(params_gsearch, dtrain, num_rounds ,evals =  watchlist, 
                      verbose_eval = None, early_stopping_rounds=100)
    
    valid_hat = Response_scaler.inverse_transform(model_gsearch.predict(xgb.DMatrix(Feature_valid_normalized)))
    test_hat = Response_scaler.inverse_transform(model_gsearch.predict(xgb.DMatrix(Feature_test_normalized)))
        
    Valid_eval = np.append(Valid_eval,np.linalg.norm(valid_hat- y_valid,1))
    Test_eval = np.append(Test_eval,np.linalg.norm(test_hat- y_test,1) )
    
    
    if count % 50 == 0:
        print('hyperparameter search progress: {:.2%}'.format(count/total_count))
    count +=1

#np.savetxt('C:/Users/Okan D Lab1/.spyder-py3/2020/time series/hyperparameter_artificial_data.txt',Valid_eval)

 
time_end = time.time()  
time_c= time_end - time_start   #运行所花时间
print('time cost of validation', time_c, 's')

arr = np.arange(1,3000)
np.random.shuffle(arr)
index = arr[:100]
plt.figure()
p1 = plt.plot(np.arange(0,index.size),Valid_eval[index],  color='r')
p2 = plt.plot(np.arange(0,index.size),Test_eval[index],  color='b')
plt.legend([p1,p2],['validation','test'])



## performance 
test_hat = Response_scaler.inverse_transform(model_xgb.predict(xgb.DMatrix(Feature_test_normalized)))



hparam_space = product(n_estimators, gamma,eta,max_depth, subsample, colsample)
best_index = np.argmin(Valid_eval)
best_index = np.argmin(Test_eval)


p_best_cv = list(hparam_space)[best_index]

params_best = {
                'n_estimators':p_best_cv[0],
                'booster': 'gbtree',
                'objective': 'reg:linear',
                'eval_metric': 'mae',
                'gamma': p_best_cv[1],
                'eta': p_best_cv[2],
                'max_depth': p_best_cv[3],
                'lambda': 1,
                'subsample': p_best_cv[4],
                'colsample_bytree': p_best_cv[5],
                'min_child_weight': 3,
                'silent': 1,
                'seed': 1000
                }

model_bestcv = xgb.train(params_best, dtrain, num_rounds ,evals =  watchlist, 
                  verbose_eval = None, early_stopping_rounds=100)
    
test_bestcv = Response_scaler.inverse_transform(model_bestcv.predict(xgb.DMatrix(Feature_test_normalized)))

plt.figure()
p1 = plt.plot(test_hat, 'r', label = 'test')
p2 = plt.plot(test_bestcv, 'g', label ='grid search'  )
p3 = plt.plot(y_test, 'k', label = 'actual')
plt.legend(loc = 'best')
plt.show()


'''============================== with differential features of exogenous variables============='''
target_ts = x_t
ts1 = x1_t[2:]
ts2 = x2_t[2:]
ts3 = x3_t[2:]
ts4 = x4_t[2:]

target_ts.size

train_ls = 124
valid_ls = 12
test_ls = 12

order = 3   #uniform order for all ts
step = 1 

[X_train,y_train,X_valid,y_valid,X_test,y_test] =  temporal_split(target_ts,order,124,12,12,step = 1)

drop_len = order + step -1 # 根据order 和 step 的大小 要去掉train length的一部分数据 来保证test 和 valid set 的大小与给定大小不变

drop_diff_front = order - 2  #去掉开头order-2 个
drop_diff_end = step         #去掉末尾step个
drop_loc = np.concatenate((np.arange(0,order-2 +1),np.arange(-step,-1 +1)))

diff_exg =  np.delete(np.diff(np.vstack((ts1,ts2,ts3))),drop_loc,axis = 1 ).T

diff_exg_train = diff_exg[ :-24 , :]
diff_exg_valid = diff_exg[ -24:-12 , :]
diff_exg_test =  diff_exg[ -12: , :]

Feature_diff_train = np.concatenate((X_train, diff_exg_train),axis =1)  #水平拼接
Feature_diff_valid = np.concatenate((X_valid,diff_exg_valid),axis =1)  #水平拼接
Feature_diff_test = np.concatenate((X_test, diff_exg_test),axis =1)  #水平拼接

Feature_diff_scaler = preprocessing.StandardScaler().fit(Feature_diff_train)
Feature_diff_train_normalized = Feature_diff_scaler.transform(Feature_diff_train)   
Feature_diff_valid_normalized = Feature_diff_scaler.transform(Feature_diff_valid)   
Feature_diff_test_normalized = Feature_diff_scaler.transform(Feature_diff_valid)   

## modeling
params_diff = {
    'n_estimators':1000,
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'gamma': 0.1,
    'max_depth': 7,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.01,
    'seed': 1000
}

dtrain_diff = xgb.DMatrix(Feature_diff_train_normalized, Response_train_normalized)
dvalid_diff = xgb.DMatrix(Feature_diff_valid_normalized, Response_valid_normalized)
num_rounds = 1000
watchlist_diff = [(dtrain_diff, 'train'), (dvalid_diff, 'valid')]

model_xgb_diff = xgb.train(params_diff, dtrain_diff, num_rounds ,evals =  watchlist_diff, 
                      verbose_eval=10, early_stopping_rounds=50)

plt.figure()
xgb.plot_importance(model_xgb_diff)
plt.show()

test_hat_diff = Response_scaler.inverse_transform(model_xgb_diff.predict(xgb.DMatrix(Feature_diff_test_normalized)))
'''====================univariate =================='''
[X_train,y_train,X_valid,y_valid,X_test,y_test] =  temporal_split(target_ts,5,124,12,12,step = 1)
dtrain_uni = xgb.DMatrix(X_train, y_train)
dvalid_uni = xgb.DMatrix(X_valid, y_valid)
num_rounds = 1000
watchlist_uni = [(dtrain_uni, 'train'), (dvalid_uni, 'valid')]

model_xgb_uni = xgb.train(params, dtrain_uni, num_rounds ,evals =  watchlist_uni, 
                      verbose_eval=10, early_stopping_rounds=50)
test_uni = model_xgb_uni.predict(xgb.DMatrix(X_test))





plt.figure()
p1 = plt.plot(test_hat_diff, 'r', label = 'diff without tuning')
p2 = plt.plot(test_hat,'g',label = 'without tuning')
p3 = plt.plot(y_test, 'k', label = 'actual')
p4 = plt.plot(test_uni,label = 'univairate')
plt.legend(loc = 'best')
plt.show()

y_naive = target_ts[-13:-1]

scale = 1/123  *np.linalg.norm( np.diff(target_ts[  : -24 ]) ,1   )
MAE_naive= 1/12*np.linalg.norm(y_naive-y_test, 1)/scale
MAE_multi_untuned = 1/12*np.linalg.norm(test_hat-y_test, 1)/scale
MAE_uni_untuned =1/12* np.linalg.norm(test_uni-y_test, 1) /scale
MAE_diff_multi_untuned = 1/12* np.linalg.norm(test_hat_diff-y_test, 1)/scale 
print('\n MAE of naive: \t',MAE_naive,
      '\n MAE of multivariate untuned:\t',MAE_multi_untuned,  '\n',
      'MAE of univariate untuned:\t',MAE_uni_untuned ,
      '\n MAE of differential multivariate untuned\t',MAE_diff_multi_untuned)



