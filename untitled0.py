# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:50:29 2020

面向过程
1.
2. univarite forecasting
3. multivarite forecasting without differential features
4. multivariate forecasting with differential features 
5. tuning using grid search 

CMprice 真实数据进行  forecasting 并且封装成函数 

用fossil fuel 做一个例子

@author: Okan D Lab1
"""


import CMprice_dataset
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
from JacobTSFlib import temporal_split

from sklearn import preprocessing 



plt.close('all')




CMprice_dataset.get_CMprice_classname() 

a = CMprice_dataset.get_data_of(1)
a_clean = a.iloc[360:,:]    
a_clean = a_clean.drop(['Coal, Colombian'],axis=1)

missing_where = np.where(a_clean==-1)  # 判断有没有missing value混在其中
assert  np.sum (missing_where[0]) + np.sum(missing_where[1])==0 ,"there are missing values in the dataset"

data_clean = np.array(a_clean.drop(['Date'],axis=1))    
data_clean = data_clean.astype(np.float)  #统一数据类型

'''==================split dataset as Train：valid: test======================'''
train_ls = int(np.round(10/12 * np.shape(data_clean)[0]))
valid_ls = int(np.round(1/12 * np.shape(data_clean)[0]))
test_ls = np.shape(data_clean)[0] - train_ls - valid_ls

# plt.close('all')
# plt.figure( )
# plt.subplot(511)
# plt.plot(data_clean[:,0])
# plt.subplot(512)
# plt.plot(data_clean[:,1])
# plt.subplot(513)
# plt.plot(data_clean[:,2])
# plt.subplot(514)
# plt.plot(data_clean[:,3])
# plt.subplot(515)
# plt.plot(data_clean[:,4])
# plt.figure( )
# plt.subplot(511)
# plt.plot(data_clean[:,5])
# plt.subplot(512)
# plt.plot(data_clean[:,6])
# plt.subplot(513)
# plt.plot(data_clean[:,7])
order = 3
step = 1

Feature_train = np.empty([302-order-step+1,0])
Feature_valid = np.empty([valid_ls,0])
Feature_test = np.empty([test_ls,0])
for i in range(np.shape(data_clean)[1]):
    
    [locals()['X_train_'+str(i)],        locals()['Y_train_'+str(i)],
     locals()['X_valid_'+str(i)],        locals()['Y_valid_'+str(i)],
     locals()['X_test_'+str(i)],         locals()['Y_test_'+str(i)],
    ]                           =  temporal_split(data_clean[:,i],order,train_ls,valid_ls,test_ls, step = step)    
    
    Feature_train = np.append(Feature_train, locals()['X_train_'+str(i)] ,axis = 1)
    Feature_valid = np.append(Feature_valid, locals()['X_valid_'+str(i)], axis = 1)
    Feature_test = np.append(Feature_test, locals()['X_test_'+str(i)],  axis = 1)
 
    
Feature_scaler = preprocessing.StandardScaler().fit(Feature_train)
#Feature_scaler = preprocessing.MinMaxScaler().fit(Feature_train)   #容易受outlier影响

Feature_train_normalized = Feature_scaler.transform(Feature_train)   
#Feature_train_inverse = Feature_scaler.inverse_transform(Feature_train_normalized)
Feature_valid_normalized = Feature_scaler.transform(Feature_valid)   
Feature_test_normalized = Feature_scaler.transform(Feature_test)   

    
for j in range(np.shape(data_clean)[1]):
    Response_train =  locals()['Y_train_'+str(j)]
    Response_valid = locals()['Y_valid_'+str(j)]
    Response_test = locals()['Y_test_'+str(j)]
    '''================================ normalization ==================='''
    Response_scaler = preprocessing.StandardScaler().fit(np.reshape(Response_train,(-1,1)))
    Response_train_normalized = Response_scaler.transform(np.reshape(Response_train,(-1,1)))   
    Response_valid_normalized = Response_scaler.transform(np.reshape(Response_valid,(-1,1)))   
    
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
    test_hat = Response_scaler.inverse_transform(model_xgb.predict(xgb.DMatrix(Feature_test_normalized)))
    
    y_naive = np.insert(Response_test[:-1],0,Response_valid[-1])

    scale =   mean_absolute_error( np.diff(Response_train), np.zeros(np.shape(np.diff(Response_train)))   )
    
    MAE_naive= 1/test_ls*np.linalg.norm(y_naive-Response_test, 1)/scale
    MAE_multi_untuned = 1/test_ls*np.linalg.norm(test_hat-Response_test, 1)/scale


    print(  '\n',   j,'th time'
            '\n MAE of naive: \t',  MAE_naive,
            '\n MAE of multivariate untuned: \t',MAE_multi_untuned
         )

     
    plt.figure(j)
    plt.title("MSE of naive = %.3f ,MSE of multi = %.3f" %(MAE_naive,MAE_multi_untuned))
#    plt.text(0, 10, "MSE of naive = {}".format(MAE_naive), size = 1)
#    plt.text(20, 10, "MSE of multi = {}".format(MAE_multi_untuned), size = 15)
    p1 = plt.plot(y_naive,'r',label = 'naive')
    p2 = plt.plot(test_hat,'g',label = 'without tuning')
    p3 = plt.plot(Response_test,'k', label = 'actual')
    plt.legend(loc = 'best')
    
    plt.show()











    
    # 每一个子序列都可以作为目标量进行预测， 其他序列作为特征，一共有 8 次








