# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:22:38 2020

proposes of this lib:
    1. basic preprocessing of time series data
    2. dealing with torch.tensor
    3. Metrics in forecasting analytics

function lists





@author: DU LIANG
"""


import pandas as pd
import numpy as np
import statsmodels
from datetime import date,datetime
import torch


def my_diff(x, order):
    if type(x) == torch.Tensor and len(x) > order:
        return torch.from_numpy(np.diff(x,order))
    else:
        print('only tensor can be proprecessed')

def slide_window(X, step):
    '''multivariate sliding window scheme
    assume X is matrix (n x p) ,n is the example number, p is the number of exogneous variables,
    and the last colomn is autoregressive part;
    step is a vector = [k1,k2,...,kp], defines the back step of each exogenous variable      
    
             X0,0   X0,1    X0,2    X0,3     ....   X0,p-1 
             X1,0   X1,1    X1,2    X1,3     ....   X1,p-1 
    X = [    X2,0   X2,1    X2,2    X2,3     ....   X2,p-1    ]
             X3,0   X3,1    X3,2    X3,3     ....   X3,p-1 .
            ...
             Xn-1,0 Xn-1,1  Xn-1,2   Xn-1,3  ....   Xn-1,p-1 
 
    '''
    if type(X) == torch.Tensor:
        n,p = X.size()
        n_real = n - max(step) 
        p_real = sum(step)
        X_real = torch.Tensor([])
        
    for col in range(p):
        feature =  torch.Tensor(n-step[col], step[col])
        response = torch.Tensor(n-step[col],1)
        for i in range(n-step[col]):
            feature[i,:] = X[i:i+step[col], col]     # i:i+1  只有I
            response[i] = X[i+step[col], col]
        
        X_real = torch.cat([X_real,feature[-n_real:,:]],1)
        if col is p-1:
            Y_real = response[-n_real:,:]
    print('feature matrix dimension is',list(X_real.size()))            
    return X_real.float(), Y_real.float()


# a = torch.tensor([3.0,5.0,4.8,5.8,4.0,5.,7.3])
# diff_a =  my_diff(a,2)
# print(diff_a)

# X = torch.randn(100,3)
# n,p = X.size()
# step = np.random.randint(1,5,size = 3, dtype =np.int64) 
# n_real = n - max(step) 
# p_real = sum(step)
# X_real = torch.Tensor([])


# for col in range(p):
#     feature =  torch.Tensor(n-step[col], step[col])
#     response = torch.Tensor(n-step[col],1)
#     for i in range(n-step[col]):
#         feature[i,:] = X[i:i+step[col], col]     # i:i+1  只有I
#         response[i] = X[i+step[col], col]
        
#     X_real = torch.cat([X_real,feature[-n_real:,:]],1)
#     if col is p-1:
#         X_real = torch.cat([X_real,response[-n_real:,:]],1)

# X_my,Y_my = slide_window(X, step)

def temporal_split(x, order, train_ls, valid_ls, test_ls ,step = 1):
    
    ts = np.squeeze(np.reshape(x,(-1,1)))
    assert train_ls + valid_ls + test_ls == ts.shape[0] , "train_size + valid_size + test_size != series_size"
    
    n_rows = ts.shape[0]-order
    n_cols = order
    
    feature_matrix = np.zeros((n_rows,order))
    reponse = ts[order:]
    
    for i in np.arange(order):
#        print(i)
        start = i
        end = start + n_rows   
        feature_matrix[:,i] = ts[start:end] 
        
    if step!=1:
        n_rows = ts.shape[0] - order - step +1 
        n_cols = order 
        feature_matrix = feature_matrix[ :n_rows ,:]
        reponse = reponse[step-1 :   ]
        
    
    X_train = feature_matrix[ :train_ls-order -step +1  ,: ]
    y_train = reponse[:train_ls-order -step +1]
    
    X_valid = feature_matrix[ train_ls-order -step +1: train_ls-order-step +1+valid_ls ,: ]
    y_valid = reponse[ train_ls-order-step +1: train_ls-order -step +1 + valid_ls]
    
    X_test = feature_matrix[ train_ls-order-step +1+valid_ls:   ,: ]
    y_test = reponse[ train_ls-order-step +1 +valid_ls :]

    return X_train,y_train, X_valid,y_valid, X_test, y_test
# x = np.random.randint(1,10,size = (1,100))
# order = 3
# train_ls =  80 
# valid_ls =  10
# test_ls  =  10
# step = 6



# ts = np.squeeze(np.reshape(x,(-1,1)))
# n_rows = ts.shape[0]-order
# n_cols = order

# feature_matrix = np.zeros((n_rows,order))
# reponse = ts[order:]

# for i in np.arange(order):
#     print(i)
#     start = i
#     end = start + n_rows   
#     feature_matrix[:,i] = ts[start:end] 
    
# if step!=1:
#     n_rows = ts.shape[0] - order - step +1 
#     n_cols = order 
#     feature_matrix = feature_matrix[ :n_rows ,:]
#     reponse = reponse[step-1 :   ]
    

# X_train = feature_matrix[ :train_ls-order -step +1  ,: ]
# y_train = reponse[:train_ls-order -step +1]

# X_valid = feature_matrix[ train_ls-order -step +1: train_ls-order-step +1+valid_ls ,: ]
# y_valid = reponse[ train_ls-order-step +1: train_ls-order -step +1 + valid_ls]

# X_test = feature_matrix[ train_ls-order-step +1+valid_ls:   ,: ]
# y_test = reponse[ train_ls-order-step +1 +valid_ls :]

#????
