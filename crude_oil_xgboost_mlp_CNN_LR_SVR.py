# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:51:14 2020

@author: Okan D Lab1
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:40:13 2020

dataload

@author: Okan D Lab1
"""

import pandas as pd
import numpy as np
#读取工作簿和工作簿中的工作表
import xlrd
import xlwt
import matplotlib.pyplot as plt
from datetime import date,datetime
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import variable
import torch.nn.functional as F
import JacobTSFlib
from torch import nn



plt.close('all')

'''
-----------------------------------------------------------------------------===============================================
'''




crude_oil_dataset =pd.read_excel('C:\\DU LIANG\Phd\dataset\dataset_crude.xlsx')

year_month = np.array(crude_oil_dataset.iloc[2:,0:2], dtype = 'float32')
Target_variable = np.array(crude_oil_dataset.iloc[2:,3],dtype = 'float32')
Explantory_variable  = np.array(crude_oil_dataset.iloc[2:,4:],dtype = 'float32')
feature = np.concatenate( (year_month, Explantory_variable), axis =1 ) 

x_data = torch.from_numpy(feature)
y_data = torch.from_numpy(Target_variable)

batch_data = torch.cat([x_data, torch.unsqueeze(y_data,1)], 1) 
#tep = np.random.randint(1,3,size = 10)
step = np.array([1, 1, 1, 2, 2, 1, 2, 2, 2, 2])
X,Y =  JacobTSFlib.slide_window(batch_data[:,2:],step)
time_info = batch_data[-119:,0:2]
X = torch.cat((time_info, X.type_as(time_info)),1)
    

X_train = X[0:95]
Y_train = Y[0:95]
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
Y_mean = Y_train.mean(axis = 0)
Y_std = Y_train.std(axis = 0)

X_train_norm = (X_train - X_mean)/X_std
Y_train_norm = (Y_train - Y_mean)/Y_std


X_valid = X[95:107, ]
Y_valid = Y[95:107]
X_test =  X[107:, ]
Y_test =  Y[107:]

X_valid_norm = (X_valid - X_mean)/X_std
Y_valid_norm = (Y_valid - Y_mean)/Y_std
X_test_norm = (X_test - X_mean)/X_std
Y_test_norm = (Y_test - Y_mean)/Y_std

'''--------------------------------------------------------------modeling using LR-------------------------------------'''
class LinearRegression(nn.Module):
    def __init__(self,num_input):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_input, 1)  # input and output is 1 dimension
 
    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression(sum(step)+2)        	
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

total_epoch = 1000
for epoch in range(total_epoch):
        #forward
        
        Y_train_hat_norm = model(X_train_norm) # 前向传播
        Y_train_hat = Y_train_hat_norm * Y_std + Y_mean
        
        loss = criterion(Y_train_hat, Y_train) # 计算los
         # backward
        optimizer.zero_grad() # 梯度归零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
         
        if (epoch+1) % 20 == 0:
            print(f'Epoch[{epoch+1}/{total_epoch}], loss: {loss.item():.6f}')
        
        
#模型保存
torch.save(model.state_dict(),'./LinearRegression.model')

#模型评估
model.eval()
Y_valid_hat_norm = model(X_valid_norm)
Y_valid_hat = Y_valid_hat_norm * Y_std + Y_mean
plt.figure( )
plt.plot(np.arange(12), Y_valid_hat.data.numpy(),color = 'r' , label = 'LR')  
plt.plot(np.arange(12), Y_valid.data.numpy(), color = 'b',label = 'Acutal Value')  
plt.legend()
plt.show() 



'''--------------------------------------------- MLP -------------------------------------------------------'''
# 这个神经网络的设计是只有一层隐含层，隐含层神经元个数可随意指定
class Net(torch.nn.Module):
    # Net类的初始化函数
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承父类的初始化函数
        super(Net, self).__init__()
        # 网络的隐藏层创建，名称可以随便起
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden)
        # 输出层(预测层)创建，接收来自隐含层的数据
        self.predict_layer = torch.nn.Linear(n_hidden, n_output)
    
    # 网络的前向传播函数，构造计算图
    def forward(self, x):
        # 用relu函数处理隐含层输出的结果并传给输出层
        hidden_result = self.hidden_layer(x)
        relu_result = F.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return predict_result

total_epoch_MLP = 200
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = sum(step)+2
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
NEURON_NUM = 32
# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
LEARNING_RATE = 0.01

MLP = Net(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
print(MLP)
optimizer = torch.optim.Adam(MLP.parameters(), lr=LEARNING_RATE)
loss_func = torch.nn.MSELoss()

for i in range(total_epoch_MLP):
    # 输入数据进行预测
    Y_train_MLP_norm = MLP(X_train_norm)
    Y_train_MLP = Y_train_MLP_norm * Y_mean + Y_std
    loss = loss_func(Y_train_MLP, Y_train)

    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    # 按照最小loss优化参数
    optimizer.step()
    
    
    if (i+1) % 20 == 0:
        print(f'Epoch[{i+1}/{total_epoch_MLP}], loss: {loss.item():.6f}')
    
    # 可视化训练结果
    # if i % 2 == 0:
    #     # 清空上一次显示结果
    
    #     plt.cla()
    #     # 无误差真值曲线
    #     plt.plot(Y_train.data.numpy(), c='blue', lw='3')
    #     # 实时预测的曲线
    #     plt.plot(Y_train_MLP.data.numpy(), c='red', lw='2')
    #     plt.text(50, 65, 'Time=%d Loss=%.4f' % (i, loss.data.numpy()), fontdict={'size': 15, 'color': 'red'})
    #     plt.pause(0.1)


# 保存整个网络
torch.save(MLP, 'MLP.pkl')
# 只保存网络中节点的参数
torch.save(MLP.state_dict(), 'MLP_params.pkl')


# 直接装载网络
net_restore = torch.load('MLP.pkl')
# 先新建个一模一样的网络，再载入参数
net_rebuild = Net(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
net_rebuild.load_state_dict(torch.load('MLP_params.pkl'))


#模型评估
MLP.eval()
Y_valid_MLP_norm = MLP(X_valid_norm)
Y_valid_MLP = Y_valid_MLP_norm * Y_std + Y_mean

plt.figure( )
plt.plot(np.arange(12), Y_valid_MLP.data.numpy(),color = 'r' , label = 'MLP')  
plt.plot(np.arange(12), Y_valid.data.numpy(), color = 'b',label = 'Acutal Value')  
plt.legend()
plt.show() 

'''----------------------------------------------------1D-CNN-----------------------------------------------'''
class CNN_Series(nn.Module):
    def __init__(self):
        super (CNN_Series, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 3),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels =16, out_channels = 32, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2))
            
        self.fc =nn.Linear(3*32, 1, bias=True)           
                
    
    def forward(self, indata):
        # Max pooling over a (2, 2) window
        x = self.conv1(indata)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out
    
CNN = CNN_Series()
lossfunc_cnn = nn.MSELoss()
optimizer_cnn = torch.optim.Adam(CNN.parameters(),lr= 0.001)

total_epoch_CNN = 500
for i in range(total_epoch_CNN):
    # 输入数据进行预测
    
    Y_train_CNN_norm = CNN(X_train_norm.unsqueeze(1))  
    
    Y_train_CNN = Y_train_CNN_norm * Y_mean + Y_std
    
    loss = lossfunc_cnn(Y_train_CNN,Y_train)

    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer_cnn.zero_grad()
    # 误差反向传播
    loss.backward()
    # 按照最小loss优化参数
    optimizer_cnn.step()
        # 可视化训练结果
    if (i+1) % 20 == 0:
            print(f'Epoch[{i+1}/{total_epoch_CNN}], loss: {loss.item():.6f}')
        # 可视化训练结果
    # if i % 2 == 0:
    #     # 清空上一次显示结果
    
    #     plt.cla()
    #     # 无误差真值曲线
    #     plt.plot(Y_train.data.numpy(), c='blue', lw='3')
    #     # 实时预测的曲线
    #     plt.plot(Y_train_CNN.data.numpy(), c='red', lw='2')
    #     plt.text(50, 65, 'Time=%d Loss=%.4f' % (i, loss.data.numpy()), fontdict={'size': 15, 'color': 'red'})
    #     plt.pause(0.1)
    
# # 保存整个网络
# torch.save(CNN, 'CNN.pkl')
# # 只保存网络中节点的参数
# torch.save(CNN.state_dict(), 'CNN_params.pkl')


# # 直接装载网络
# net_restore = torch.load('CNN.pkl')
# # 先新建个一模一样的网络，再载入参数
# CNN_rebuild = CNN_Series()
# CNN_rebuild.load_state_dict(torch.load('CNN_params.pkl'))


#模型评估
CNN.eval()
Y_valid_CNN_norm = CNN(X_valid_norm.unsqueeze(1))
Y_valid_CNN = Y_valid_CNN_norm * Y_std + Y_mean

plt.figure( )
plt.plot(np.arange(12), Y_valid_CNN.data.numpy(),color = 'r' , label = 'CNN')  
plt.plot(np.arange(12), Y_valid.data.numpy(), color = 'b',label = 'Acutal Value')  
plt.legend()
plt.show() 



'''------------------------------------------------------LSTM ----------------------------------------------------'''
class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


lstm = LSTM_Regression(X_train_norm.size(1), 8, output_size=1, num_layers=2)
lossfc_lstm = nn.MSELoss()
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=1e-2)

total_epoch_lstm = 1000;
for i in range(total_epoch_lstm):                   
      Y_train_lstm_norm = lstm(X_train_norm.unsqueeze(1))
      
      Y_train_lstm = Y_train_lstm_norm.squeeze(1)* Y_std + Y_mean
      loss_lstm = lossfc_lstm(Y_train_lstm, Y_train)
      
      optimizer_lstm.zero_grad()
      loss_lstm.backward()
      optimizer_lstm.step()
      

      if (i+1) % 100 == 0:
          print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss_lstm.item()))

      # if i % 2 == 0:
      #    # 清空上一次显示结果
         
      #    plt.cla()
      #    # 无误差真值曲线
      #    plt.plot(Y_train.data.numpy(), c='blue', lw='3')
      #    # 实时预测的曲线
      #    plt.plot(Y_train_lstm.squeeze().data.numpy(), c='red', lw='2')
      #    plt.text(50, 65, 'Time=%d Loss=%.4f' % (i, loss_lstm.data.numpy()), fontdict={'size': 15, 'color': 'red'})
      #    plt.pause(0.1)

# 保存整个网络
torch.save(lstm, 'lstm.pkl')
# 只保存网络中节点的参数
torch.save(lstm.state_dict(), 'lstm_params.pkl')


# 直接装载网络
net_restore = torch.load('lstm.pkl')
# 先新建个一模一样的网络，再载入参数
# net_rebuild = lstm(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
# net_rebuild.load_state_dict(torch.load('lstm_params.pkl'))


#模型评估
lstm.eval()
Y_valid_lstm_norm = lstm(X_valid_norm.unsqueeze(1))
Y_valid_lstm = Y_valid_lstm_norm.squeeze(1) * Y_std + Y_mean

plt.figure( )
plt.plot(np.arange(12), Y_valid_lstm.data.numpy(),color = 'r' , label = 'lstm')  
plt.plot(np.arange(12), Y_valid.data.numpy(), color = 'b',label = 'Acutal Value')  
plt.legend()
plt.show() 

'''============================================== Xgboost  ============================================== '''

 








