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




'''MLP 自定义
'''
class Net(torch.nn.Module):
    # Net类的初始化函数
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承父类的初始化函数
        super().__init__()
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
    
# 训练次数
TRAIN_TIMES = 300
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 1
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
NEURON_NUM = 32
# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
LEARNING_RATE = 0.1

x_data = torch.unsqueeze(torch.linspace(-4, 4, 80), dim=1)
# randn函数用于生成服从正态分布的随机数
y_data = x_data.pow(3) + 3 * torch.randn(x_data.size())
y_data_real = x_data.pow(3)

# 建立网络
net = Net(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
print(net)

# 训练网络
# 这里也可以使用其它的优化方法
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
# 定义一个误差计算方法
loss_func = torch.nn.MSELoss()

for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    # 计算预测值与真值误差，注意参数顺序问题
    # 第一个参数为预测值，第二个为真值
    loss = loss_func(prediction, y_data)

    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    # 按照最小loss优化参数
    optimizer.step()

    # 可视化训练结果
    if i % 2 == 0:
        # 清空上一次显示结果
        plt.cla()
        # 无误差真值曲线
        plt.plot(x_data.numpy(), y_data_real.numpy(), c='blue', lw='3')
        # 有误差散点
        plt.scatter(x_data.numpy(), y_data.numpy(), c='orange')
        # 实时预测的曲线
        plt.plot(x_data.numpy(), prediction.data.numpy(), c='red', lw='2')
        plt.text(-0.5, -65, 'Time=%d Loss=%.4f' % (i, loss.data.numpy()), fontdict={'size': 15, 'color': 'red'})
        plt.pause(0.1)


# 保存整个网络
torch.save(net, 'net.pkl')
# 只保存网络中节点的参数
torch.save(net.state_dict(), 'net_params.pkl')



# 直接装载网络
net_restore = torch.load('net.pkl')
# 先新建个一模一样的网络，再载入参数
net_rebuild = Net(n_feature=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM)
net_rebuild.load_state_dict(torch.load('net_params.pkl'))

##快速prototye
net2 = torch.nn.Sequential(
    torch.nn.Linear(INPUT_FEATURE_DIM, NEURON_NUM),
    torch.nn.ReLU(),
    torch.nn.Linear(NEURON_NUM, OUTPUT_FEATURE_DIM)
)
print(net2)



'''
-----------------------------------------------------------------------------===============================================
'''




crude_oil_dataset =pd.read_excel('C:\\DU LIANG\Phd\dataset\dataset_crude.xlsx')

year_month = np.array(crude_oil_dataset.iloc[2:,0:2], dtype = 'float64')
Target_variable = np.array(crude_oil_dataset.iloc[2:,3],dtype = 'float64')
Explantory_variable  = np.array(crude_oil_dataset.iloc[2:,4:],dtype = 'float64')
feature = np.concatenate( (year_month, Explantory_variable), axis =1 ) 

x_data = torch.from_numpy(feature)
y_data = torch.from_numpy(Target_variable)

print(x_data.size(), y_data.size())

'''
Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。

DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)

现在，我们先展示直接使用 TensorDataset 来将数据包装成Dataset类

'''
deal_dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(dataset = deal_dataset,
                          batch_size = 121,
                          shuffle = False,
                          num_workers = 0
                                                  
                          )
total_epoach = 5

for epoach in range(total_epoach):
    for i,(batch_x, batch_y) in enumerate(train_loader) :
        # 将数据从train_loader 中读出来，一次取batch_size的样本数

        
        # 将这些数据转换成variable 类型
        #inputs, labels = Variable(inputs), Variable(labels)
        
        
        
        
        
        
        
        
        #训练模型
        print('Epoach:',epoach, '|batch_No:',i, 'batch_x:',batch_x)
'''
刚才是用自带的TensorDataset 来封装数据，现在自己写一个继承Dataset的类,只需要重写__len__ 和
__getitem__两个函数


class myDataset(Dataset):
    def __init__(self,csv_file,txt_file,root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file,'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv_data)

    def __gettime__(self,idx):
        data = (self.csv_data[idx],self.txt_data[idx])
        return data

'''










