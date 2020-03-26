# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:21:18 2020
pytorch 基本的tensor操作 和 numpy数组转换
@author: Okan D Lab1
"""

from __future__ import print_function
import torch
from torch import autograd

# 生成tensor的变量
x1 = torch.empty(5, 3)
print(x1)

x2 = torch.rand(5, 3)
print(x2)

x3 = torch.zeros(5, 3, dtype = torch.long)
print(x3)

x4 = torch.tensor([5.5000,5.000])
print(x4)

# 根据已有的tensor 来生成新的

x5 =  x1.new_ones(5,3, dtype = torch.double)
print(x5)

x6 = torch.randn_like(x5, dtype = torch.float)
print(x6)

#获取tensor的形状
size_x1 = x1.size()   #本质上是tuple

# 加法
print(x1 + x2)
print(torch.add(x1,x2))
print(x2.add_(x1)) #结果会改变x2
# 索引
print(x5[1,1])
# 改变形状
x7 = torch.randn(4, 4)
y = x7.view(16)
z = x7.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x7.size(), y.size(), z.size())

#把一个tensor 变成 numpy 数组 但是底层是用同一个存储区域，所以其中一个发生改变时 另一个也改变
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# array -> tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


#autograd

x = torch.tensor(1.)    #x 赋值为1 
a = torch.tensor(1., requires_grad = True)
b = torch.tensor(2., requires_grad = True)
c = torch.tensor(3., requires_grad = True)

y = a**2 * x + b*x + c

print('before:',a.grad, b.grad, c.grad )
grads = autograd.grad(y, [a,b,c])
print('after:', grads[0], grads[1],grads[2])



