# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:11:46 2020
%% univariate 
@author: Okan D Lab1
"""

import pandas as pd
import numpy as np
#读取工作簿和工作簿中的工作表
import xlrd
import xlwt
import matplotlib.pyplot as plt
from datetime import date,datetime

crude_oil_dataset =pd.read_excel('C:\\DU LIANG\Phd\dataset\dataset_crude.xlsx')

year_month = np.array(crude_oil_dataset.iloc[2:,0:2])
Target_variable = np.array(crude_oil_dataset.iloc[2:,3])
Explantory_variable  = np.array(crude_oil_dataset.iloc[2:,4:])


#%matplotlib qt5 
plt.subplot(3,3,1)
plt.scatter(Explantory_variable[:,0],Target_variable, color = 'k')
plt.xlabel('CPI')
plt.subplot(3,3,2)
plt.scatter(Explantory_variable[:,1],Target_variable, color = 'k')
plt.xlabel('IPI')
plt.subplot(3,3,3)
plt.scatter(Explantory_variable[:,2],Target_variable, color = 'k')
plt.xlabel('UOI')
plt.subplot(3,3,4)
plt.scatter(Explantory_variable[:,3],Target_variable, color = 'k')
plt.xlabel('BEDTI')
plt.subplot(3,3,5)
plt.scatter(Explantory_variable[:,4],Target_variable, color = 'k')
plt.xlabel('CU')
plt.subplot(3,3,6)
plt.scatter(Explantory_variable[:,5],Target_variable, color = 'k')
plt.xlabel('LUS')
plt.subplot(3,3,7)
plt.scatter(Explantory_variable[:,6],Target_variable, color = 'k')
plt.xlabel('SP500')
plt.subplot(3,3,8)
plt.scatter(Explantory_variable[:,7],Target_variable, color = 'k')
plt.xlabel('DEJPUS')
plt.subplot(3,3,9)
plt.scatter(Explantory_variable[:,8],Target_variable, color = 'k')
plt.xlabel('DEXCHUS')