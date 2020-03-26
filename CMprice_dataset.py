# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:44:20 2020

Data file for paper2 :Decomposition-based embedding in multivariate forecasting for fossil fuel price 

@author: Okan D Lab1
"""
#import keras

import pandas as pd
import numpy as np
#读取工作簿和工作簿中的工作表
import xlrd
import xlwt
import matplotlib.pyplot as plt
from datetime import date,datetime

CMprice_dataset = pd.read_excel('C:\\DU LIANG\\Phd\\research\\publication\\2020.3\\dataset\\CMOHistoricalDataMonthly.xlsx',
                                sheet_name = "Monthly Prices", #返回第一张表
                                header = [4],     #指定某一行作为属性
                               skiprows = [5],   #省略指定行的数据
                                #skip_footer =      #省略从尾部数的行数据
                                #index_col = 0       #指定列为索引列，也可以使用 u’string’
                                na_values = '..'
                                )
                                
                                
                                
CMprice = CMprice_dataset.fillna(-1)   # 用-1 更改所有的Na值
CMprice.rename(columns={'Unnamed: 0':'Date'}, inplace = True)
CMprice = CMprice.drop([0])

# 数据分组*10   能源, 饮料，油， 粮食，非粮食类食物， 木材，工业原料，化肥，金属价格 贵金属价格
Fossil_fuel = CMprice.loc[ :,['Date', 'Crude oil, Brent', 'Crude oil, Dubai',
                              'Crude oil, WTI', 'Coal, Australian', 'Coal, South African',
                              'Natural gas, US', 'Natural gas, Europe','Coal, Colombian',
                              'Liquefied natural gas, Japan']]

Beverage = CMprice.loc[ :,['Date', 'Cocoa','Coffee, Arabica', 'Coffee, Robusta', 
                           'Tea, avg 3 auctions','Tea, Colombo', 'Tea, Kolkata',
                           'Tea, Mombasa']]

Oil = CMprice.loc[ :,['Date','Coconut oil','Groundnuts', 'Fish meal', 
                      'Groundnut oil', 'Palm oil','Palm kernel oil', 
                      'Soybeans', 'Soybean oil', 'Soybean meal','Copra',
                      'Rapeseed oil', 'Sunflower oil']]

Grain = CMprice.loc[ :,['Date','Barley', 'Maize', 'Sorghum','Rice, Thai 5% ',
                        'Rice, Thai 25% ', 'Rice, Thai A.1','Rice, Viet Namese 5%', 
                        'Wheat, US SRW', 'Wheat, US HRW','Wheat, Canadian']]

Food = CMprice.loc[:,['Date','Banana, Europe', 'Banana, US', 'Orange', 'Beef',
                      'Meat, chicken','Meat, sheep', 'Shrimps, Mexican', 
                      'Sugar, EU', 'Sugar, US','Sugar, world']]      

Timber = CMprice.loc[:,['Date','Logs, Cameroon','Logs, Malaysian', 
                        'Sawnwood, Cameroon', 'Sawnwood, Malaysian',
                        'Plywood', 'Woodpulp']]

Other_Raw = CMprice.loc[:,['Date','Cotton, A Index', 'Rubber, TSR20',
                           'Cotton, Memphis','Rubber, US',
                           'Rubber, SGP/MYS']]
                     
Fertilizer = CMprice.loc[:,['Date','Phosphate rock', 'DAP', 'TSP', 'Urea ',
                            'Potassium chloride']]   

Metal = CMprice.loc[:,['Date', 'Aluminum', 'Iron ore, cfr spot', 'Copper',
                       'Lead', 'Tin', 'Nickel', 'Zinc',
                       'Steel, cold rolled coilsheet', 
                       'Steel, hot rolled coilsheet',
                       'Steel rebar', 
                       'Steel wire rod']]

Precious_Metal = CMprice.loc[:,['Date','Gold','Platinum','Silver']]

 

def get_CMprice_classname ( ):
    print(
            '\n Price of Fossil Fuel        \t 1'     ,   
            '\n Price of Beverage           \t 2'     ,   
            '\n Price of oil                \t 3'     ,   
            '\n Price of Grain              \t 4'     ,   
            '\n Price of Food               \t 5'     ,   
            '\n Price of Timber             \t 6'     ,   
            '\n Price of Raw material       \t 7'     ,   
            '\n Price of Fertilizer         \t 8'     ,   
            '\n Price of Metal              \t 9'     ,   
            '\n Price of Precious_Metal     \t 10'      
        )   
    return  

def get_data_of (index):
    CMprice_dataset = pd.read_excel('C:\\DU LIANG\\Phd\\research\\publication\\2020.3\\dataset\\CMOHistoricalDataMonthly.xlsx',
                                sheet_name = "Monthly Prices", #返回第一张表
                                header = [4],     #指定某一行作为属性
                               skiprows = [5],   #省略指定行的数据
                                #skip_footer =      #省略从尾部数的行数据
                                #index_col = 0       #指定列为索引列，也可以使用 u’string’
                                na_values = '..'
                                )
    CMprice = CMprice_dataset.fillna(-1)   # 用-1 更改所有的Na值
    CMprice.rename(columns={'Unnamed: 0':'Date'}, inplace = True)
    CMprice = CMprice.drop([0])
    
    # 数据分组*10   能源, 饮料，油， 粮食，非粮食类食物， 木材，工业原料，化肥，金属价格 贵金属价格
    Fossil_fuel = CMprice.loc[ :,['Date', 'Crude oil, Brent', 'Crude oil, Dubai',
                                  'Crude oil, WTI', 'Coal, Australian', 'Coal, South African',
                                  'Natural gas, US', 'Natural gas, Europe','Coal, Colombian',
                                  'Liquefied natural gas, Japan']]
    
    Beverage = CMprice.loc[ :,['Date', 'Cocoa','Coffee, Arabica', 'Coffee, Robusta', 
                               'Tea, avg 3 auctions','Tea, Colombo', 'Tea, Kolkata',
                               'Tea, Mombasa']]
    
    Oil = CMprice.loc[ :,['Date','Coconut oil','Groundnuts', 'Fish meal', 
                          'Groundnut oil', 'Palm oil','Palm kernel oil', 
                          'Soybeans', 'Soybean oil', 'Soybean meal','Copra',
                          'Rapeseed oil', 'Sunflower oil']]
    
    Grain = CMprice.loc[ :,['Date','Barley', 'Maize', 'Sorghum','Rice, Thai 5% ',
                            'Rice, Thai 25% ', 'Rice, Thai A.1','Rice, Viet Namese 5%', 
                            'Wheat, US SRW', 'Wheat, US HRW','Wheat, Canadian']]
    
    Food = CMprice.loc[:,['Date','Banana, Europe', 'Banana, US', 'Orange', 'Beef',
                          'Meat, chicken','Meat, sheep', 'Shrimps, Mexican', 
                          'Sugar, EU', 'Sugar, US','Sugar, world']]      
    
    Timber = CMprice.loc[:,['Date','Logs, Cameroon','Logs, Malaysian', 
                            'Sawnwood, Cameroon', 'Sawnwood, Malaysian',
                            'Plywood', 'Woodpulp']]
    
    Other_Raw = CMprice.loc[:,['Date','Cotton, A Index', 'Rubber, TSR20',
                               'Cotton, Memphis','Rubber, US',
                               'Rubber, SGP/MYS']]
                         
    Fertilizer = CMprice.loc[:,['Date','Phosphate rock', 'DAP', 'TSP', 'Urea ',
                                'Potassium chloride']]   
    
    Metal = CMprice.loc[:,['Date', 'Aluminum', 'Iron ore, cfr spot', 'Copper',
                           'Lead', 'Tin', 'Nickel', 'Zinc',
                           'Steel, cold rolled coilsheet', 
                           'Steel, hot rolled coilsheet',
                           'Steel rebar', 
                           'Steel wire rod']]
    
    Precious_Metal = CMprice.loc[:,['Date','Gold','Platinum','Silver']]
    
    
    numbers = {
        1 : Fossil_fuel  ,
        2 : Beverage     ,
        3 : Oil,
        4 : Grain,
        5 : Food,
        6 : Timber,
        7 : Other_Raw,
        8 : Fertilizer,
        9 : Metal,
        10: Precious_Metal
    }
    return numbers.get(index)



















