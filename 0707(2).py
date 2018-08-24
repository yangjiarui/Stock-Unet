# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:54:15 2018

@author: 834235185
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('sp500.csv',index_col='03/21/2018').iloc[:-1,0:3]

df.index=pd.DatetimeIndex(df.index)
df=df.fillna(method='ffill')
df=df.fillna(method='bfill')
q=df.resample('m',how='mean')
q=q.iloc[:,0]
plt.figure(figsize=(10,6))
plt.title('sp500 index')
plt.plot(q.index,q.values)


def standarized_data(dataframe,foward_steps):
    all=pd.DataFrame()
    for seq,i in enumerate(dataframe.columns):
        cur=[]
        if np.isnan(dataframe.iloc[0,seq]):
            dataframe.iloc[0,seq]=dataframe.iloc[:,seq].mean()
        cur.append(dataframe.iloc[0,seq])
        for j in dataframe.iloc[1:,seq]:
            if np.isnan(j):
                cur.append(np.mean(cur[-abs(foward_steps):]))
            else:
                cur.append(j)
        all[i]=pd.Series(cur,index=dataframe.index)
    return all


ds=standarized_data(pd.read_csv('sp500.csv',index_col='03/21/2018').sort_index(),foward_steps=5)

ds=ds.iloc[:,0]
a=[]
for i in range(len(ds)-1):
    if ds[i+1]<ds[i]:
        a.append('lowwer')
    elif ds[i+1]==ds[i]:
        a.append('same')
    else:
        a.append('bigger')
        
from collections import Counter
print('sp500:',Counter(pd.Series(a)))
        

reit=pd.read_excel('E:\\jerry_git\\研究生论文\\projrct\\new_data\\MSCI_REIT_index.xlsx')
reit.index=pd.DatetimeIndex(reit.iloc[:,0])
reit=reit.iloc[:,1:]
        
reit=standarized_data(reit,foward_steps=3)
reit=reit[reit.index<='2018-03-16']
ds=reit.sort_index()
ds=ds.iloc[:,0]
a=[]
for i in range(len(ds)-1):
    if ds[i+1]<ds[i]:
        a.append('lowwer')
    elif ds[i+1]==ds[i]:
        a.append('same')
    else:
        a.append('bigger')
        
from collections import Counter
print('REIT:',Counter(pd.Series(a)))
        



sp=pd.read_excel('E:\\jerry_git\\研究生论文\\projrct\\new_data\\sp500.xlsx').iloc[:-1,:]
sp.index=pd.DatetimeIndex(sp.iloc[:,0])

sp=sp.iloc[:,1:]
cur=sp[sp.index>='1995-01-03']
q=cur[cur.index<='2018-03-16']
        
        
        
        
        
        
        