#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-5-10下午2:12'
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
os.chdir('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/factors')

lis=glob('*(*)*')
lis.pop(3)

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

from collections import Counter

def data_inception(Serise,top_largest_count):
    q=pd.Series(Counter(Serise)).sort_values().iloc[-abs(top_largest_count):]
    return len(Serise)-sum(q)




q={}
for i in lis:
    df=pd.read_excel(i)
    df.dropna(axis=0,how='all',inplace=True)
    df.index=pd.DatetimeIndex(df.index.droplevel())
    data=standarized_data(dataframe=df,foward_steps=5)
    q[i]={}
    for seq,j in enumerate(data.columns):
        q[i][j]=data_inception(Serise=data[j],top_largest_count=3)

####find _avaliabe REIT compant shares:
all=pd.DataFrame()
for _ in range(len(q)):
    w=q.popitem()
    df=pd.read_excel(w[0])
    df.dropna(axis=0,how='all',inplace=True)
    df.index=pd.DatetimeIndex(df.index.droplevel())
    data=standarized_data(dataframe=df,foward_steps=3)

    data=data

    nod=pd.Series(w[1])

    for seq,i in enumerate(nod):
        if i >=4097:
            all[nod.index[seq]]=pd.Series(data[nod.index[seq]])

    all.to_csv('my_avaliabe_REIT_company_exchanges_data.csv')