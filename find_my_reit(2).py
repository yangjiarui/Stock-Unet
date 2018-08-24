#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-7-3下午8:32'
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import Newton_interpot as newtow
os.chdir('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/factors')

lis=glob('*(*)*')[2:]
# lis.pop(3)

def standarized_data(dataframe):
    for seq,i in enumerate(dataframe.columns):
        nul=[]
        havedone=0
        # if newtow.isNum2(dataframe.iloc[0,seq])==False:
        #     dataframe.iloc[0,seq]=np.mean(dataframe.iloc[:,seq])
        for r in range(len(dataframe.iloc[:,seq])):
            if newtow.isNum2(dataframe.iloc[r,seq]) == False:
                nul.append(r)
        for j in range(len(dataframe.iloc[1:,seq])):
            if newtow.isNum2(dataframe.iloc[j,seq]) == False:
                try:
                    cur=newtow.find_y_accroding_to_x(Num=j,y_data=dataframe.iloc[:,seq].values)
                    if str(cur[0])=='nan':
                        cur[0]=0
                    else:
                        havedone+=1
                        print('%s:%d fillna by newtown: %f'%(i,j,cur[0]))
                    lis=[dataframe.iloc[j-1,seq],dataframe.iloc[j+1,seq],cur[0]]
                    dataframe.iloc[j,seq]=np.mean([q for q in lis if str(q)!='nan'])
                except (IndexError,ZeroDivisionError) as e:
                    print('there is worng in %s:%s worngtype: %s '%(i,j,e))
                    lis = [dataframe.iloc[j - 1, seq], dataframe.iloc[j + 1, seq]]
                    dataframe.iloc[j, seq] = np.mean([q for q in lis if str(q) != 'nan'])
        try:
            print('%s has realized %.4f pure newton method interplotation and is merged by np.mean(df[x-1],df[x+1])' % (i, float(havedone / len(nul))))
        except ZeroDivisionError as e:
            print('%s is good raw data and no need to change'%i)
        print('%s isnull: %s'%(i,dataframe.iloc[:,seq].isnull().any()))
    return dataframe

from collections import Counter

def data_inception(Serise,top_largest_count):
    q=pd.Series(Counter(Serise)).sort_values().iloc[-abs(top_largest_count):]
    return len(Serise)-sum(q)
q={}
def fillna_simple_data(dataframe,foward_steps):
    all=pd.DataFrame()
    for seq,i in enumerate(dataframe.columns):
        cur=[]
        if np.isnan(dataframe.iloc[0,seq])==True or dataframe.iloc[0,seq]==0:
            dataframe.iloc[0,seq]=dataframe.iloc[:,seq].mean()
        cur.append(dataframe.iloc[0,seq])
        for j in dataframe.iloc[1:,seq]:
            if np.isnan(j)==True or j==0:
                cur.append(np.mean(cur[-abs(foward_steps):]))
            else:
                cur.append(j)
        all[i]=pd.Series(cur,index=dataframe.index)
    return all

for i in lis:
    df=pd.read_excel(i)
    df.dropna(axis=0,how='all',inplace=True)
    df.index=pd.DatetimeIndex(df.index.droplevel())
    print(10*'*'+'working on: '+str(i)+10*'%')
    data=standarized_data(dataframe=df)
    data=fillna_simple_data(dataframe=data,foward_steps=3)
    q[i]={}
    for seq,j in enumerate(data.columns):
        q[i][j]=data_inception(Serise=data[j],top_largest_count=3)

####find _avaliabe REIT compant shares:
# all=pd.DataFrame()
# for _ in range(len(q)):
#     w=q.popitem()
#     df=pd.read_excel(w[0])
#     df.dropna(axis=0,how='all',inplace=True)
#     df.index=pd.DatetimeIndex(df.index.droplevel())
#     data=standarized_data(dataframe=df)
#
#     data=data
#
#     nod=pd.Series(w[1])
#
#     for seq,i in enumerate(nod):
#         if i >=4097:
#             all[nod.index[seq]]=pd.Series(data[nod.index[seq]])
#
#     all.to_csv('my_avaliabe_REIT_company_exchanges_data(1).csv')