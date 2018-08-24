# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 03:27:58 2018

@author: 834235185
"""

import pandas as pd
import numpy as np
df=pd.read_excel('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/factors/biggest(market_cap)REIT_companies.xlsx')#0.25:0.75
#df=pd.read_excel('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/factors/relativily_small(market_cap_REIT)_companies.xlsx') #0.35:0.85

df=df.dropna(how='all',axis=0)

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
import torch as t

df.index=df.index.droplevel()
q=standarized_data(dataframe=df.sort_index(),foward_steps=3)
log_r=np.log(q/q.shift(1))
#log_r.plot(subplots=True,figsize=(16,29))
log_r=log_r.iloc[:-1,:]

def get_potfilio_describe(one_row_in_data_frame,NUM_weights):
    c_r,c_std=[],[]
    for i in range(NUM_weights):
        weights= abs(np.random.uniform(0,1,len(one_row_in_data_frame.index)))
       # weights=t.rand(1,16)
        #weights=weights.cuda()
        weights /= weights.sum()
        #one_row_in_data_frame=t.from_numpy(one_row_in_data_frame.values)
        #one_row_in_data_frame=t.FloatTensor(one_row_in_data_frame.tolist()).cuda()
        #cur=one_row_in_data_frame*weights
        cur=one_row_in_data_frame*weights
        c_r.append(cur.sum())
        #c_r.add(cur.sum())
       # c_std.append(cur.std())
        # print('%s Done ' %one_row_in_data_frame.index)
    ccc=set(c_r)  #unique values
    ccc=list(ccc)
    #
    ccc.sort()
    ccc=ccc[int(0.35*len(ccc)):int(0.85*len(ccc))]   #-min and max values exclude outliners

#    del c[c.index(max(c))]
    return np.mean(ccc),np.std(ccc)
    
new=pd.DataFrame(index=log_r.index,columns=['simu_logreturn','simu_std'])
import time
interval=30
def Main(start):
    global interval
    cur_df=pd.DataFrame(index=log_r.index,columns=['simu_logreturn','simu_std'])
    for i in range(start,start+interval):
        cur_time=time.time()
        print('working on %s' %cur_df.index[i])
        c_log_r,c_std=get_potfilio_describe(one_row_in_data_frame=log_r.iloc[i,:],NUM_weights=80000)
        cur_df.iloc[i,0]=c_log_r
        cur_df.iloc[i,1]=c_std
        print ('%s has simuDone and time used is %s'%(cur_df.index[i],time.time()-cur_time))
    print('%d to %d has Done'%(start,start+interval))
    return cur_df.dropna(how='all',axis=0)
        
from multiprocessing import Process
import multiprocessing
pool=multiprocessing.Pool(processes=interval)
#pocess=[]
result=[]
for j in range(0,len(log_r),interval):
    result.append(pool.apply_async(Main,args=(j,)))
pool.close()
pool.join()


full=pd.DataFrame(columns=new.columns)
for j in result:
    cur=j.get()
    full=full.append(cur)

full=full.astype('float32')
full.index=pd.DatetimeIndex(full.index)
full=full.sort_index(ascending=True)
full.to_csv('biggest(market_cap)REIT_simu.csv')

