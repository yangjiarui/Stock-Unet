# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 11:28:08 2018

@author: 834235185
"""

import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans


f=open('googtrends.csv','r')
df=pd.read_csv(f)
df.index=pd.date_range(start='2004-01',periods=len(df),freq='M')
df=df.iloc[:,1:]
full_index=pd.date_range(start='2005-06-17',end='2018-05-16')
new=pd.DataFrame(index=full_index,columns=df.columns)
for i in range(len(new)):
    if new.index[i] in df.index:
        q=df[df.index==new.index[i]]
        new.iloc[i,:]=pd.Series(q.values.reshape(-1),index=q.columns)
        
new=new.fillna(method='bfill')
new=new[(new.index>='2005-06-17') & (new.index<='2018-03-16')]

def work_on_Serise(serise,n_cluster):
    clf=KMeans(n_clusters=n_cluster,n_jobs=6)
    clf.fit(serise.values.reshape(-1,1))
    c=pd.DataFrame(clf.cluster_centers_).sort_values(0)
    w=pd.rolling_mean(c,2).iloc[1:,0]
    w=[0]+list(w)+[serise.max()]
    return pd.cut(serise,w,labels=range(n_cluster))

q=work_on_Serise(serise=new.iloc[:,1],n_cluster=8)
for w in range(len(new.columns)):
   q=work_on_Serise(serise=new.iloc[:,1],n_cluster=4)
   q.hist()