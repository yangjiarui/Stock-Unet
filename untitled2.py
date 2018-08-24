# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:24:20 2018

@author: 834235185
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
#df=pd.read_csv('sp500.csv',index_col='03/21/2018').iloc[:-1,0:3]
#
#df.index=pd.DatetimeIndex(df.index)
#df=df.fillna(method='ffill')
#df=df.fillna(method='bfill')
#q=df.resample('m',how='mean')
#q=q.iloc[:,0]
#plt.figure(figsize=(10,6))
#plt.title('sp500 index')
#plt.plot(q.index,q.values)


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


#ds=standarized_data(pd.read_csv('sp500.csv',index_col='03/21/2018').sort_index(),foward_steps=5)


        
from collections import Counter

        

reit=pd.read_excel('MSCI_REIT_index_normalized_raw_data.xlsx')   #E:\\jerry_git\\研究生论文\\projrct\\new_data\\MSCI_REIT_index.xlsx
reit.index=pd.DatetimeIndex(reit.iloc[:,0])
reit=reit.iloc[:,1:]
        
reit=standarized_data(reit,foward_steps=3)
reit=reit[reit.index<='2018-03-16']
reit_div=reit.iloc[:,2]   ############
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
        



sp=pd.read_excel('sp500_normalized_raw_data.xlsx')
sp.index=pd.DatetimeIndex(sp.iloc[:,0])

sp=sp.iloc[:,1:]
cur=sp[sp.index>='2005-06-17']
sp500=cur[cur.index<='2018-03-16']
q=sp500.iloc[:,2]
ds=sp500.iloc[:,0]
a=[]
for i in range(len(ds)-1):
    if ds[i+1]<ds[i]:
        a.append('lowwer')
    elif ds[i+1]==ds[i]:
        a.append('same')
    else:
        a.append('bigger')
print('sp500:',Counter(pd.Series(a)))


trend=pd.read_csv(open('googtrends.csv','r'))
trend.index=pd.DatetimeIndex(trend.iloc[:,0])
trend=trend.iloc[:,1:]
        
bond=pd.read_excel('/home/jerry/inteligient/MASTER_project/sp500_Bond_INdex.xlsx')
bond.index=bond.iloc[:,0]
bond=bond.iloc[:,1:]
bond=bond[(bond.index>='2005-06-17') & (bond.index<='2018-03-16')]
        
#df=pd.read_csv('I:\MASTER_project\\biggest(market_cap)REIT_simu.csv')
#df.index=pd.DatetimeIndex(df.iloc[:,0])
#q=df.resample('8D',how='mean')
#q.plot(figsize=(16,9),title='biggest(market_cap)REIT_simu')
#plt.show()

f=open('/home/jerry/inteligient/MASTER_project/big_cap_simu.csv','r')
big_cap=pd.read_csv(f)
big_cap.index=pd.DatetimeIndex(big_cap.iloc[:,0])
big_cap=big_cap.iloc[:,1]
big_cap=big_cap[(big_cap.index>='2005-06-17') & (big_cap.index<='2018-03-16')]


reit=reit.sort_index()
reit=np.log(reit.iloc[:,0]/reit.iloc[:,0].shift(1))
sp500=np.log(sp500.iloc[:,0]/sp500.iloc[:,0].shift(1))




f=open('small_cap_simu.csv','r')
small_cap=pd.read_csv(f)
small_cap.index=pd.DatetimeIndex(small_cap.iloc[:,0])
small_cap=small_cap.iloc[:,1:]
small_cap=small_cap[(small_cap.index>='2005-06-17') & (small_cap.index<='2018-03-16')]


f=open('/home/jerry/inteligient/MASTER_project/big_PB_simu.csv','r')
big_PB=pd.read_csv(f)
big_PB.index=pd.DatetimeIndex(big_PB.iloc[:,0])
big_PB=big_PB.iloc[:,1:]
big_PB=big_PB[(big_PB.index>='2005-06-17') & (big_PB.index<='2018-03-16')]


f=open('/home/jerry/inteligient/MASTER_project/small_PB_simu.csv','r')
small_PB=pd.read_csv(f)
small_PB.index=pd.DatetimeIndex(small_PB.iloc[:,0])
small_PB=small_PB.iloc[:,1:]
small_PB=small_PB[(small_PB.index>='2005-06-17') & (small_PB.index<='2018-03-16')]




# common_index=big_cap.index
# step=20
# new=pd.DataFrame()
# new['MSCI_REIT']=pd.Series(reit,index=common_index)#.values[-step:]
# new['sp500']=pd.Series(sp500,index=common_index)#.values[-step:]
# new['big_cap_simu']=big_cap
# new['small_cap_simu']=small_cap.iloc[:,1]
# new['big_PB_simu']=big_PB.iloc[:,1]
# new['small_PB_simu']=small_PB.iloc[:,1]


from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA

#
# reit=pd.read_excel('/home/jerry/inteligient/MASTER_project/MSCI_REIT_index_normalized_raw_data.xlsx')   #E:\\jerry_git\\研究生论文\\projrct\\new_data\\MSCI_REIT_index.xlsx
# reit.index=pd.DatetimeIndex(reit.iloc[:,0])
# reit=reit.iloc[:,1:]
#
# reit=standarized_data(reit,foward_steps=3)
# reit=reit[reit.index<='2018-03-16']
# ds=reit.sort_index()
# reit_div=ds.iloc[:,2]
reit_div=reit_div.resample('1M','mean')


sp500_div=q.resample('30D','mean').diff(1)[1:]
print(ADF(sp500_div))




def div_predict(sp500_div):
    pmax=int(len(sp500_div)/100)
    qmax=int(len(sp500_div)/100)
    from sklearn.preprocessing import StandardScaler,scale

    #
    ssssss=StandardScaler()

    bix=[]
    for p in range(pmax+1):
        tm=[]
        for q in range(qmax+1):
            try:
                tm.append(ARIMA(sp500_div,order=(p,1,q)).fit().bic)
            except:
                tm.append(None)
        bix.append(tm)

    import matplotlib.pyplot as plt
    find=pd.DataFrame(bix)
    find.columns.name='p'
    find.index.name='q'
    q=find.unstack().astype('float32')
    print('q,p: ',q.idxmin())
    clf=ARIMA(sp500_div,order=(3,1,2)).fit()  #(3,2)
    print(clf.summary())
    plt.figure()
    clf.plot_predict()
    plt.show()
    return clf.forecast(1)[0]

reit_div_pre=div_predict(reit_div)
sp500_div_pre=div_predict(sp500_div)

common_index=big_cap.index
step=20
new=pd.DataFrame()
new['MSCI_REIT']=pd.Series(reit,index=common_index)#.values[-step:]
new['sp500']=pd.Series(sp500,index=common_index)#.values[-step:]
new['big_cap_simu']=big_cap
new['small_cap_simu']=small_cap.iloc[:,1]
new['big_PB_simu']=big_PB.iloc[:,1]
new['small_PB_simu']=small_PB.iloc[:,1]
new['reit_div']=reit_div
new['sp500_div']=sp500_div
new['bond']=bond.pct_change()
new=new.fillna(method='ffill')
new=new.fillna(new.mean())
print(new.corr())
#bix=[]
#
#q=ARIMA(sp500_div,order=(11,0,2)).fit()
#print(q.summary())
