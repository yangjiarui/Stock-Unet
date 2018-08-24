# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:55:01 2018

@author: 834235185
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

df=pd.read_excel('us_MACRO.xls').iloc[6:-2,:]
index=df.iloc[:,0]
df.index=pd.DatetimeIndex(index)
a=df['LIBOR:美元:隔夜'][:]
b=df['美国:个人消费支出物价指数:季调'][:]
c=df['美国:联邦基金利率(日)'][:]
d=df['美国:国债长期平均实际利率'][:]
e=df['美国:30年期抵押贷款固定利率'][:]
f=df['美国:劳动力参与率:季调']*df['美国:就业率:季调']
#f=f.fillna(method='ffill')
#f=f.fillna(method='bfill')
q=f[f.notnull()].astype('float32')
f=q/q.shift(1)
#f=f[f.notnull()].astype('float32')

#a=a.fillna(method='ffill')
#b=b.fillna(method='ffill')
#c=c.fillna(method='ffill')
#d=d.fillna(method='ffill')
#a=a.fillna(method='bfill')
#b=b.fillna(method='bfill')
#c=c.fillna(method='bfill')
#d=d.fillna(method='bfill')
ds=pd.DataFrame()
ds['LIBOR'],ds['interst'],ds['CPI'],ds['bond_yeld'],ds['long_term_loan_ rate'],ds['labor']=a,c,b,d,e,f
ds=ds[(ds.index>='2005-06-17') & (ds.index<='2018-03-16')]
#del ds['CPI']

sp=pd.read_excel('sp500_normalized_raw_data.xlsx')
sp.index=pd.DatetimeIndex(sp.iloc[:,0])
sp=sp.iloc[:,1:]
#cur=sp[sp.index>='2005-06-17']
#sp500=cur[cur.index<='2018-03-16']
sp500=sp.iloc[:,0]

reit=pd.read_excel('MSCI_REIT_index_normalized_raw_data.xlsx')
reit.index=pd.DatetimeIndex(reit.iloc[:,0])
reit=reit.sort_index()
reit=np.log(reit.iloc[:,1]/reit.iloc[:,1].shift(1))
sp500=np.log(sp500/sp500.shift(1))
reit=reit[(reit.index>='2005-06-17') & (reit.index<='2018-03-16')]
sp500=sp500[(sp500.index>='2005-06-17') & (sp500.index<='2018-03-16')]
full=pd.concat([sp500,reit,ds],axis=1)
full=full.fillna(method='bfill')
full=full.fillna(method='ffill')

import statsmodels.api as sm

others=[j for j in range(len(full.columns)) if j!= 1 and j!=0 and j!=7]  
c=pd.DataFrame(columns=['F_pvalue','adj_R**2','total_mse','fitted_columns'])
#for x in range(80):
#    cur=np.random.choice(a=others,size=2)
#    clf=sm.OLS(full.iloc[:,1],sm.add_constant(full.iloc[:,min(cur):max(cur)])).fit()
#   # a['REIT~%s to %s '%(min(cur),max(cur))]=
#    
#    c=c.append(pd.Series({'F_pvalue':clf.f_pvalue,'adj_R**2':clf.rsquared_adj,'total_mse':clf.mse_total,'fitted_columns':'REIT~ %s to %s '%(min(cur),max(cur))}),ignore_index=True)
#    
        
for s in range(len(others)):
    for i in range(1,len(others)-s):
        print(others[s:s+i])
        clf=sm.OLS(full.iloc[:,1],sm.add_constant(full.iloc[:,others[s:s+i]])).fit()
        c=c.append(pd.Series({'F_pvalue':clf.f_pvalue,'adj_R**2':clf.rsquared_adj,'total_mse':clf.mse_total,'fitted_columns':'REIT~ %s to %s '%(others[s],others[s+i])}),ignore_index=True)
        
c.index=c.iloc[:,-1]
c=c.iloc[:,:-1]
minn=100

import scipy.stats as scs
z=scs.normaltest(full) 
d=[]
from sklearn.decomposition import PCA
for i in range(1,len(full.iloc[1,2:])):
    clf=PCA(n_components=i)
    new=clf.fit_transform(full.iloc[:,2:])
    for j in range(len(new[1,:])):
        aa=sm.OLS(full.iloc[:,1],sm.add_constant(new[:,:j])).fit()
        if aa.f_pvalue<minn:
            minn=aa.f_pvalue
            print('found it',new[:,:j])
        else:
            print('not found')
            minn=minn
        d.append(aa.f_pvalue)
            
            
fin_m=new[:,:j]






import matplotlib.pyplot as plt





I=40000
M=50
theta=full.iloc[-30:,2].mean()
sigma=full.iloc[-180:,2].std()
T=10000
kappa=np.random.randn(1)[0]
dt=T/M
def srd_exact(s0,theta,sigma,kappa,I=I):
    x2=np.zeros((M+1,I))
    x2[0]=s0
    for t in range(1,1+M):
        df=4*theta*kappa/sigma**2
        c=(sigma**2*(1-np.exp(-kappa*dt)))/(4*kappa)
        nc=np.exp(-kappa*dt)/c*x2[t-1]
        x2[t]=c*np.random.noncentral_chisquare(df,nc,size=I)
    return x2
#q=srd_exact(s0=full.iloc[-1,2],kappa=kappa,sigma=sigma,theta=theta)
#print(q)

def hehe(x):
    a=[]
    for i in x:
        if i>=0:
            a.append(i)
        else:
            a.append(0)
    return a
def Main(num,I):
    new=full.iloc[:,num].values.tolist()
    k=0
    start=10
    #np.random.seed(13)
    sigma=full.iloc[-180:,num].std()
    kappa=abs(np.random.uniform(0,1,1))[0]
    theta=full.iloc[-start:,num].mean()
    while 1:
        q=srd_exact(s0=new[-1],kappa=kappa,sigma=sigma,theta=theta,I=I)
        m=[]
        for i in range(q.shape[0]-1,45,-1):
            cur=q[i,:]
            cur.sort()
            m.append(np.mean(cur[int(len(cur)*0.15):int(len(cur)*0.85)]))
        new.append(np.mean(m))
        M=50
        theta=np.mean(new[-start:])
        start+=1
        sigma=np.std(new[-180:])
        
        if theta<0:
            c=[]
            for j in range(len(new)-1,int(0.85*len(new)),-1):
                if new[j] >= 0:
                    c.append(new[j])
            theta=np.mean(c)
        T=1.0
        kappa=abs(np.random.uniform(0,1,1))[0]
        dt=T/M
        print('added:',np.mean(m))
        k+=1
        if k==50:
            break
    plt.figure()
    plt.plot(new[-100:])
    plt.annotate('begin_to_predict', xy=(49, new[-50]), xytext=(45, pd.Series(new).describe()['75%']),
                arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.scatter(49,new[-51],marker='*',c='r')
    plt.grid(True)
    plt.legend(loc=0)
    # plt.figure()
    # plt.plot(q[:,:8])
    plt.show()
    return new
def sigmoid(x):
    return 1/1-np.exp(-x)

import multiprocessing.pool as pool
pool=pool.Pool(4)
res=[]
commo_index=full.iloc[:,2].index
for i in [2,3,6,5]:
   res.append(pool.apply_async(Main,args=(i,10000)))
pool.close()
pool.join()


final_res=[]
for d in res:
    cur=d.get()
    final_res.append(cur)




############################################
#
# LIBOR=Main(num=2,I=100000)
# interest=Main(num=3,I=100000)
# long_loan=Main(num=6,I=100000)
# bond_yield=Main(num=5,I=100000)
#
#
LIBOR_interst=PCA(n_components=1).fit_transform(np.array([final_res[0],final_res[1]]).T)
bond_yd_long_loan=PCA(n_components=1).fit_transform(np.array([final_res[2],final_res[3]]).T)
#
#
#plt.figure()
#plt.plot(LIBOR_interst[-100:])
#plt.figure()
#plt.plot(bond_yd_long_loan[-100:])
#plt.show()
from sklearn.cluster import KMeans


def work_on_Serise(serise, n_cluster):
    serise=serise.astype('float64')
    clf = KMeans(n_clusters=n_cluster, n_jobs=6)
    clf.fit(serise.values.reshape(-1, 1))
    c = pd.DataFrame(clf.cluster_centers_).sort_values(0)
    w = c.rolling(2).mean()
    w=w[1:].values.tolist()
    w = [0] + list(map(lambda x:x[0],w)) + [serise.max()]
    return pd.cut(serise, w, labels=range(n_cluster))

q=ds['labor']
q=q[q.notnull()]
q=work_on_Serise(q,int(len(q)/10))
cat=pd.Series(q,index=ds.index)
cat=cat.fillna(method='ffill')
pd.Series(cat.values.astype('float32')).plot()
plt.show()

df['labor']=q

ds['LIBOR_interst']=pd.Series([i[0] for i in LIBOR_interst[:len(commo_index)]],index=commo_index)
ds['bond_yd_long_loan']=pd.Series([i[0] for i in bond_yd_long_loan[:len(commo_index)]],index=commo_index)
ds=ds.fillna(method='ffill')
ds.plot(subplots=True,figsize=(16,14))
plt.show()



