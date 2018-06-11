#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-5-4下午7:10'
'''
import pandas as pd
import sys,time
#reload(sys)
#sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import pca
st1=time.clock()
big_cap=pd.read_excel('factors/largest_10_(PB)_REIT_companies.xlsx')
big_cap.dropna(axis=0,how='all',inplace=True)
small_cap=pd.read_excel('factors/relativily_small_PB(REIT)_companies.xlsx')
small_cap.dropna(axis=0,how='all',inplace=True)
small_cap.index=small_cap.index.droplevel()
big_cap.index=big_cap.index.droplevel()
small_cap=small_cap.sort_index(ascending=True)
big_cap=big_cap.sort_index(ascending=True)
small_cap.index=pd.DatetimeIndex(small_cap.index)
big_cap.index=pd.DatetimeIndex(big_cap.index)

big_cap=np.log(big_cap/big_cap.shift(1))
small_cap=np.log(small_cap/small_cap.shift(1))


def normalize_raw_data(df):
    j=[]
    j.append(0)
    for i in df:
        if np.isnan(i) == True:
            #print(len(j))
            j.append(np.mean(j[-5:]))
        else:
            j.append(i)

    j=j[1:]
    #print(np.mean(j))
    return pd.Series(j,index=df.index)


new_small_cap = pd.DataFrame()
for seq, i in enumerate(small_cap.columns):
    new_small_cap[i] = normalize_raw_data(small_cap.iloc[:, seq])
    print(str(i) + 'Done')
print ('@@@@@@'+'add'+'@@@@@@@@@@')
new_big_cap = pd.DataFrame()
for seq, i in enumerate(big_cap.columns):
    new_big_cap[i] = normalize_raw_data(big_cap.iloc[:, seq])
    print(str(i) + 'Done')

def potfileo_reture(data_frame):
    weights = abs(np.random.uniform(0,1,len(data_frame.columns)))
    weights /= np.sum(weights)
    ret=np.sum(weights*data_frame.mean())*252
    std=np.sqrt(np.dot(weights.T,np.dot(data_frame.cov()*252,weights)))
    return ret,std,weights
#equle_weights=np.random.normal(0,1,16)

reit=pd.read_excel('repository/S5REITS INDEX.xlsx').iloc[1:,:2]
reit.index=reit.iloc[:,0]
reit.index=pd.DatetimeIndex(reit.index)
nod=reit.iloc[:,1].astype('float64')
reit=np.log(nod/nod.shift(1))
reit=reit.fillna(reit.mean())
import threading
#weights
from multiprocessing import Process

import multiprocessing
def MainRange(start):
    #global std,ret,weughts
    ret=[]
    std=[]
    weughts=[]
    for i in range(start,start+10000):
        print('loading'+str(i))
        a,b,c=potfileo_reture(new_big_cap)   #change
        ret.append(a)
        std.append(b)
        weughts.append(c)
    return ret,std,weughts

pool=multiprocessing.Pool(processes=20)
#pocess=[]
result=[]
for j in range(0,10000000,10000):
    result.append(pool.apply_async(MainRange,args=(j,)))
pool.close()
pool.join()

sum_ret = []
sum_std = []
sum_weughts = []
for node in result:
    ret, std, weughts=node.get()
    sum_ret=sum_ret+ret
    sum_std=sum_std+std
    sum_weughts=sum_weughts+weughts
plt.figure(figsize=(16,9))
plt.xlabel('std')
plt.ylabel('ret')
plt.scatter(np.array(sum_std), np.array(sum_ret), c=(np.mean(sum_weughts) - np.mean(reit)) / np.array(sum_std),linewidths=0.2, marker='.')
w=pd.DataFrame()
w['ret']=pd.Series(sum_ret)
w['std']=pd.Series(sum_std)

w['weights']=pd.Series(sum_weughts)
w.to_csv('10000000_new_big_PB.csv')   #change
with open('simulation_10000000_new_big_PB_index.txt','w') as f:
    f.write(str(new_small_cap.columns.tolist()))


def start_optimize_Main():
    import scipy.optimize as spo

    def statistics(weights):
        global new_big_cap,new_small_cap,reit
        weights=np.array(weights)
        potfilo_return=np.sum(new_small_cap.mean()*weights)*252
        potfilo_std=np.sqrt(np.dot(weights.T,np.dot(new_small_cap.cov()*252,weights)))
        return np.array([potfileo_reture,potfilo_std,(potfilo_return-reit.mean())/potfilo_std])

    def min_sharpe(weights):
        return -statistics(weights)[2]
    def min_varians(weights):
        return statistics(weights)[1]
    cons=({'type':'eq','fun':lambda x:sum(x)-1})
    bnds=tuple((-1,1) for x in range(len(new_small_cap.columns)))
    opts=spo.minimize(min_sharpe,len(new_small_cap.columns)*[1./len(new_small_cap.columns),],method='SLSQP',bounds=bnds,constraints=cons)

    target_returns = np.linspace(0.0, 0.5, 50)

    target_variance = []

    for tar in target_returns:
        cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0]-tar}, {'type': 'eq', 'fun': lambda x: np.sum(x)-1})

        res = spo.minimize(min_varians,len(new_small_cap.columns)*[1./len(new_small_cap.columns),], method='SLSQP', bounds=bnds, constraints=cons)

        target_variance.append(res['fun'])

    target_variance = np.array(target_variance)
    print(opts)

    #eturn #opts,
    return opts



#q=start_optimize_Main()









print('time used:',time.clock()-st1)
#plt.show()

#t1 = Process(target=MainRange,args=(0,))
#pocess.append(t1)
#t2 = Process(target=MainRange,args=(10000,))
#pocess.append(t2)

# t3 = threading.Thread(target=MainRange,args=(20000,))
# threads.append(t3)

# t4 = threading.Thread(target=MainRange,args=(30000,))
# threads.append(t4)
#
# t5 = threading.Thread(target=MainRange,args=(40000,))
# threads.append(t5)
#
# t6 = threading.Thread(target=MainRange,args=(50000,))
# threads.append(t6)
#
# t7 = threading.Thread(target=MainRange,args=(60000,))
# threads.append(t7)
#
# t8 = threading.Thread(target=MainRange,args=(70000,))
# threads.append(t8)
#
# t9 = threading.Thread(target=MainRange,args=(80000,))
# threads.append(t9)
#
# t10 = threading.Thread(target=MainRange,args=(90000,))
# threads.append(t10)
# t11 = threading.Thread(target=MainRange,args=(100000,))
# threads.append(t11)
# t12 = threading.Thread(target=MainRange,args=(110000,))
# threads.append(t12)
# t13 = threading.Thread(target=MainRange,args=(120000,))
# threads.append(t13)
#st2=time.clock()
#for t in pocess:
    #t.setDaemon(True)
 #   t.start()

#t.join()
#print('time used:',time.clock()-st1)

#
# plt.scatter(np.array(std), np.array(ret), c=(np.mean(ret) - np.mean(reit)) / np.array(std), marker='*')
# plt.show()t

