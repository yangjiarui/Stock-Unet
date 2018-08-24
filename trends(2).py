#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-7-29下午7:57'
'''

import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans

f = open('googtrends.csv', 'r')
df = pd.read_csv(f)
df.index = pd.date_range(start='2004-01', periods=len(df), freq='M')
df = df.iloc[:, 1:]
full_index = pd.date_range(start='2005-06-17', end='2018-05-16')
new = pd.DataFrame(index=full_index, columns=df.columns)
for i in range(len(new)):
    if new.index[i] in df.index:
        q = df[df.index == new.index[i]]
        new.iloc[i, :] = pd.Series(q.values.reshape(-1), index=q.columns)

new = new.fillna(method='bfill')
new = new[(new.index >= '2005-06-17') & (new.index <= '2018-03-16')]


def work_on_Serise(serise, n_cluster):
    clf = KMeans(n_clusters=n_cluster, n_jobs=6)
    clf.fit(serise.values.reshape(-1, 1))
    c = pd.DataFrame(clf.cluster_centers_).sort_values(0)
    w = c.rolling(2).mean().iloc[1:, 0]
    w = [0] + list(w) + [serise.max()]
    return pd.cut(serise, w, labels=range(n_cluster))


# q=work_on_Serise(serise=new.iloc[:,1],n_cluster=8)


rent = new.iloc[:, 1]
REIT=new.iloc[:,0]
Real_estate=new.iloc[:,2]
stock=new.iloc[:,3]
pmax = int(len(rent) / 600)
qmax = int(len(rent) / 600)
from sklearn.preprocessing import StandardScaler, scale
from statsmodels.tsa.arima_model import ARIMA

#
#
#
# bix = []
# for p in range(pmax + 1):
#     tm = []
#     for q in range(qmax + 1):
#         try:
#             cur=ARIMA(sp500_div, order=(p, 0, q)).fit().bic
#             tm.append(cur)
#             print(cur)
#         except:
#             tm.append(None)
#     bix.append(tm)
# find = pd.DataFrame(bix)
# find.columns.name = 'p'
# find.index.name = 'q'
# q = find.unstack().astype('float32')
# print('q,p: ', q.idxmin())  # (3,2)
import matplotlib.pyplot as plt


def make_predict(data):
    n=data.values.tolist()
    k=0
    while 1:
        qq=ARIMA(n[k:], order=(1, 0, 0)).fit()
        try:
            add=qq.forecast(1)[0]
            if len(add)>=0:
                n.append(add[0])
                print(n[-1])
        except:
            print("error")
        k+=1
        if k==150:
            break
    return work_on_Serise(pd.Series(n),n_cluster=15)

rent,REIT,stock,real_estate=make_predict(rent),make_predict(REIT),make_predict(stock),make_predict(Real_estate)
from sklearn.decomposition import PCA
clf=PCA(n_components=1)
clf.fit_transform(pd.concat([real_estate,REIT],ignore_index=True,axis=1))




# ww=work_on_Serise(pd.Series(n),n_cluster=15)
# ww.hist(bins=40)
# plt.show()
# #print(qq.summary())

