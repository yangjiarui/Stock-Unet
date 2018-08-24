# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:16:40 2018

@author: 834235185
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False
np.random.seed(123)

reit=pd.read_excel('E:\\jerry_git\\研究生论文\\projrct\\new_data\\MSCI_REIT_index_normalized_raw_data.xlsx')   #E:\\jerry_git\\研究生论文\\projrct\\new_data\\MSCI_REIT_index.xlsx
reit.index=pd.DatetimeIndex(reit.iloc[:,0])
reit=reit.iloc[:,1:]
df=reit.iloc[:,0]
df=df.sort_index()
df=df.iloc[-50:]
def gen_signal():
    '''在这添加0,1涨跌信号的判断
    当前采用随机给出(0,1)的方式'''
    return np.random.choice([0,1],size=1,p=[0.3,0.7])[0]



buy,sell,relized=[],[],[]
start=df.mean()
r=pd.Series(index=df.index)
for i in range(1,len(df)):
    signal=gen_signal()
    
    if signal == 1:
        buy_price=df.iloc[i]
        if start==0:
            start=buy_price*1.11001102
            buy.append(df.index[i].strftime('%Y-%m-%d'))
    else:
        sell_price=df.iloc[i]
        if start!=0:
            HPR=(sell_price*0.99988-start)/start
            start=0
            sell.append(df.index[i].strftime('%Y-%m-%d'))
            relized.append(HPR)
            r.iloc[i]=HPR

from matplotlib.lines import Line2D
#hh=pd.DataFrame({'buy_date':buy,'relized':relized,'sell_date':sell})
#if hh.iloc[1,1]<hh.iloc[1,0]:
#    hh.iloc[1,:]=hh.iloc[1,:].shift(1)

#hh.iloc[:,1].cumsum().plot(title='algoraism')
plt.plot(relized,label='algoriasm：一次完整交易的持有期收益')
plt.legend(loc=0)
plt.figure()
seged=pd.Series(index=df.index)
for seq,i in enumerate(seged.index):
    i=i.strftime('%Y-%m-%d')
    if i in sell:
        seged.iloc[seq]=10000
        plt.plot([seq,seq],[0,800],color='r')
    elif i in buy:
        seged.iloc[seq]=10000
        plt.plot([seq,seq],[0,1000],color='g')
    else:
        seged.iloc[seq]=None
        plt.plot([seq,seq],[0,500])
plt.figure()
df=(df-df.shift(1))/df.shift(1)
df.cumsum().plot(title='真实日收益率累加')
print('第二幅图中:\n红色代表当天的卖出（统计一次持有期收益）\n绿色代表当天的买入操作\n其余代表“无效的交易信号”')




