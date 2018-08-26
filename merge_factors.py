# -*- coding: utf-8 -*-
#作者: handsome_jerry
#时间:18-8-25上午9:40
#文件:merge_factors.py
import pandas as pd
import numpy as np
import os,time
import matplotlib.pyplot as plt
macro,micro=pd.read_csv('final_macro_factors.csv'),pd.read_csv('final_micro_factors.csv')
def pocess(df):
    df.index=df.iloc[:,0]
    return df.iloc[:,1:]
names=os.listdir('/home/jerry/inteligient/git_version/Stock-Unet/factors/avalible_REITs')

macro=pocess(macro)
micro=pocess(micro)
data=pd.concat([macro,micro],join='outer',join_axes=[micro.index],axis=1)
# data.plot(subplots=True,figsize=(18,20))
#plt.show()
click=pd.read_csv('click_data.csv')
#click
click=pocess(click)
for i in names:
    time.sleep(3)
    cur=pd.read_csv('/home/jerry/inteligient/git_version/Stock-Unet/factors/avalible_REITs/'+i,header=None)
    print(i)
    cur.columns=['time',i.split('.')[0]]
    cur=pocess(cur)
    cur = cur.sort_index(ascending=True)
    cur=np.log(cur/cur.shift(1))
    plt.figure()
    cur=pd.concat([cur,data,click],join='outer',join_axes=[data.index],axis=1)
    new=pd.DataFrame()
    try:
        new['市场因子']=cur.iloc[:,0]-cur['MSCI_REIT']
        new['规模因子']=cur['small_cap_simu']-cur['big_cap_simu']
        new['价值因子']=cur['big_PB_simu']-cur['small_PB_simu']
        new['劳动因素'] =cur['labor']
        new['利率类']=cur['LIBOR_interst']
        new['收益率类']=cur['bond_yd_long_loan']
        new['通胀指数']=cur['CPI']/cur['CPI']
        new['标普500对数收益率']=cur['sp500']
        new['债券指数对数收益率']=cur['bond']
        new['处理后REIT股利']=cur['reit_div']
        new['处理后标普500股利']=cur['sp500_div']
        new['处理后REIT类点击量']=cur['REIT_reit_estate_click']
        new['处理后股票类点击量']=cur['rent_stock_click']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        new.plot(subplots=True, figsize=(18, 20))
        plt.show()
    except:
        print('error')


