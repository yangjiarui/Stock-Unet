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
names=os.listdir('factors/avalible_REITs')




macro=pocess(macro)
micro=pocess(micro)
data=pd.concat([macro,micro],join='outer',join_axes=[micro.index],axis=1)
# data.plot(subplots=True,figsize=(18,20))
#plt.show()
click=pd.read_csv('click_data.csv')

#click
click=pocess(click)
risk_free=pd.read_excel('us_MACRO.xls')
risk_free=risk_free.iloc[7:-3,:]
risk_free.index=risk_free.iloc[:,0]
risk_free=risk_free.iloc[:,4].astype('float64')
risk_free.index=list(map(lambda x:x.strftime('%Y-%m-%d'),risk_free.index))
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier
from sklearn.model_selection import cross_val_predict,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def create_nn_model(input_dim, activation,final_activation, layers, loss):
    model = Sequential()
    l_num = 0
    for l in layers[:-1]:
        if l_num == 0:
            model.add(Dense(l, input_dim=input_dim, activation=activation, init='he_normal'))
            model.add(Dropout(0.6))
        else:
            model.add(Dense(l, activation=activation, init='he_normal'))
            model.add(Dropout(0.4))
        l_num = l_num + 1
    model.add(Dense(layers[-1],activation=final_activation))
    # model.add(Dropout(0.4))
    model.compile(optimizer='sgd', loss=loss)
    return model


def make_one_col_label(new):
    new=new.values
    label=[]
    for i in range(len(new)):
        if new[i]>=0:
            label.append(1)
        else:
            label.append(0)
    return pd.Series(label)

def Main(new,pre_mean_step,epoch):
    train_new = new.iloc[:int(0.95 * len(new)), :]
    test_new = new.iloc[int(0.95 * len(new)):, :]
    model=create_nn_model(input_dim=len(train_new.columns),activation='relu',final_activation='sigmoid',loss='binary_crossentropy',layers=[500,1000,860,500,250,80,1])
    print(model.summary())
    label=make_one_col_label(train_new.iloc[:,0])
    model.fit(StandardScaler().fit_transform(train_new.values),label.values,epochs=epoch,batch_size=8,validation_split=0.2)
    pre=model.predict(StandardScaler().fit_transform(test_new.values))
    lis=model.predict(StandardScaler().fit_transform(train_new.values))
    lis=lis.tolist()
    lis=[i[0] for i in lis]
    mean=pd.Series(lis[-pre_mean_step:]).median()
    pre_label=[]
    for predict,ture in zip(pre,make_one_col_label(test_new.iloc[:,0])):
        if predict[0]>=mean and ture==1:
            pre_label.append(predict)
        elif predict[0]<mean and ture==0:
            pre_label.append(predict)
        lis.append(predict[0])
        mean=(pd.Series(lis[-pre_mean_step:]).median())
    plt.subplots(2)
    plt.subplot(211)
    plt.plot(pre_label)
    plt.subplot(212)
    plt.plot(test_new.iloc[:,0].values)
    plt.show()
    accuracy=float(len(pre_label)/len(make_one_col_label(test_new.iloc[:,0])))
    print('%s 对应的 总准确率为：%.4f'%(train_new.iloc[:,0].name,accuracy))
    return accuracy,model


stat={}
for i in names[:]:
    time.sleep(5)
    f=open('factors/avalible_REITs/'+i,'r')
    cur=pd.read_csv(f,header=None)

    print(i)
    cur.columns=['time',i.split('.')[0]]
    cur=pocess(cur)
    cur = cur.sort_index(ascending=True)
    cur=np.log(cur/cur.shift(1))
    plt.figure()
    cur=pd.concat([cur,data,click,risk_free],join='outer',join_axes=[data.index],axis=1)
    #cur=pd.concat([cur,w],join='outer',join_axes=[cur.index],axis=1)
    new=pd.DataFrame()
    try:
        new['%s'%i.split('.')[0]]=cur.iloc[:,0]
        new['%s下市场因子对应的序列'%i.split('.')[0]]=cur['MSCI_REIT']-cur.iloc[:,-1].pct_change()
        new['%s下规模因子对应的序列'%i.split('.')[0]]=cur['small_cap_simu']-cur['big_cap_simu']
        new['%s下价值因子对应的序列'%i.split('.')[0]]=cur['big_PB_simu']-cur['small_PB_simu']
        new['劳动因素'] =cur['labor']
        new['利率类']=cur['LIBOR_interst']
        new['收益率类']=cur['bond_yd_long_loan']
        new['通胀指数']=cur['CPI']/cur['CPI'].shift(1)
        new['标普500对数收益率']=cur['sp500']
        new['债券指数对数收益率']=cur['bond']
        new['处理后REIT股利']=cur['reit_div']
        new['处理后标普500股利']=cur['sp500_div']
        new['处理后REIT类点击量']=cur['REIT_reit_estate_click']
        new['处理后股票类点击量']=cur['rent_stock_click']
        new = new.dropna(axis=0, how='any')
        acc,model=Main(new,pre_mean_step=150,epoch=25)
        # model.save('/media/jerry/test/git_version_control/Stock-Unet/model/%s_acc_%s.h5'%(i.split('.')[0],acc))
        stat[i.split('.')[0]]=acc
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        new.plot(subplots=True, figsize=(18, 25))
        plt.legend(loc=0)
        plt.grid(True)
        sns.pairplot(new[20:],kind='reg')
        plt.show()
    except:
        print('error')

##### netual network


    # model=Sequential()
    # model.add(Dense(input_dim=len(new.columns),activation='relu'))
