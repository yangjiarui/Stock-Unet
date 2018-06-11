#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-5-12下午2:46'
'''

import numpy,sys
reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
from pandas import read_csv
import math,os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from glob import glob
import numpy as np
os.chdir('/media/jerry/JERRY/MASTER_project(0)/factors')
# load the dataset
import pandas as pd
####train avaliable REIT-stocks
li=glob('*exchanges*')
dataframe = read_csv(li[0],index_col='Unnamed: 0')






####train_all_US_commom_stocks
dataframe = pd.read_excel('/media/jerry/JERRY/MASTER_project(0)/factors/all.xlsx')
dataframe=dataframe.dropna(how='all',axis=0)
dataframe.index=pd.DatetimeIndex(dataframe.index.droplevel())
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
dataframe=standarized_data(dataframe=dataframe,foward_steps=4)
dataframe=dataframe.sort_index()






#train china 300 stocks
list_names=os.listdir('/media/jerry/JERRY/MASTER_project(0)/a_all_data')
dataframe=pd.DataFrame()
def standarized_data_Serires(Serise,foward_steps):
    all=pd.Series()
    cur=[]
    if np.isnan(Serise.iloc[0]):
        Serise.iloc[0]=Serise.mean()
    cur.append(Serise.iloc[0])
    for j in Serise[1:]:
        if np.isnan(j):
            cur.append(np.mean(cur[-abs(foward_steps):]))
        else:
            cur.append(j)
    return pd.Series(cur,index=Serise.index)


def create_dataset(dataset, look_back=1,nub_of_the_col_index=0):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), nub_of_the_col_index]
        dataX.append(a)
        dataY.append(dataset[i + look_back,nub_of_the_col_index ])
    return numpy.array(dataX), numpy.array(dataY).reshape(-1,1)
import time

def get_LSTM_model():
    model = Sequential()
    model.add(LSTM(look_back + 1, input_shape=(1, look_back), return_sequences=True))
    # model.add(Dense(200))
    # model.add(Dropout(0.2))
    # model.add(LSTM(
    #     50,
    #     return_sequences=True))
    # #
    # # model.add(Dense(200))
    # # model.add(Dropout(0.2))
    # # model.add(LSTM(100,return_sequences=True))
    # # model.add(Dense(2000))
    # # model.add(Dropout(0.4))
    # #
    # model.add(LSTM(
    #     200,
    #     return_sequences=True))
    # model.add(LSTM(
    #     100,
    #     return_sequences=True))
    # model.add(LSTM(
    #     20,
    #     return_sequences=True))
    model.add(LSTM(
        50,
        return_sequences=False))
    # model.add(Dense(4000))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return  model
#def my_test_miti_GPU_train(start):
switch=1
indexxx=0
#while switch!=4:
all_train=[]
df=pd.read_csv('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/a_all_data/300017.SZ.CSV')
df.index=pd.DatetimeIndex(df.iloc[:,2])
#df=df[df.index>'2017-5'].iloc[:,3]

df=standarized_data_Serires(Serise=df.iloc[:,3],foward_steps=3)
dff=pd.DataFrame()
#dff[name.split('.')[0]] = df
dff['300017']=df
dff['ADD_ed']=pd.Series(np.nan,index=df.index)

dataset = dff.values
dataset = dataset.astype('float32')
numpy.random.seed(int(time.clock()))

dataset = dataset
#train_size = int(len(dataset) * 0.856)
#test_size = len(dataset) - train_size
train = dataset




#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# model.add(LSTM(
#     input_shape=(periods-1,1),
#     output_dim=1,
#     return_sequences=True))
# create and fit the LSTM network
switch=1
all_train_data=[]
all_train_data.append(train.tolist())
name='300017'
new_stock_plot=[]
for _ in range(len(dataset)):
    new_stock_plot.append(np.nan)
while switch:
    #globals(dataset)
    #switch=0
    look_back = 4
    trainX, trainY = create_dataset(np.array(all_train_data[-1]), look_back, nub_of_the_col_index=0)
    # testX, testY = create_dataset(test, look_back, nub_of_the_col_index=0)

    trainX_scalar, trainY_scalar = MinMaxScaler(), MinMaxScaler()
    # testX_scalar, testY_scalar = MinMaxScaler(), MinMaxScaler()
    trainX, trainY = trainX_scalar.fit_transform(trainX), trainY_scalar.fit_transform(trainY)
    # testX, testY = testX_scalar.fit_transform(testX), testY_scalar.fit_transform(testY)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #new_stock_plot=[]
    model=get_LSTM_model()
    print (model.summary())
    print trainX.shape,trainY.shape
    model.fit(trainX, trainY, epochs=15, batch_size=8, verbose=2, validation_split=0.2)

    trainPredict = model.predict(trainX)

    trainPredict = trainY_scalar.inverse_transform(trainPredict)
    trainY = trainY_scalar.inverse_transform(trainY)
    nod_addedd = trainPredict[-1]

    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    #testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
    #print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    train_Y, trainPredictPlot = numpy.zeros((np.array(all_train_data[-1]).shape[0])), numpy.zeros((np.array(all_train_data[-1]).shape[0]))
    trainPredictPlot[:] = numpy.nan
    #train_Y=np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict.reshape(-1)
    train_Y[look_back:len(trainPredict) + look_back] = trainY.reshape(-1)
    # shift test predictions for plotting
    #test_Y, testPredictPlot = numpy.zeros((dataset.shape[0], 1)), numpy.zeros((dataset.shape[0], 1))
    # testPredictPlot[:,:] = numpy.nan
    #testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1] = testPredict
    #test_Y[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1] = testY
    # plot baseline and predictions


    #indexxx=1
    new_stock_plot.append(nod_addedd.tolist()[0])
    q=all_train_data[-1]
    q.append([nod_addedd.tolist()[0],np.nan])


    new_dataset=q
#        print len(q)
    print 'current_trading_date_is: '+str(dff.index[-1])+' '+str(indexxx)+'trading days after----'



    #print 'ewfhukefhukwehfkwhefukhf',new_dataset.shape




    name=name
    plt.figure(figsize=(16, 9))
    plt.title('inspection is :' + str(name) + 'see it below')
    # plt.plot(dataset,'--')
    plt.plot(trainPredictPlot, 'r-', lw=0.5, label='trainPredictPlot')
    plt.plot(train_Y, 'g.', lw=0.5, label='origin_train_Y')
    #plt.plot(testPredictPlot, 'g-', lw=0.5, label='testPredictPlot')
    #plt.plot(test_Y, 'r.', lw=0.5, label='origin_test_Y')
    plt.title(" current_trading_date_is: "+str(dff.index[-1])+" "+str(indexxx)+" : trading days after----"+20*"#")

    plt.plot(new_stock_plot,'*y',label='projected_one_step')
    plt.legend(loc=0)
    plt.grid(True)



    #train=new_dataset   ###must
    all_train_data.append(new_dataset)
    indexxx+=1


    if switch/100 == 1:
        switch = 0
    else:
        switch += 1
pd.Series(trainY.reshape(-1)).to_csv(str(indexxx)+'days_after.csv')



#plt.show()



        #nod[str(name) + '_train_acc:' + str(trainScore) + '_and_trainPredictPlot'] = pd.Series(
        #    trainPredictPlot.reshape(-1),index=dataframe.index)
        #nod[str(name) + '_train_Y'] = pd.Series(train_Y.reshape(-1),index=dataframe.index)
    #    nod[str(name) + '_test_acc:' + str(testScore) + '_and_testPredictPlot'] = pd.Series(testPredictPlot.reshape(-1),index=dataframe.index)
        #nod[str(name) + '_test_Y'] = pd.Series(test_Y.reshape(-1),index=dataframe.index)
        # all[str(name)+'acc:' +' test_score: '+str(testScore)+' text_score: '+str(trainScore)]=nod
        #nod.index = dataframe.index

        #nod.to_csv('/media/jerry/JERRY/MASTER_project(0)/a_all_data_LSTM_predict/'+str(name)+'_LSTM_result.csv')



# import multiprocessing
# pool=multiprocessing.Pool(processes=4)
# resualt=[]
# for r in range(0,300,2):
#     resualt.append(pool.apply_async(my_test_miti_GPU_train,args=(r,)))
# pool.close()
# #pool.join()










######train_on us_MACR_data
#li=glob('us*')
#dataframe=pd.read_excel(li[0])
#dataframe=dataframe.fillna(method='bfill')
#dataframe=dataframe.fillna(method='ffill')
#dataframe=dataframe.iloc[:,[3,5]].dropna()










# 将整型变为floa

#plt.plot(dataset)
#plt.show()

# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix

import time
# fix random seed for reproducibility



a,b=create_dataset(dataset=dataset,look_back=3,nub_of_the_col_index=0)############changed
# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)

# split into train and test sets






# use this function to prepare the train and test datasets for modeling


# all={}
# nod=pd.DataFrame()
# for i,name in enumerate(dataframe.columns[:2]):
#     print(4*'*'+'working_on: '+str(name)+12*'^')
#
#     look_back = 5
#     trainX, trainY = create_dataset(train, look_back,nub_of_the_col_index=i)
#     testX, testY = create_dataset(test, look_back,nub_of_the_col_index=i)
#
#
#     trainX_scalar,trainY_scalar=MinMaxScaler(),MinMaxScaler()
#     testX_scalar,testY_scalar=MinMaxScaler(),MinMaxScaler()
#     trainX, trainY =trainX_scalar.fit_transform(trainX),trainY_scalar.fit_transform(trainY)
#     testX, testY = testX_scalar.fit_transform(testX),testY_scalar.fit_transform(testY)
#
#
#
#
#
#     # reshape input to be [samples, time steps, features]
#     trainX = numpy.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
#     testX = numpy.reshape(testX, (testX.shape[0],1, testX.shape[1]))
#     # model.add(LSTM(
#     #     input_shape=(periods-1,1),
#     #     output_dim=1,
#     #     return_sequences=True))
#     # create and fit the LSTM network
#
#     model = Sequential()
#     model.add(LSTM(look_back+1, input_shape=(1,look_back),return_sequences=True))
#     # model.add(Dense(200))
#     # model.add(Dropout(0.2))
#     # model.add(LSTM(
#     #     50,
#     #     return_sequences=True))
#     # #
#     # # model.add(Dense(200))
#     # # model.add(Dropout(0.2))
#     # # model.add(LSTM(100,return_sequences=True))
#     # # model.add(Dense(2000))
#     # # model.add(Dropout(0.4))
#     # #
#     # model.add(LSTM(
#     #     200,
#     #     return_sequences=True))
#     # model.add(LSTM(
#     #     100,
#     #     return_sequences=True))
#     # model.add(LSTM(
#     #     20,
#     #     return_sequences=True))
#     model.add(LSTM(
#          50,
#          return_sequences=False))
#     # model.add(Dense(4000))
#     model.add(Dropout(0.6))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     print (model.summary())
#     model.fit(trainX, trainY, epochs=20, batch_size=4, verbose=2,validation_split=0.2)
#
#
#
#     # make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)
#
#
#
#
#
#
#     # invert predictions
#     trainPredict = trainY_scalar.inverse_transform(trainPredict)
#     trainY = trainY_scalar.inverse_transform(trainY)
#     testPredict = testY_scalar.inverse_transform(testPredict)
#     testY = testY_scalar.inverse_transform(testY)
#
#
#     trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
#     print('Test Score: %.2f RMSE' % (testScore))
#
#
#     # shift train predictions for plotting
#     train_Y,trainPredictPlot = numpy.zeros((dataset.shape[0],1)),numpy.zeros((dataset.shape[0],1))
#     #trainPredictPlot[:] = numpy.nan
#     trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
#     train_Y[look_back:len(trainPredict)+look_back]=trainY
#     # shift test predictions for plotting
#     test_Y,testPredictPlot = numpy.zeros((dataset.shape[0],1)), numpy.zeros((dataset.shape[0],1))
#     #testPredictPlot[:,:] = numpy.nan
#     testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict
#     test_Y[len(trainPredict)+(look_back*2)+1:len(dataset)-1]=testY
#     # plot baseline and predictions
#     plt.figure(figsize=(16,9))
#     plt.title('inspection is :'+str(name)+'see it below')
#     #plt.plot(dataset,'--')
#     plt.plot(trainPredictPlot,'r-',lw=0.5,label='trainPredictPlot')
#     plt.plot(train_Y,'g.',lw=0.5,label='origin_train_Y')
#     plt.plot(testPredictPlot,'g-',lw=0.5,label='testPredictPlot')
#     plt.plot(test_Y,'r.',lw=0.5,label='origin_test_Y')
#     plt.legend(loc=0)
#     plt.grid(True)
#
#     nod[str(name)+'_train_acc:'+str(trainScore)+'_and_trainPredictPlot']=pd.Series(trainPredictPlot.reshape(-1))
#     nod[str(name)+'_train_Y']=pd.Series(train_Y.reshape(-1))
#     nod[str(name)+'_test_acc:'+str(testScore)+'_and_testPredictPlot']=pd.Series(testPredictPlot.reshape(-1))
#     nod[str(name)+'_test_Y']=pd.Series(test_Y.reshape(-1))
#     #all[str(name)+'acc:' +' test_score: '+str(testScore)+' text_score: '+str(trainScore)]=nod
# nod.index=dataframe.index
# nod.to_csv('all_stocks_LSTM_prediction.csv')



#plt.show()
