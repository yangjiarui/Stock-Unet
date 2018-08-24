import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, AveragePooling2D, UpSampling2D, Dropout, Cropping2D,Deconv2D,Conv1D,UpSampling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from data import *
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
#
# class myUnet(object):
#
#     def __init__(self, img_rows=64, img_cols=64):
#         self.img_rows = img_rows
#         self.img_cols = img_cols
#
#     def load_data(self):
#         mydata = dataProcess(self.img_rows, self.img_cols)
#         imgs_train, imgs_mask_train = mydata.load_train_data()
#         imgs_test = mydata.load_test_data()
#         return imgs_train, imgs_mask_train, imgs_test
#
#     def get_unet(self):
#         inputs = Input((self.img_rows, self.img_cols, 1))
#
#         '''
#         unet with crop(because padding = valid)
#
#         conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
#         print "conv1 shape:",conv1.shape
#         conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
#         print "conv1 shape:",conv1.shape
#         crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
#         print "crop1 shape:",crop1.shape
#         pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#         print "pool1 shape:",pool1.shape
#
#         conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
#         print "conv2 shape:",conv2.shape
#         conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
#         print "conv2 shape:",conv2.shape
#         crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
#         print "crop2 shape:",crop2.shape
#         pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#         print "pool2 shape:",pool2.shape
#
#         conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
#         print "conv3 shape:",conv3.shape
#         conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
#         print "conv3 shape:",conv3.shape
#         crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
#         print "crop3 shape:",crop3.shape
#         pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#         print "pool3 shape:",pool3.shape
#
#         conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
#         conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
#         drop4 = Dropout(0.5)(conv4)
#         crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
#         pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#         conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
#         conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
#         drop5 = Dropout(0.5)(conv5)
#
#         up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#         merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
#         conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
#         conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
#
#         up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
#         merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
#         conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
#         conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
#
#         up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
#         merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
#         conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
#         conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
#
#         up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
#         merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
#         conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
#         conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
#         conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
#         '''
#
#         conv1 = Conv1D(64, 4, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#         print ("conv1 shape:", conv1.shape)
#         conv1 = Conv1D(64, 4, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#         print "conv1 shape:", conv1.shape
#         pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
#         print "pool1 shape:", pool1.shape
#
#         conv2 = Conv1D(128, 4, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#         print "conv2 shape:", conv2.shape
#         conv2 = Conv1D(128, 4, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#         print "conv2 shape:", conv2.shape
#         pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
#         print "pool2 shape:", pool2.shape
#
#         conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#         print "conv3 shape:", conv3.shape
#         conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#         print "conv3 shape:", conv3.shape
#         pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
#         print "pool3 shape:", pool3.shape
#
#         conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#         conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#         drop4 = Dropout(0.5)(conv4)
#         pool4 = AveragePooling2D(pool_size=(2, 2))(drop4)
#
#         conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#         conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#         drop5 = Dropout(0.5)(conv5)
#
#         up6 = Conv1D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling1D(size=(2,2))(drop5))
#         merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
#         conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#         conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#         up7 = Conv1D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling1D(size=(2,2))(conv6))
#         merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
#         conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#         conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#         up8 = Conv1D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling1D(size=(2,2))(conv7))
#         merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
#         conv8 =Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#         conv8 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#         up9 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#             UpSampling2D(size=(2, 2))(conv8))
#         merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
#         conv9 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#         conv9 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#         conv9 = Conv1D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#         conv10 = Conv1D(1, 1, activation='sigmoid')(conv9)
#
#         model = Model(input=inputs, output=conv10)
#
#         model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
#
#         return model
#
#     def train(self):
#         print("loading data")
#         imgs_train, imgs_mask_train, imgs_test = self.load_data()
#         print("loading data done")
#         model = self.get_unet()
#         print("got unet")
#
#         model_checkpoint = ModelCheckpoint('editted_unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
#         print('Fitting model...')
#         print(model.summary())
#         model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, verbose=1, validation_split=0.2, shuffle=True,
#                   callbacks=[model_checkpoint])
#
#         print('predict test data')
#         imgs_mask_test = model.predict(imgs_test, batch_size=2, verbose=1)
#         np.save('/home/jack/PycharmProjects/untitled/unet/predicted_index/test_seged.npy', imgs_mask_test)  #/home/jack/PycharmProjects/untitled/unet/resualts/imgs_mask_test.npy
#
#     def save_img(self):
#         print("array to image")
#         imgs = np.load('/home/jack/PycharmProjects/untitled/unet/predicted_index/test_seged.npy')
#         for i in range(imgs.shape[0]):
#             img = imgs[i]
#             img = array_to_img(img)
#             img.save("/home/jack/PycharmProjects/untitled/unet/resualts/%d.jpg" % (i))
#  #           print('regerge',img.shape)
#          #   plt.figure()
#          #   plt.imshow(img.reshape(512,512))
#          #   plt.show()
#
#
#
#
# #if __name__ == '__main__':
# #    myunet = myUnet()
# #    myunet.train()
# ##      myunet.save_img()
#




from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense,Dropout,Embedding,Conv1D,GlobalAveragePooling1D,MaxPooling1D,AveragePooling1D
x=[]
from keras.layers import Flatten,embeddings
for i in range(10000):
    qq=np.random.choice([12,21,543,654,23.6,12,5436,345,31,756])
    x.append(qq)
y=np.random.choice([1,0],len(x))

import pandas as pd
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#ds=pd.read_excel('/home/jerry/PycharmProjects/untitled/venv/computer/unet/my test/REIT INDEX.xlsx',header=None)   #2,4,8,16,32
ds=pd.read_excel('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/repository/sp500.xlsx')[16:-2]  #12 test passed:16  and [:-2]

df=ds.iloc[0:,[0,1]]
df.index=df.iloc[:,0]
df=df.iloc[:,1]
#new train

df=df.fillna(method='bfill')
df=df.fillna(method='ffill')

smooth = 1.


# Tensorflow version for the model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


ds=ds.iloc[2:,[1,2]]
ds=ds.fillna(method='bfill')
ds=ds.fillna(method='ffill')
ds=ds.values


label=[]
label.append(1)
for i in range(len(df)-1):
    if df[i+1] >= df[i]:
        label.append(1)
    else:
        label.append(0)

from sklearn.cross_validation import train_test_split
x_train,y_train,x_test,y_test=train_test_split(df,np.array(label),test_size=0.1)



def get_1d_unet():
    inputs=Input(shape=(len(df),))
    embedding_layer = Embedding(len(df) + 1,
                                1,
                                trainable=False)
    embbeding=embedding_layer(inputs)
    conv1=Conv1D(64,4,activation='relu',padding='same')(embbeding)
    conv2=Conv1D(64,4, activation='relu',padding='same')(conv1)
    conv2=Dropout(0.2)(conv2)
    conv2=Conv1D(128,4,activation='relu',padding='same')(conv2)
    print('conv2',conv2)
    polling1d=MaxPooling1D(pool_size=2)(conv2)
    #(1560,64)
    conv3=Conv1D(128, 4, activation='relu',padding='same')(polling1d)
    conv3 = Dropout(0.2)(conv3)
    conv3=Conv1D(256, 4, activation='relu',padding='same')(conv3)
    polling2d=MaxPooling1D(pool_size=2)(conv3)
    #(780,128)
    # conv4=Conv1D(128,4,activation='relu',padding='same')(polling2d)
    # conv4=Dropout(0.2)(conv4)
    # conv4=Conv1D(128,4,activation='relu',padding='same')(conv4)
    # polling3d=MaxPooling1D(pool_size=2)(conv4)
    # #(390,256)
    conv4=Conv1D(256,4,activation='relu',padding='same')(polling2d)
    conv4=Dropout(0.2)(conv4)
    conv4=Conv1D(256,4,activation='relu',padding='same')(conv4)
    conv4 = Conv1D(512, 4, activation='relu', padding='same')(conv4)
  #  conv4 = Conv1D(1024, 4, activation='relu', padding='same')(conv4)
  #  conv4 = Conv1D(2048, 4, activation='relu', padding='same')(conv4)
    print('conv4',conv4)
    print('conv3',conv3)
 #   print(conv5)
    # #(1560)


    up1=merge([UpSampling1D(size=2)(conv4),conv3],mode='concat',concat_axis=2)
    print('up1',up1)
    conv6=Conv1D(128,4,activation='relu',padding='same')(up1)
    conv6=Dropout(0.2)(conv6)
    conv6=Conv1D(128,4,activation='relu',padding='same')(conv6)

    print('cov6',conv6)
    print('conv2',conv2)
    print('conv3',conv3)
    up2=merge([UpSampling1D(size=2)(conv6),conv2],mode='concat',concat_axis=2)
  #  conv6 = Dropout(0.2)(conv6)
  #  conv6=Conv1D(320,4,activation='relu',padding='same')(conv6)

    print('up2',up2)
    conv7 = Conv1D(640, 3, activation='relu', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv1D(320, 3, activation='relu', padding='same')(conv7)
    conv7 =Conv1D(120,3,activation='relu',padding='same')(conv7)
    conv7 =Dropout(0.2)(conv7)
    conv7 =Conv1D(3,3,activation='relu',padding='same')(conv7)
    conv7 = Conv1D(1, 3, activation='relu', padding='same')(conv7)
    print('conv7',conv7)
    print('conv2',conv2)
#    up3=merge([UpSampling1D(size=2)(conv7),conv2],mode='concat',concat_axis=2)



    # globlepooling=Flatten()(conv4)
    # dropout=Dropout(0.9)(globlepooling)
    # dense=Dense(len(label), activation='softmax')(dropout)


    model=Model(inputs=inputs,outputs=conv7)
#    model=Model(inputs=inputs,outputs=dense)
    model.compile(loss=dice_coef_loss,
                  optimizer='sgd',
                  metrics=[dice_coef])   #########  binary_crossentropy
    print(model.summary())
    return model
#check=ModelCheckpoint(filepath='/home/jerry/PycharmProjects/untitled/venv/computer/unet/conv1_unet_test_model_log/')

from keras.callbacks import History,LearningRateScheduler,ModelCheckpoint
from keras.callbacks import History
#his=History()
#model.fit(np.array(x).reshape(1,len(x),1),y.reshape(1,len(y)),batch_size=4,epochs=4)
#          callbacks=[check])
model=get_1d_unet()
model.fit(df.reshape(1,len(df)),np.array(label).reshape(1,len(label),1),batch_size=2,epochs=150)  # acc:  around better 70%
loss=model.predict(df.reshape(1,len(df)))-np.array(label).reshape(-1,len(label),1)
w=pd.DataFrame()
w['label']=pd.Series(label)
w['predicted']=pd.Series(model.predict(df.reshape(1,len(df))).reshape(-1))
w['oringin']=pd.Series(df.values)
w.to_csv('/home/label.csv')








