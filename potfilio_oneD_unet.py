#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-6-11上午11:54'
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, AveragePooling2D, UpSampling2D, Dropout, Cropping2D,Deconv2D,Conv1D,UpSampling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from data import *
import sys,time

#reload(sys)
#sys.setdefaultencoding('utf-8')
#import matplotlib.pyplot as plt
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




#from keras.models import Sequential
#from keras.callbacks import ReduceLROnPlateau,TensorBoard
#from keras.layers import Dense,Dropout,Embedding,Conv1D,GlobalAveragePooling1D,MaxPooling1D,AveragePooling1D
#x=[]
#from keras.layers import Flatten,embeddings
#for i in range(10000):
#    qq=np.random.choice([12,21,543,654,23.6,12,5436,345,31,756])
#    x.append(qq)
#y=np.random.choice([1,0],len(x))

import pandas as pd
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#ds=pd.read_excel('/home/jerry/PycharmProjects/untitled/projects/MASTER_project/REIT INDEX.xlsx',header=None)   #2,4,8,16,32
#ds=pd.read_excel('/home/jerry/PycharmProjects/untitled/projects/MASTER_project/repository/sp500.xlsx')[:-2]#[len(ds)-4096,:]  #12 test passed:16  and [:-2]

smooth = 1.


# Tensorflow version for the model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


#ds=ds.iloc[2:,[1,2]]
#ds=ds.fillna(method='bfill')
#ds=ds.fillna(method='ffill')
#ds=ds.values



#from sklearn.cross_validation import train_test_split
#x_train,y_train,x_test,y_test=train_test_split(df,np.array(label),test_size=0.1)
import tensorflow as tf
from tensorflow import Session as sess
from keras.layers import AveragePooling1D,MaxPooling1D,Flatten,Dense,Reshape,BatchNormalization
from keras import callbacks
def get_1d_unet(df):
    #global df

    inputs=Input(shape=(len(df),1))
    #embedding_layer = Embedding(len(df) + 1,
    #                            1,
    #                            trainable=False)
    #embbeding=embedding_layer(inputs)
    conv1=Conv1D(16,kernel_size=5,strides=1,activation='relu',padding='same')(inputs)

   # conv1=BatchNormalization(axis=2)(conv1)
    conv2=Conv1D(16,kernel_size=2,strides=1, activation='relu',padding='same')(conv1)
    conv2=Dropout(0.2)(conv2)
    #conv2=Conv1D(8,4,activation='relu',padding='same')(conv2)
    print('conv2',conv2)
    polling1d=MaxPooling1D(pool_size=2)(conv2)
    #(2048,128)
    conv3=Conv1D(16, kernel_size=2,strides=1, activation='relu',padding='same')(polling1d)
 #   conv3=BatchNormalization(axis=2)(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3=Conv1D(32, kernel_size=5,strides=1, activation='relu',padding='same')(conv3)
    polling2d=MaxPooling1D(pool_size=2)(conv3)
    #(1024,256)
    # conv4=Conv1D(128,4,activation='relu',padding='same')(polling2d)
    # conv4=Dropout(0.2)(conv4)
    # conv4=Conv1D(128,4,activation='relu',padding='same')(conv4)
    # polling3d=MaxPooling1D(pool_size=2)(conv4)
    # #(1024,256)
    conv4=Conv1D(32,kernel_size=5,strides=1,activation='relu',padding='same')(polling2d)
    conv4=Dropout(0.2)(conv4)
    conv4=Conv1D(64,kernel_size=2,strides=1,activation='relu',padding='same')(conv4)
    #conv4 = Conv1D(32, 4, activation='relu', padding='same')(conv4)
    polling3d=MaxPooling1D(pool_size=2)(conv4)
  #  conv4 = Conv1D(1024, 4, activation='relu', padding='same')(conv4)
  #  conv4 = Conv1D(2048, 4, activation='relu', padding='same')(conv4)
    print('conv4',conv4)
    print('conv3',conv3)
 #   print(conv5)
    # #(512,512)
    conv5=Conv1D(64,kernel_size=5,strides=1,activation='relu',padding='same')(polling3d)
    conv5=Dropout(0.1)(conv5)
    conv5=Conv1D(128,kernel_size=2,strides=1,activation='relu',padding='same')(conv5)

    polling4d=MaxPooling1D(pool_size=2)(conv5)

    #(256,1024)
    conv6 = Conv1D(128, kernel_size=5,strides=1, activation='relu', padding='same')(polling4d)
    conv6 = Dropout(0.2)(conv6)
    #conv6 = Conv1D(128, 4, activation='relu', padding='same')(conv6)
    conv6 = Conv1D(256, kernel_size=5,strides=1, activation='relu', padding='same')(conv6)
    polling5d =MaxPooling1D(pool_size=2)(conv6)
    #(128,2048)
    conv7 = Conv1D(256, kernel_size=5,strides=1, activation='relu', padding='same')(polling5d)
    conv7 = Dropout(0.2)(conv7)
    #conv7 = Conv1D(256, 4, activation='relu', padding='same')(conv7)
    conv7 = Conv1D(512, kernel_size=5,strides=1, activation='relu', padding='same')(conv7)
    polling6d =MaxPooling1D(pool_size=2)(conv7)
    #(64,4096)
    conv8 = Conv1D(512, kernel_size=5,strides=1, activation='relu', padding='same')(polling6d)
    conv8 = Dropout(0.4)(conv8)
    #conv8 = Conv1D(512, 4, activation='relu', padding='same')(conv8)
    conv8 = Conv1D(1024, kernel_size=5,strides=1, activation='relu', padding='same')(conv8)
    polling7d =MaxPooling1D(pool_size=2)(conv8)
    #(32,4096*2)
    conv9 = Conv1D(1024, kernel_size=5,strides=1, activation='relu', padding='same')(polling7d)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv1D(2048, kernel_size=2,strides=1, activation='relu', padding='same')(conv9)
    #conv9 = Conv1D(1024, 4, activation='relu', padding='same')(conv9)
    #polling8d = MaxPooling1D(pool_size=2)(conv9)



    #(16,4092*4)
   #  conv10 = Conv1D(2048, 4, activation='relu', padding='same')(polling8d)
   #  conv10 = Dropout(0.2)(conv10)
   #  conv10 = Conv1D(2048, 4, activation='relu', padding='same')(conv10)
   #  conv10 = Conv1D(4096, 4, activation='relu', padding='same')(conv10)
   #  polling9d = MaxPooling1D(pool_size=2)(conv10)
   #  #(8,4096*8)
   #  conv11 = Conv1D(4096, 4, activation='relu', padding='same')(polling9d)
   #  conv11 = Dropout(0.2)(conv11)
   #  conv11 = Conv1D(4092 * 2, 4, activation='relu', padding='same')(conv11)
   #  conv11 = Conv1D(4092 * 4, 4, activation='relu', padding='same')(conv11)
   # #polling4d = MaxPooling1D(pool_size=2)(conv11)





   #change the correspoding_upsampling_layers
    #upsample
    up1=merge([UpSampling1D(size=2)(conv9),conv8],mode='concat',concat_axis=2)
    print('up1',up1)
    upconv1=Conv1D(1024,kernel_size=5,strides=1,activation='relu',padding='same')(up1)
    #with tf.Session() as sess:

    #    sess.run(tf.global_variables_initializer())
    #    print(upconv1.eval())
    upconv1=Dropout(0.3)(upconv1)
    upconv1=Conv1D(1024,kernel_size=2,strides=1,activation='relu',padding='same')(upconv1)
    #(64,)
    print('cov6',conv6)
    print('conv2',conv2)
    print('conv3',conv3)

  #  conv6 = Dropout(0.2)(conv6)
  #  conv6=Conv1D(320,4,activation='relu',padding='same')(conv6)
    #(32,4096*4)
#    print('up2',up2)
    #(128,)
    up2=merge([UpSampling1D(size=2)(upconv1),conv7],mode='concat',concat_axis=2)
    upconv2 = Conv1D(512, kernel_size=5,strides=1, activation='relu', padding='same')(up2)
    upconv2 = Dropout(0.4)(upconv2)
    upconv2 = Conv1D(512, kernel_size=5,strides=1, activation='relu', padding='same')(upconv2)
    #(256,4096*2)

    up3 = merge([UpSampling1D(size=2)(upconv2), conv6], mode='concat', concat_axis=2)
    upconv3 = Conv1D(256, kernel_size=2,strides=1, activation='relu', padding='same')(up3)
    upconv3 = Dropout(0.4)(upconv3)
    upconv3 = Conv1D(256, kernel_size=2,strides=1, activation='relu', padding='same')(upconv3)
    #(512,4096)
    up4= merge([UpSampling1D(size=2)(upconv3), conv5], mode='concat', concat_axis=2)
    upconv4 = Conv1D(128 , kernel_size=2,strides=1, activation='relu', padding='same')(up4)
    upconv4 = Dropout(0.4)(upconv4)
    upconv4 = Conv1D(128 , kernel_size=2,strides=1, activation='relu', padding='same')(upconv4)
    #(1024,2048)
    up5 = merge([UpSampling1D(size=2)(upconv4), conv4], mode='concat', concat_axis=2)
    upconv5 = Conv1D(64, kernel_size=2,strides=1, activation='relu', padding='same')(up5)
    upconv5 = Dropout(0.4)(upconv5)
    upconv5 = Conv1D(64, kernel_size=2,strides=1, activation='relu', padding='same')(upconv5)
    #(2048,1024)
    up6 = merge([UpSampling1D(size=2)(upconv5), conv3], mode='concat', concat_axis=2)
    upconv6 = Conv1D(32, kernel_size=5,strides=1, activation='relu', padding='same')(up6)
    upconv6 = Dropout(0.4)(upconv6)
    upconv6 = Conv1D(32, kernel_size=2,strides=1, activation='relu', padding='same')(upconv6)
    #(4096,512)
    print('up6:',upconv6)
    up7 = merge([UpSampling1D(size=2)(upconv6), conv2], mode='concat', concat_axis=2)
    upconv7 = Conv1D(16, kernel_size=2,strides=1, activation='relu', padding='same')(up7)
    upconv7 = Dropout(0.4)(upconv7)
    upconv7 = Conv1D(16, kernel_size=2,strides=1, activation='relu', padding='same')(upconv7)




    #
    #
    # #(2048,128)
    # up8 = merge([UpSampling1D(size=2)(upconv7), conv3], mode='concat', concat_axis=2)
    # upconv8 = Conv1D(128, 4, activation='relu', padding='same')(up8)
    # upconv8 = Dropout(0.2)(upconv8)
    # upconv8 = Conv1D(128, 4, activation='relu', padding='same')(upconv8)
    # #(4096,64)
    # up9 = merge([UpSampling1D(size=2)(upconv8), conv2], mode='concat', concat_axis=2)
    # upconv9 = Conv1D(64, 4, activation='relu', padding='same')(up9)
    # upconv9 = Dropout(0.2)(upconv9)
    # upconv9 = Conv1D(64, 4, activation='relu', padding='same')(upconv9)
    #
    convend=Conv1D(8,kernel_size=2,strides=1,activation='relu',padding='same')(upconv7)
   # convend=BatchNormalization(axis=2)(convend)
    convend=Dropout(0.4)(convend)
    convend=Conv1D(5,kernel_size=2,strides=1,activation='relu',padding='same')(convend)
    convend=Dropout(0.2)(convend)
    convend=Conv1D(3,kernel_size=2,strides=1,activation='relu',padding='same')(convend)
    convend=Flatten()(convend)
    print(convend)
    convend=Reshape((-1,3))(convend)
    convend=Dropout(0.2)(convend)
    convend=Dense(500,activation='relu')(convend)
    convend = Dropout(0.4)(convend)
    convend = Dense(1, activation='sigmoid')(convend)
    # conv7 =Conv1D(120,3,activation='relu',padding='same')(conv7)
    # conv7 =Dropout(0.2)(conv7)
    # conv7 =Conv1D(3,3,activation='relu',padding='same')(conv7)
    # conv7 = Conv1D(1, 3, activation='relu', padding='same')(conv7)
    # print('conv7',conv7)
    # print('conv2',conv2)
    #
    #
#    up3=merge([UpSampling1D(size=2)(conv7),conv2],mode='concat',concat_axis=2)
    # globlepooling=Flatten()(conv4)
    # dropout=Dropout(0.9)(globlepooling)
    # dense=Dense(len(label), activation='softmax')(dropout)
    model=Model(inputs=inputs,outputs=convend)
#    model=Model(inputs=inputs,outputs=dense)

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['acc'])   #########  binary_crossentropy
    #tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    #
    #cbks = [tb_cb]
    #history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_
    # ,
    #                    verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))
    print(model.summary())
    return model
#check=ModelCheckpoint(filepath='/home/jerry/PycharmProjects/untitled/venv/computer/unet/conv1_unet_test_model_log/')

from keras.callbacks import History,LearningRateScheduler,ModelCheckpoint
from keras.callbacks import History
#his=History()
#model.fit(np.array(x).reshape(1,len(x),1),y.reshape(1,len(y)),batch_size=4,epochs=4)
#          callbacks=[check])

#model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss<10%', save_best_only=True)
from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler
from keras.models import load_model
import time
#import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/1-np.exp(-x)


def test(struddle,df,labels,NUM_col):
    global cutted_full_index
    #time.sleep(30)
  #  df= np.log(df/df.shift(1))[1:4097]
   # labels=labels[:4096]   #test it
   # df=df[:4096]
   # labels=labels[:4096]
 #   cur=sigmoid(df.values)
    cur=MinMaxScaler().fit_transform(df)
    df=pd.DataFrame(cur,columns=df.columns)   #cur.reshape(-1)



    model = get_1d_unet(df)
    #model.load_weights(str(struddle)+'model.h5')
    #model.load('model.h5')

    tb_cb = callbacks.TensorBoard(log_dir='log')

    cbks = [tb_cb]
    #history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #                    verbose=1, callbacks=cbks, validation_data=(X_test, Y_test))
    pd.DataFrame(df).to_csv('df.csv')

    #mea=np.mean(df.values,axis=0)
    #std=np.std(df.values,axis=0)
    #train_df=(df-mea) / std
    train_df=StandardScaler().fit_transform(df)   #changed it
    train_df=df.values
    #print(np.array(train_df))
    #train_df=df.values
    #q=MinMaxScaler()

    pd.DataFrame(train_df).to_csv('train_df.csv')
    #train_df=q.fit_transform(train_df)
    #train_df = train_df.fillna(method='bfill')
    #train_df = train_df.fillna(method='ffill')
    #pd.DataFrame(train_df).plot()
    #plt.show()
    model.fit(train_df.reshape(-1,len(df),1),np.array(labels).reshape(-1,len(labels),1),
              verbose=1,batch_size=8,epochs=3)#validation_split=0.2)#,callbacks=[model_checkpoint])  # acc:  around better 70%
    loss,acc=model.evaluate(train_df.reshape(-1,len(df),1),np.array(labels).reshape(-1,len(labels),1))
    #print(df.shape,np.array(labels).shape)

    #loss=model.predict(scale(df).reshape(1,len(df),1))-np.array(label).reshape(1,len(label),1)
   # model.save(str(struddle)+'model.h5')
    json_string = model.to_json()  # 等价于 json_string = model.get_config()
    open('./models/'+str(struddle)+'_model_architecture.json', 'w').write(json_string)
    model.save_weights('./models/'+str(struddle)+'_model_weights.h5')
    # 加载模型数据和weights
    #model = model_from_json(open('my_model_architecture.json').read())
    #model.load_weights('my_model_weights.h5')






    #model.save_weights(str(struddle)+'model.h5')
    w={}
    w['label']=labels
    w['predicted']=pd.DataFrame(model.predict(train_df[:,NUM_col].reshape(-1,len(df),1)).reshape(4096,-1))
    w['oringin']=df
   #  model.save_weights(str(struddle)+'model.h5')
    del model

    w['predicted'].to_csv(str(struddle)+'label.csv')
    #df=pd.read_csv('label.csv',header=None)
    #print(w)
    print('-----------inspiring_1st_data_predicted--------------')

    w=w['predicted']

    w=w.astype('float64')
    q=w.describe()
    #print(q.head())
    print(20*'@'+'describe:'+'#'*10)
    #print('\ntoday_is_',df.index[0])



    print('\n')

    for seq,k in enumerate([df.columns[NUM_col]]):
        print(seq)
        if w.iloc[0,seq]<=q.iloc[:,seq]['25%']:

            #print('tomorrow is bigger than today')
            #print( k,cutted_full_index.iloc[0,seq],'bigger',acc)
            yield k,cutted_full_index.iloc[0,seq],'bigger',acc,train_df
        else:
            #print('tomorrow is lowwer than today')
            #print( k,cutted_full_index.iloc[0,seq],'lowwer',acc)
            yield k,cutted_full_index.iloc[0,seq],'lowwer',acc,train_df


import os
import threading,gc
import sys
#reload(sys)
#sys.setdefaultencoding('ISO-8859-1')`
#final=[
file_list=os.listdir('factors/avalible_REITs')[:]
def normalize_raw_data(df):
    j=[]
#    j.append(0)
    for i in df:
        if np.isnan(i) == True:
            #print(len(j))
            j.append(np.mean(j[-5:]))
        else:
            j.append(i)

    #j=j[1:]
    #print(np.mean(j))
    return pd.Series(j,index=df.index)


def make_df_from_files(file_list):
    index=pd.DataFrame()
    w=pd.DataFrame()
    for i in file_list:
        data=pd.read_csv('factors/avalible_REITs/'+i,header=None)
        #data=data.sort_index(ascending=False)
       # data=data[:5000]

        #data=normalize_raw_data(dat.iloc[:,2])


        data=data.fillna(method='bfill')
        data=data.fillna(method='ffill')
        #data = data.fillna(method='bfill')
        #data = data.fillna(method='ffill')
        index[i]=data.iloc[:,0]
        close=data.iloc[:,1]
        w[i]=close    #has changed
        #w=w.fillna(method='bfill')
        #w = w.fillna(method='ffill')
        print('file: '+str(i)+' _Loading:Done')
    return w,index
full_df,full_index=make_df_from_files(file_list)



def make_one_col_label(new):
    new=new.values
    label=[]
    label.append(1)
    for i in range(len(new)-1):
        if new[i+1]>=new[i]:
            label.append(1)
        else:
            label.append(0)

    return pd.Series(label)
#with open('results.txt','w') as f:
#def MainRange(start,stop):
save=pd.DataFrame()
name, current_date, next_trading_day, acc_is = [], [], [], []
next_trading_mode=[]
#for s in range(2):

def Main(start,end,interval,NUM_col=0):
    global cutted_full_index
    for i in range(start,end,interval):
        print('\nDone\n')
        print(13*'*'+'predicting:'+str(i)+'steps'+'*'*12)
        print('colling_the_gpu(needed)')
        time.sleep(5)
        print(13*'%'+'starting_current_processing'+'%'*12)

        #ds = pd.read_excel('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/equity+market/aapl.xlsx')
        #ds = ds.fillna(method='bfill')
        #ds = ds.fillna(method='ffill')


        test_num=0
        df = full_df[i+test_num:4096 + i+test_num ]  # cut the oringin data
        cutted_full_index=full_index[test_num+i:test_num+4096+i]
        #df = ds.iloc[0:, [1, 2]]
        #df=df.dropna(axis=0,how='any')
        #df.index = df.iloc[:, 0]
        #df = df.iloc[:, 1]
        # new train

        #df = df.fillna(method='bfill')
        #df = df.fillna(method='ffill')
        labels = pd.DataFrame()
        for seq,j in enumerate(df.columns):
            new=df.iloc[:,seq]
            labels[j]=make_one_col_label(new)


        #print(df.shape,np.array(label).shape)
        #a,b,c=test(df,labels)   #like 3tupple: ('ge.xlsx', '03/20/2018', 'lowwer', 0.12144715338945389)
                                #('aapl.xlsx', '03/20/2018', 'lowwer', 0.12144715338945389)


        struddle=i
        for a,b,c,d,oringn_df in test(struddle,df,labels,NUM_col=NUM_col):
            name.append(a)
            current_date.append(b)
            next_trading_day.append(c)
            acc_is.append(d)
            #f.write(str(['name:',a,'current_date:',b,'next_trading_day:',c,'acc_is:',d])+'\n')

    save['name']=pd.Series(name)
    save['current_date']=pd.Series(current_date)
    save['next_trading_day']=pd.Series(next_trading_day)
    save['acc_is']=pd.Series(acc_is)
    print(save)
    return save,oringn_df,labels.iloc[:,NUM_col]

#if sum(save['acc_is'])>=2.2: #and np.std(save['acc_is'])<=0.08:
#    print ('can use')
#else:
#    print ('try again')

def jugment(cur_index,full_df,NUM_col):

    if full_df.iloc[cur_index - 1, NUM_col] >= full_df.iloc[cur_index, NUM_col]:
        return 'bigger'
    else:
        return 'lowwer'



def evaluate_mode(defined_df,NUM_col):
    global full_df,full_index
    j=0
    good=[]
    for seq,i in enumerate(defined_df['current_date']):
        cur_index=full_index[full_index.iloc[:,0]==i].index[0]
       # print(cur_index)
        if cur_index == 0:
            print('model '+str(seq)+' contains_unknown_predict')
            continue
        else:
           # if defined_df['next_trading_day'][cur_index]==jugment(cur_index,full_df):
            if defined_df['next_trading_day'][seq] == jugment(cur_index, full_df,NUM_col=NUM_col):
                j+=1
                good.append(seq)
                print(str(seq)+' model_is_correct\n')
            #else:
             #   good.append(seq)
    print('\nrealized_acc(nomarator contains first Index):',j/len(defined_df.index))
    return good,j/len(defined_df.index)
NUM_col=29
resualt_file,oring_df,index_col=Main(start=0,end=11,interval=1,NUM_col=NUM_col)
good,score=evaluate_mode(resualt_file,NUM_col=NUM_col)


pre=pd.read_csv('factors/avalible_REITs/'+file_list[NUM_col],header=None)
pre=MinMaxScaler().fit_transform(pre.iloc[:,1].values.reshape(-1,1))[:4096]
#from keras.models import load_model

look=[]

if index_col.sum()/len(index_col) >= 0.5:
    most_offen='lowwer'
    other='bigger'
else:
    most_offen='bigger'
    other='lowwer'
if len(good)>=1 or len(good)%2==1:
    for g in good:
        new_model = model_from_json('./models/'+open(str(g)+'_model_architecture.json').read())
        new_model.load_weights('./models/'+(str(g))+'_model_weights.h5')
       # new_model=load_model(str(good[0])+'model.h5',custom_objects={'dice_coef_loss_loss':dice_coef_loss})
        pre=new_model.predict(pre.reshape(-1,4096,1))
        pre=pd.Series(pre.reshape(-1)).astype('float64')

        q=pre.describe()
        if pre[0] <= q['mean']:

            #print('tomorrow is bigger than today')
            #print( k,cutted_full_index.iloc[0,seq],'bigger',acc)
    #        print ('next traiding day is lowwer')
            look.append(1)
        else:
            #print('tomorrow is lowwer than today')
            #print( k,cutted_full_index.iloc[0,seq],'lowwer',acc)
      #      print ('next traiding day is bigger')
            look.append(0)
    if np.mean(look) >= 0.5:
        print('next traiding day is %s' % most_offen)
    else:
        print('next traiding day is %s or same' % other)
else:
    print ('try agian (due to statues of GPU)')




#next_trading_mode.append(save['next_trading_day'])
        #os.system('reset')
    #for i in locals().keys():
    #    del i
    #gc.collect()


        #final.append([a,b,'acc_is:',c])
        #os.system('clear')
        #return [a,b,'acc_is:',c]
# threads=[]
# t1 = threading.Thread(target=MainRange,args=(0,4))
# threads.append(t1)
# t2 = threading.Thread(target=MainRange,args=(4,8))
# threads.append(t2)
#
# t3 = threading.Thread(target=MainRange,args=(8,12))
# threads.append(t3)
#
# t4 = threading.Thread(target=MainRange,args=(12,16))
# threads.append(t4)
#
# t5 = threading.Thread(target=MainRange,args=(16,20))
# threads.append(t5)
#
# for t in threads:
#     t.setDaemon(True)
#     t.start()
# t.join()
# print ("ok")


#print(final)
    ###and predict##

    # ds=pd.read_excel('/home/jerry/PycharmProjects/untitled/projects/MASTER_project/aapl.xlsx')
    # ds=ds[7:4096+7]   #cut the oringin data
    # df=ds.iloc[0:,[1,2]]
    # df.index=df.iloc[:,0]
    # df=df.iloc[:,1]ryman hospitall
    # #new train
    #
    # df=df.fillna(method='bfill')
    # df=df.fillna(method='ffill')
    #
    # w=pd.DataFrame()
    # w['label']=pd.Series(label)
    # w['predicted']=pd.Series(model.predict(scale(df).reshape(1,len(df),1)).reshape(-1))
    # w['oringin']=pd.Series(df.values)
#w.to_csv('label.csv')
#df=pd.read_csv('label.csv',header=None)

# print('-----------inspiring_2nd_data_predicted--------------')
# w=w.astype('float64')
# q=w.describe()
#
# print('\ntoday_is_',df.index[0])
#
#
#
# print('\n')
#
#
# if w.iloc[0,1]==q.iloc[:,1]['min']:
#
#     print('tomorrow is bigger than today')
# else:
#     print('tomorrow is lowwer than today')
#