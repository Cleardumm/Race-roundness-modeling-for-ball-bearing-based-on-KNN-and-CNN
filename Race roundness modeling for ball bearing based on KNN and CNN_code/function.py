#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
import math
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import operator



def BasisFunIdent(x,y,unit,num_b,min_i):
"""
    author: Peng Zhang, Xuanfu Huang, Xiancheng Ouyang
    Development date: 22.Oct.2021
    Version: python 3.x
    Contact Email: zhangpeng960321@gmail.com
    param x: list, The location or time of the signal measurement
    param y: list, x corresponding measured value
    param unit: float, unit of param x
    param num_b: int, number of target basis functions
    param min_i: int, the minimum interval between extreme points
    return: tuple(list of Period and amplitude pairs of all possible basis functions,
            numpyArray of the period and amplitude pair of the basis function obtained by classification)
"""
    def kmeans_building(x1,x2,types_num,types,colors,shapes):#function of KNN
        X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
        kmeans_model = KMeans(n_clusters=types_num).fit(X)
        x1_result=[]; x2_result=[]
        for i in range(types_num):
            temp=[]; temp1=[]
            x1_result.append(temp)
            x2_result.append(temp1)
        for i, l in enumerate(kmeans_model.labels_):
            x1_result[l].append(x1[i])
            x2_result[l].append(x2[i])
      

        return kmeans_model,x1_result,x2_result

    max_v = []
    max_index = []
    min_v = []
    min_index = []
    max_val=[]
    min_val=[]
    max_in=[]
    min_in=[]
    for i in range(len(x)-2):#get local maximum and local minimam
        if y[i] <= y[i+1] and y[i+1] >= y[i+2]:
            max_v.append(y[i+1])
            max_in.append(i+1)
        elif y[i] >= y[i+1] and y[i+1] <= y[i+2]:
            min_v.append(y[i+1])
            min_in.append(i+1)
   
        
    index1=max_in+min_in
    
    index2=sorted(index1)
    
    index3=[]
    x_new=[]
    for i in range(len(index2)-1):#Filter for extremes with spacing greater than 3
        if index2[i+1]-index2[i]>=min_i :
            index3.append(index2[i])
    for i in range(len(index3)-2):
        if y[index3[i]]<=y[index3[i+1]] and y[index3[i+1]]>=y[index3[i+2]]:
            max_val.append(y[index3[i+1]])
            max_index.append(index3[i+1])
        elif y[index3[i]]>=y[index3[i+1]] and y[index3[i+1]]<=y[index3[i+2]]:
            min_val.append(y[index3[i+1]])
            min_index.append(index3[i+1])
   
    if len(max_val)==0:#if there is no local extreme, get the global extreme
        max_val.append(np.max(y))
        
        max_ind,_=max(enumerate(list(y)), key=operator.itemgetter(1))
        max_index.append(max_ind)
        
        
    if len(min_val)==0:
        min_val.append(np.min(y))
        
        min_ind,_=min(enumerate(list(y)), key=operator.itemgetter(1))
        min_index.append(min_ind)
        
    max_index=unit*np.array(max_index)
    min_index=unit*np.array(min_index)
    
    H = []
    B = []
    for j in range(len(min_val)):#calculate the H and B
        for k in range(len(max_val)):
            if min_index[j] < max_index[k] and max_val[k] - min_val[j] > 0: 
                H.append(max_val[k] - min_val[j])
                B.append(max_index[k] - min_index[j])
                break
    if len(B)==0:
        H.append(max_val[0]-min_val[0])
        B.append(abs(max_index[0] - min_index[0]))
  
    
    colors = ['b', 'g', 'r'] 
    shapes = ['o', 's', 'D'] 
    labels=['A','B','C'] 
    if len(B)>=num_b:#cluster the H and B

        kmeans_model,x1_result,x2_result=kmeans_building(H, B, num_b, labels, colors, shapes)
    else:
        kmeans_model,x1_result,x2_result=kmeans_building(H, B, len(B), labels, colors, shapes)

    HBbasis=kmeans_model.cluster_centers_
    
    return list(zip(H,B)),HBbasis
    
   





 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.optimize as optimize
import linecache



def RoundSampleSplit(data_original,sample_length,times,mode):
"""
    author: Peng Zhang, Xuanfu Huang, Xiancheng Ouyang
    Development date: 22.Oct.2021
    Version: python 3.x
    Contact Email: zhangpeng960321@gmail.com
    param data_original: data in logfile
    param sample_length: int, the length of one sample
    param times: int, Multiplier between the number of samples and the number of original data
    param mode: int, 0 or 1, 0:get the sample which's number is the same as original data. 1:get the sample which's number is x times than original data
    return: numpyArray: sample data  dict: sample data with their num.
"""
    def str_Nempty(s):#get data from txt file
        return list(filter(lambda s: s and s.strip(),s))
    if mode ==0:
        
        tr_data = []#get test data
        for te_da in data_original:
            line_num = 1
            length_file = len(open(te_da, encoding='utf-8').readlines())
            while line_num <= length_file:
                line = linecache.getline(te_da, line_num)
                line = line.strip()
                tr_data.append(line)
                line_num = line_num + 1
        tr_data=str_Nempty(tr_data)
        
        tr_data=pd.to_numeric(tr_data)#Make only numbers exist in the array and delete the empty set

        a=np.nan_to_num(tr_data)
        train_data=[]
        j1=0
        for i in range(int(len(a)/sample_length)):#Sort the training set by circles
            train_data.append(a[j1:j1+sample_length])
            j1+=sample_length
        sample_data=np.array(train_data)
        sample_data_num={}
        for i,j in enumerate(sample_data):
            sample_data_num[i]=j
            
    elif mode ==1:
        tr_data = []#get train data
        for tr_da in data_original:
            length_file = len(open(tr_da, encoding='utf-8').readlines())
            for step in range(times):#get 10 times train data
                line_num = int(sample_length/times) * step + 1
                while line_num <= length_file:
                    line = linecache.getline(tr_da, line_num)
                    line = line.strip()
                    tr_data.append(line)
                    line_num = line_num +1
                line_num1 = 1
                while line_num1 < int(sample_length/times) * step + 1:
                    line1 = linecache.getline(tr_da, line_num1)
                    line1 = line1.strip()
                    tr_data.append(line1)
                    line_num1 = line_num1 + 1
        tr_data=str_Nempty(tr_data)
        
        tr_data=pd.to_numeric(tr_data)#Make only numbers exist in the array and delete the empty set

        a=np.nan_to_num(tr_data)
        train_data=[]
        j1=0
        for i in range(int(len(a)/sample_length)):#Sort the training set by circles
            train_data.append(a[j1:j1+sample_length])
            j1+=sample_length
        sample_data=np.array(train_data)
        sample_data_num={}
        for i,j in enumerate(sample_data):
            sample_data_num[i]=j
    return sample_data,sample_data_num


# In[ ]:


def NormalizeMult(data):#function of nomolization


    normalize = np.arange(2*data.shape[1],dtype='float64')
    normalize = normalize.reshape(data.shape[1],2)

    for i in range(0,data.shape[1]):

        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])

        normalize[i,0] = listlow
        normalize[i,1] = listhigh

        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta

    return  data,normalize


# In[ ]:


import tkinter
import pandas as pd
import keras 
import tensorflow
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.datasets import imdb

import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Dense,Dropout,Conv1D,Flatten,LSTM,Input,MaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping
import os

import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def RoundPre(train_X,train_Y,test_X,test_Y,mode,test_num,train_num,train_nom,test_nom):
"""
    author: Peng Zhang, Xuanfu Huang, Xiancheng Ouyang
    Development date: 22.Oct.2021
    Version: python 3.x
    Contact Email: zhangpeng960321@gmail.com
    param train_X: matrix: input of train data
    param train_Y: matirx: label of train data
    param test_X: matrix: input of test data
    param test_Y: matrix: original data of test data
    param mode: int: 0,1,2,3  0:CNN 1:LSTM 2:2CNN+1LSTM(series) 3:2CNN+LSTM(parallel)
    param test_num: int :number of output data of predict test data
    param train_num: int :number of output data of predict train data
    train_nom: return of NormalizeMult by train data
    test_nom: return of NormalizeMult by test data
    return: list:RMSE of predict test data, list:RMSE of predict train data, list: predict test data, list: predict train data
"""
    def NormalizeMult(data):#function of nomolization

        normalize = np.arange(2*data.shape[1],dtype='float64')
        normalize = normalize.reshape(data.shape[1],2)
        for i in range(0,data.shape[1]):

            list = data[:,i]
            listlow,listhigh =  np.percentile(list, [0, 100])

            normalize[i,0] = listlow
            normalize[i,1] = listhigh

            delta = listhigh - listlow
            if delta != 0:
                for j in range(0,data.shape[0]):
                    data[j,i]  =  (data[j,i] - listlow)/delta

        return  data,normalize


    if mode==0:
        def trainModel(train_X, train_Y):#train model
            input=Input(shape=(train_X.shape[1], train_X.shape[2]))

            c_layer1 = Conv1D(32,10, activation='relu')(input)
            c_layer3 = MaxPooling1D(2)(c_layer1)
            c_layer4 = Conv1D(16, 10)(c_layer3)
            c_layer5 = MaxPooling1D(2)(c_layer4)
            l_layer3=Flatten()(c_layer5)

            output=Dense(train_Y.shape[1])(l_layer3)
            model=Model(inputs=[input],outputs=output)   

            model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
            model.fit(train_X, train_Y, epochs=500, batch_size=16, verbose=1,validation_data=(x_validate, y_validate))

            return model
    elif mode==1:
        def trainModel(train_X, train_Y):#train model
            input=Input(shape=(train_X.shape[1], train_X.shape[2]))

            l_layer1 = LSTM(50,return_sequences=True)(input)
            l_layer3=Flatten()(l_layer1)

            output=Dense(train_Y.shape[1])(l_layer3)
            model=Model(inputs=[input],outputs=output)   

            model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
            model.fit(train_X, train_Y, epochs=500, batch_size=16, verbose=1,validation_data=(x_validate, y_validate))

            return model
    elif mode ==2:
        def trainModel(train_X, train_Y):#train model
            input=Input(shape=(train_X.shape[1], train_X.shape[2]))

            c_layer1 = Conv1D(32,10, activation='relu')(input)
            c_layer3 = MaxPooling1D(2)(c_layer1)
            c_layer4 = Conv1D(16, 10)(c_layer3)
            c_layer5 = MaxPooling1D(2)(c_layer4)
            l_layer1 = LSTM(50,return_sequences=True)(c_layer5)
            l_layer3=Flatten()(l_layer1)
            output=Dense(train_Y.shape[1])(l_layer3)
            model=Model(inputs=[input],outputs=output)   

            model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
            model.fit(train_X, train_Y, epochs=5, batch_size=16, verbose=1,validation_data=(x_validate, y_validate))

            return model
    elif mode ==3:
        def trainModel(train_X, train_Y):#train model
            input=Input(shape=(train_X.shape[1], train_X.shape[2]))

            c_layer1 = Conv1D(32,10, activation='relu')(input)
            c_layer2 = MaxPooling1D(2)(c_layer1)
            c_layer3 = Conv1D(16, 10)(c_layer2)
            c_layer4 = MaxPooling1D(2)(c_layer3)
            c_layer5=Flatten()(c_layer4)

            l_layer1 = LSTM(50,return_sequences=True)(input)
            l_layer2=Flatten()(l_layer1)
            merge = keras.layers.concatenate([l_layer2,c_layer5])

            output=Dense(train_Y.shape[1],activation='sigmoid')(merge)

            model=Model(inputs=[input],outputs=output)       
            model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
            model.fit(train_X, train_Y, epochs=45, batch_size=128, verbose=1,validation_data=(x_validate, y_validate))
            return model
    def reshape_y_hat(y_hat,dim):#get predict shape from testing data
        re_y = np.zeros(len(y_hat),dtype='float64')
        length =int(len(y_hat)/dim)
        re_y = re_y.reshape(length,dim)

        for curdim in range(dim):
            for i in range(length):
                re_y[i,curdim] = y_hat[i + curdim*length]

        return  re_y
    def FNormalizeMult(data,normalize):#get original data from nomolized data

        data = np.array(data)
        for i in  range(0,data.shape[1]):
            listlow =  normalize[i,0]
            listhigh = normalize[i,1]
            delta = listhigh - listlow
            if delta != 0:
                for j in range(0,data.shape[0]):
                    data[j,i]  =  data[j,i]*delta + listlow

        return data
    x_train,x_validate,y_train,y_validate=train_test_split(train_X,train_Y,test_size=0.3)#get validated data and train data
    model = trainModel(train_X,train_Y)#train
    model.summary()#get parameter from model


    model.save("./mdt3beforenom.h5")
   
    predict_test=[]
    RMSE_sum=[]
    outwrite=[]
    for i in range(test_num):
        basis3=np.array([test_X[i]])
        test_dataY=np.array([test_Y[i]])
        y_pre  =  model.predict(basis3)#predict the test data
        y_pre = y_pre.reshape(y_pre.shape[1])
        p_test_Y = test_dataY.reshape(test_dataY.shape[1])
        p_test_Y = reshape_y_hat(p_test_Y,1)
        y_pre = reshape_y_hat(y_pre,1)

        y_pre = FNormalizeMult(y_pre, test_nom)#get original data from nomolized data
        p_test_Y = FNormalizeMult(p_test_Y, test_nom)
        
        predict_test.append(y_pre)


        diff = np.sum(np.abs(p_test_Y[:,0] - y_pre[:,0]))#get the MAE MAPE RMSE from predict data and test data
        MAE = diff/len(y_pre)
        MAPE = np.sum(np.abs((p_test_Y[:,0] - y_pre[:,0])/p_test_Y[:,0]))/len(y_pre)
        MSE = np.sum(np.square(y_pre[:,0] - p_test_Y[:,0]))/len(y_pre)
        RMSE =np.sqrt(MSE)
        #print('MAE'+str(i)+':',MAE,'MAPE'+str(i)+':',MAPE,'RMSE'+str(i)+':',RMSE)
        RMSE_sum.append(RMSE)
        plt.plot
        plt.plot(y_pre[:,0],linewidth=1,color='red',label='predicted_Y')
        plt.plot(p_test_Y[:,0],linewidth=1,color='blue',label='test_Y')

        plt.legend(loc='upper left')

        plt.savefig("./image_test/test"+str(i)+".png")#save the image result
        plt.close()
        outwrite.append('MAE'+str(i)+':'+' '+str(MAE)+'    '+'MAPE'+str(i)+':'+' '+str(MAPE)+'    '+'RMSE'+str(i)+':'+' '+str(RMSE))
    acc_test={}
    for i,j in enumerate(RMSE_sum):
        acc_test[i]=j
    with open("out_test.txt","w",encoding='utf-8') as f:#outwrite the result to the logfile
        for i in outwrite:
            f.writelines(i+'\n')
    predict_train=[]
    outwrite1=[]
    RMSE_sum1=[]
    for i in range(train_num):
        basis3=np.array([train_X[i]])
        test_dataY=np.array([train_Y[i]])
        y_pre  =  model.predict(basis3)#predict the test data
        y_pre = y_pre.reshape(y_pre.shape[1])
        p_test_Y = test_dataY.reshape(test_dataY.shape[1])
        p_test_Y = reshape_y_hat(p_test_Y,1)
        y_pre = reshape_y_hat(y_pre,1)

        y_pre = FNormalizeMult(y_pre, train_nom)#get original data from nomolized data
        p_test_Y = FNormalizeMult(p_test_Y, train_nom)
        predict_train.append(y_pre)

        diff = np.sum(np.abs(p_test_Y[:,0] - y_pre[:,0]))#get the MAE MAPE RMSE from predict data and test data
        MAE = diff/len(y_pre)
        MAPE = np.sum(np.abs((p_test_Y[:,0] - y_pre[:,0])/p_test_Y[:,0]))/len(y_pre)
        MSE = np.sum(np.square(y_pre[:,0] - p_test_Y[:,0]))/len(y_pre)
        RMSE =np.sqrt(MSE)
        #print('MAE'+str(i)+':',MAE,'MAPE'+str(i)+':',MAPE,'RMSE'+str(i)+':',RMSE)
        RMSE_sum1.append(RMSE)
        plt.plot
        plt.plot(y_pre[:,0],linewidth=1,color='red',label='predicted_Y')
        plt.plot(p_test_Y[:,0],linewidth=1,color='blue',label='train_Y')

        plt.legend(loc='upper left')

        plt.savefig("./image_train/test"+str(i)+".png")#save the image result
        plt.close()
        outwrite1.append('MAE'+str(i)+':'+' '+str(MAE)+'    '+'MAPE'+str(i)+':'+' '+str(MAPE)+'    '+'RMSE'+str(i)+':'+' '+str(RMSE))
    acc_train={}
    for i,j in enumerate(RMSE_sum1):
        acc_train[i]=j
            
    with open("out_train.txt","w",encoding='utf-8') as f:#outwrite the result to the logfile
        for i in outwrite1:
            f.writelines(i+'\n')
    return acc_test,acc_train,predict_test,predict_train

