#!/usr/bin/env python
# coding: utf-8


import os
import linecache
root = input('please input the dir:')#get txt file
file_names = os.listdir(root)

file_ob_list = []
for file_name in file_names:
    fileob = root + '/' + file_name
    file_ob_list.append(fileob)
    
import random
te = [] 
tr = []
te = random.sample(file_ob_list, int(0.3*(len(file_ob_list)))) #Randomly assign 30 percent of the test set and 70 percent of the training set
for testData in te: 
    file_ob_list.remove(testData)
tr = file_ob_list 

from function import RoundSampleSplit
train_data,train_data_num=RoundSampleSplit(tr,1024,10,1)
#print(train_data_num)
test_data,test_data_num=RoundSampleSplit(te,1024,1,0)

from function import NormalizeMult
train_data,train_nom=NormalizeMult(train_data)
test_data,test_nom=NormalizeMult(test_data)

from function import BasisFunIdent
import numpy as np
HB_all=[]
HB=[]
for i in train_data:#get H and B from train set
    x=np.arange(0,2*np.pi,2*np.pi/1024)
    y = np.array(i)
    Hb,HBbasis=BasisFunIdent(x,y,2*np.pi/1024,3,3)
    HB.append(HBbasis)
    HB_all.append(Hb)
#print(HB_all)
#print(HB)

HBt=[]
for i in test_data:#get H and B from test set
    x=np.arange(0,2*np.pi,2*np.pi/1024)
    y = np.array(i)
    _,HBbasis=BasisFunIdent(x,y,2*np.pi/1024,3,3)
    HBt.append(HBbasis)

basis=[]
for i in HB:#get basic function from training set
    if len(i)==3:#The case of the three basis functions            
        x1=np.arange(0,2*i[0][1],2*np.pi/1024)
        Y1=i[0][0]*np.sin(np.pi/i[0][1]*x1)
        x2=np.arange(0,2*i[1][1],2*np.pi/1024)
        Y2=i[1][0]*np.sin(np.pi/i[1][1]*x2)
        x3=np.arange(0,2*i[2][1],2*np.pi/1024)
        Y3=i[2][0]*np.sin(np.pi/i[2][1]*x3)        
        x1=list(x1)
        x2=list(x2)
        x3=list(x3)
        Y1=list(Y1)
        Y2=list(Y2)
        Y3=list(Y3)                
        x_tr=[]
        for j in range(len(Y1)+len(Y2)+len(Y3)):
            if j<=(len(Y1)-1):
                x_tr.append(Y1[j])
            elif j>(len(Y1)-1) and j<=(len(Y1)+len(Y2)-1):
                x_tr.append(Y2[j-len(Y1)])
            elif j>(len(Y1)+len(Y2)) and j<=(len(Y1)+len(Y2)+len(Y3)-1):
                x_tr.append(Y3[j-len(Y1)-len(Y2)])               
        if len(x_tr)<=1024:
            x_tr=x_tr+(1024-len(x_tr))*x_tr[0:1]
        else:
            x_tr=x_tr[0:1024]       
        basis.append(x_tr) 
               
    elif len(i)==2:#The case of the two basis functions
        x1=np.arange(0,2*i[0][1],2*np.pi/1024)
        Y1=i[0][0]*np.sin(np.pi/i[0][1]*x1)
        x2=np.arange(0,2*i[1][1],2*np.pi/1024)
        Y2=i[1][0]*np.sin(np.pi/i[1][1]*x2)                
        x1=list(x1)
        x2=list(x2)      
        Y1=list(Y1)
        Y2=list(Y2)               
        x_tr=[]
        for j in range(len(Y1)+len(Y2)):
            if j<=(len(Y1)-1):
                x_tr.append(Y1[j])
            elif j>(len(Y1)-1) and j<=(len(Y1)+len(Y2)-1):
                x_tr.append(Y2[j-len(Y1)])            
        if len(x_tr)<=1024:
            x_tr=x_tr+(1024-len(x_tr))*x_tr[0:1]
        else:
            x_tr=x_tr[0:1024]
        basis.append(x_tr)
                
    elif len(i)==1:#The case of the one basis function
        x_tr=[]
        x1=np.arange(0,2*i[0][1],2*np.pi/1024)#[:,np.newaxis]
        Y1=i[0][0]*np.sin(np.pi/i[0][1]*x1)
        for i in range(len(Y1)):
            x_tr.append(Y1[i])
        if len(x_tr)<=1024:
            x_tr=x_tr+(1024-len(x_tr))*x_tr[0:1]
        else:
            x_tr=x_tr[0:1024]        
        basis.append(x_tr)        
basis=np.array(basis)
basis1=[]
for i in basis:
    x111=[]
    for j in i:        
        x111.append([j])
    basis1.append(x111)
basis1=np.array(basis1)

basist=[]
for k in HBt:#get basic function of testing set
    if len(k)==3:#The case of the three basis functions           
        x11=np.arange(0,2*k[0][1],2*np.pi/1024)
        Y11=k[0][0]*np.sin(np.pi/k[0][1]*x11)
        x22=np.arange(0,2*k[1][1],2*np.pi/1024)
        Y22=k[1][0]*np.sin(np.pi/k[1][1]*x22)
        x33=np.arange(0,2*k[2][1],2*np.pi/1024)
        Y33=k[2][0]*np.sin(np.pi/k[2][1]*x33)       
        x11=list(x11)
        x22=list(x22)
        x33=list(x33)
        Y11=list(Y11)
        Y22=list(Y22)
        Y33=list(Y33)                
        x_te=[]
        for l in range(len(Y11)+len(Y22)+len(Y33)):
            if l<=(len(Y11)-1):
                x_te.append(Y11[l])
            elif l>(len(Y11)-1) and l<=(len(Y11)+len(Y22)-1):
                x_te.append(Y22[l-len(Y11)])
            elif l>(len(Y11)+len(Y22)) and l<=(len(Y11)+len(Y22)+len(Y33)-1):
                x_te.append(Y33[l-len(Y11)-len(Y22)])
        if len(x_te)<=1024:
            x_te=x_te+(1024-len(x_te))*x_te[0:1]
        else:
            x_te=x_te[0:1024]
        basist.append(x_te)
              
    elif len(k)==2:#The case of the two basis functions
        x11=np.arange(0,2*k[0][1],2*np.pi/1024)
        Y11=k[0][0]*np.sin(np.pi/k[0][1]*x11)
        x22=np.arange(0,2*k[1][1],2*np.pi/1024)
        Y22=k[1][0]*np.sin(np.pi/k[1][1]*x22)                
        x11=list(x11)
        x22=list(x22)       
        Y11=list(Y11)
        Y22=list(Y22)               
        x_te=[]
        for l in range(len(Y11)+len(Y22)):
            if l<=(len(Y11)-1):
                x_te.append(Y11[l])
            elif l>(len(Y11)-1) and l<=(len(Y11)+len(Y22)-1):
                x_te.append(Y22[l-len(Y11)])            
        if len(x_te)<=1024:
            x_te=x_te+(1024-len(x_te))*x_te[0:1]
        else:
            x_te=x_te[0:1024]
        basist.append(x_te)
        
    elif len(k)==1:#The case of the one basis function
        x11=np.arange(0,2*k[0][1],2*np.pi/1024)
        Y11=k[0][0]*np.sin(np.pi/k[0][1]*x11)
        x11=list(x11)
        Y11=list(Y11)
        x_te=[]
        for l in range(len(Y11)):
            x_te.append(Y11[l])
        if len(x_te)<=1024:
            x_te=x_te+(1024-len(x_te))*x_te[0:1]
        else:
            x_te=x_te[0:1024]
        basist.append(x_te)
basist=np.array(basist)
basis2=[]
for i in basist:
    x112=[]
    for j in i:
        x112.append([j])
        
    basis2.append(x112)
basis2=np.array(basis2)

from function import RoundPre
acc_test,acc_train,pre_test,pre_train=RoundPre(basis1,train_data,basis2,test_data,2,48,48,train_nom,test_nom)
#print(acc_test)
#print(acc_train)
#print(pre_test)
#print(pre_train)







