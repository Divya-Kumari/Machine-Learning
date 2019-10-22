# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:33:26 2019

@author: kumar
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pt
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as sklearn
xyz=pd.read_csv("C:\\Users\\kumar\\Desktop\\train.csv")
sc=sklearn.StandardScaler()
sc.fit(xyz)
data=(sc.transform(xyz)).T
tdata=(pd.read_csv("C:\\Users\\kumar\\Desktop\\test.csv"))
sc.fit(tdata)
r=(sc.transform(tdata))
a=data.shape
XT=np.array([np.ones(a[0]),r[:,0],r[:,1],r[:,2],r[:,3]])
Y=np.array(data[4])
X=np.array([np.ones(a[1]),data[0],data[1],data[2],data[3]])
def GD(B,X,Y,alfa,lamda):
    m=len(Y)
    z=alfa/m
    B=B-(z*((((B.T.dot(X))-Y).dot(X.T)).T)+lamda*B)
    return(B)
def Regularized_regression(X,Y):
    B=np.matrix([0.1,0.1,0.2,0.1,0.2])
    alfa=0.0000000001
    lamda=0.001
    NOI=1000
    Thresh_error=0.001
    J=[]
    for i in range(NOI):
        yp=B.dot(X)
        er=yp-Y
        J.append((1/(2*len(Y))*((np.sum(np.square(er)))+lamda*np.sum(np.square(B)))))
        if(J[i]<Thresh_error):
            break
        elif(len(J)>10 and np.mean(J[-10:])==J[-1]):
            break
        else:
            B=GD(B.T,X,Y,alfa,lamda).T
    return(J,B)
def predict_y(X,B):
    y=X.dot(B)
    return(y)    
def testing(X,Y,B):
    yp=B.dot(X)
    rmse=np.sqrt(mean_squared_error(Y,yp))
    return(rmse)
def plot_error(J):
    a=[]
    for i in range(len(J)):
        a.append(i)
    pt.plot(a,J)

def plot_contour(j,x,y):
    pt.contour(j,x,y,colour='black')  