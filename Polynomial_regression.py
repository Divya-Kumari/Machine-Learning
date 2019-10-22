# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 10:39:42 2019

@author: kumar
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pt
from sklearn.preprocessing import PolynomialFeatures as pf,StandardScaler
from sklearn.metrics import mean_squared_error
data=pd.read_csv("C:\\Users\\kumar\\Desktop\\train.csv")
tdata=pd.read_csv("C:\\Users\\kumar\\Desktop\\test.csv")
ivalue=data.values
sc=StandardScaler()
transformed=ivalue
X_origin=transformed[:,0:4]
Y_origin=transformed[:,4]
p=pf(2)
X=p.fit_transform(X_origin)
def GD(B,X,Y,alfa):
    m=len(Y)
    z=alfa/m
    B=B-(z*(((B.T.dot(X))-Y).dot(X.T)).T)
    return(B)
def Polynomial_regression(X,Y):
    a=X.shape
    B=np.matrix([np.zeros(a[0])])
    alfa=0.00000000000001
    NOI=1000
    Thresh_error=0.001
    J=[]
    for i in range(NOI):
        yp=B.dot(X)
        er=yp-Y
        J.append((1/(2*len(Y))*np.sum(np.square(er))))
        if(J[i]<Thresh_error):
            break
        elif(len(J)>10 and np.mean(J[-10:])==J[-1]):
            break
        else:
            B=GD(B.T,X,Y,alfa).T
    return(J,B)
def testing(X,Y,B):
    yp=B.dot(X)
    e=np.sqrt((1/len(Y))*np.sum(np.square(yp-Y)))
    return(e)
def plot_error(J):
    a=[]
    for i in range(len(J)):
        a.append(i)
    pt.plot(a,J)
def predict_y(X,B):
    y=B.dot(X)
    return(y)    

    
