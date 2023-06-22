# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:30:18 2023

@author: grover.laporte
"""

import numpy as np
import pandas as pd

import streamlit as st
st.write("""
          # Curve fitting explanation
          Experiment to show what happens to the curve when things change. 
          Change the time using the slider bar and change the percent above
          and below the approximation by changing the percent using the + and -
          """)
    
data = ['example',93.5,89,91,93,120,110,109,97,81,76,94,110,102,97,99,97,
        102,81,81,78,96,97,109,92]

data = pd.DataFrame(data).T
data.columns = ['name',0,5,10,15,20,25,30,40,50,60,70,80,90,100,
                110,120,130,140,150,160,170,180,210,240]

def cubic_spline(x,a):
    n = len(x)-1
    x = np.array(x)
    a = np.array(a)
    h = np.zeros(n)
    for i in range(n):
        h[i]=x[i+1]-x[i]
    alpha = np.zeros(n+1)
    for i in range(1,n):
        alpha[i] = 3/h[i]*(a[i+1]-a[i])-3/h[i-1]*(a[i]-a[i-1])
        
    l = np.zeros(n+1)
    mu = np.zeros(n+1)
    z = np.zeros(n+1)
    b = np.zeros(n+1)
    c = np.zeros(n+1)
    d = np.zeros(n+1)
    l[0] = 1
    
    for i in range(1,n):
        l[i] = 2*(x[i+1]-x[i-1])-h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
    
    l[n-1] = 1
    
    for j in range(n-1,-1,-1):
        c[j] = z[j]-mu[j]*c[j+1]
        b[j] = (a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    return(a[:-1],b[:-1],c[:-1],d[:-1])

def S(x,X,a,b,c,d):
    if x<X[0] or x>X[len(X)-1]:
        print("Not a good approximation for this value")
        return(None)
    a = np.array(a);b = np.array(b);c = np.array(c);d = np.array(d)
    X = np.array(X)
    idx = np.where(X[x<=X][0]==X)
    if idx[0][0] != 0:
        idx[0][0] -= 1
    ans = a[idx]+b[idx]*(x-X[idx])+c[idx]*(x-X[idx])**2+d[idx]*(x-X[idx])**3
    return(ans)

class Curve(object):
    def __init__(self,row,col):
        self.name = row.iloc[0]
        max_time = int(list(row.columns)[len(row.columns)-1])
        self.labels = list(row.columns[col:])
        self.num_labels = np.array([int(c) for c in self.labels])
        self.data = row.iloc[0,col:]
        self.num_data = self.data.values
        self.spline_coeff()
        self.deltat = 0.1
        self.t = np.arange(0,max_time,self.deltat)
        self.f = self.splines(self.t)
        self.df = pd.DataFrame(self.f).T
        self.df.columns = self.t
    def __call__(self,X):
        return(self.splines(X))
    def spline_coeff(self):
        a,b,c,d = cubic_spline(self.num_labels, self.num_data)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    def splines (self,X):
        try:
            res = []
            for x in X:
                res.append(S(x,self.num_labels,self.a,self.b,self.c,self.d))
            return np.array(res)
        except:
            return float(S(X,self.num_labels,self.a,self.b,self.c,self.d))

### Case 1 - Missing t = 100 ####################
c1 = Curve(data,1)


def find_approximate_data(data,t):
    t_vals = np.array(data.columns[1:])
    idx = (int(np.where(t_vals==t)[0]))+1
    x_im1=(data.iloc[0,idx-1])
    x_ip1=(data.iloc[0,idx+1])
    return (x_im1+x_ip1)/2
    
idx_time = st.slider("Choose a time:",min_value = 2,
                     max_value = len(data.columns)-2,value=14) 
st.write("### Current Time",data.columns[idx_time])
perc = st.number_input("Percent +/- (1-25)",min_value=1,max_value=25,
                          step = 2,value=10)
st.write("### Current Percent:",perc)
percent = perc/100
time = data.columns[idx_time]

approx=find_approximate_data(data,time)
show_values = np.array([approx-percent*approx,approx-percent/2*approx,
                        approx+percent/2,approx+percent*approx])
chart = c1.df.copy()
for val in show_values:
    new_data = data.copy()
    new_data[time]=val
    c2 = Curve(new_data,1)
    chart = pd.concat([chart,c2.df])
chart.index = ["approximation",str(percent*100)+"% lower", str(percent*50)+"% lower",
               str(percent*50)+"% higher",
               str(percent*100)+"% higher"]
chart = chart.T
st.line_chart(chart)

