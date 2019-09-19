#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:12:42 2019

@author: tai
"""

import numpy as np
import matplotlib.pyplot as plt
import emachine as EM

n_var = 10; m = 2; sp = 0.; g = 2. 

mx = np.array([m for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T

w = EM.generate_interactions(n_var,m,g,sp,i1i2)

plt.imshow(w,cmap='rainbow',origin='lower')
plt.xlabel('i')
plt.xlabel('j')

print('true, zero elements:',(w == 0).sum())

print(w[0,:])

"""
i = 0
j = 1
# for i in range(n):
i1,i2 = i1i2[i,0],i1i2[i,1]
# for j in range(n):
j1,j2 = i1i2[j,0],i1i2[j,1]

print(j1,j2,np.sum(w[0,j1:i2-1],axis=1))
"""
        #if (i < j): 
            #w[i1:i2,j2-1] = -np.sum(w[i1:i2,j1:i2-1],axis=1)
            #w[i2-1,j1:j2] = -np.sum(w[i1:i2-1,j1:j2],axis=0) 