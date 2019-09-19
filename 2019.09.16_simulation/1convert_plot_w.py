#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import emachine as EM
#=======================================================================

# 1d
w_true_ops = np.loadtxt('w_true_ops.dat')
w = np.loadtxt('w.dat')

print('w_ops(1d):')
plt.plot([-2,2],[-2,2])
plt.scatter(w_true_ops,w)
plt.show()

# 2d
w_true = np.loadtxt('w_true.dat')

n_var = 10; m = 4
#mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx = np.array([m for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
mx_sum = mx.sum()
n_linear = n_var*(m-1)

# convert w (1d as ops index) to 2d
ij2d = EM.ij_2d_from_1d(n_var,m,i1i2)
n_ops = len(w)

w2d = np.zeros((mx_sum,mx_sum))
for iops in range(n_linear,n_ops):
    w2d[int(ij2d[iops,0]),int(ij2d[iops,1])] = w[int(iops)]

plt.plot([-2,2],[-2,2])    
plt.scatter(w_true,w2d)
plt.show()

#==========================================================
print('correct the last aa:')

# last aa
for i in range(n_var-1):
    i1,i2 = i1i2[i,0],i1i2[i,1]
    for j in range(i+1,n_var):
        j1,j2 = i1i2[j,0],i1i2[j,1]

        w2d[i1:i2,j2-1] = -np.sum(w2d[i1:i2,j1:j2-1],axis=1)
        w2d[i2-1,j1:j2] = -np.sum(w2d[i1:i2-1,j1:j2],axis=0)

print('true, zero elements:',(w_true==0).sum())
print('predict, zero elements:',(w2d==0).sum())

plt.plot([-2,2],[-2,2])    
plt.scatter(w_true,w2d)
plt.show()