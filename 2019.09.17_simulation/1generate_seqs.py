import numpy as np
import matplotlib.pyplot as plt
import emachine as EM

from sklearn.preprocessing import OneHotEncoder

np.random.seed(0)
#===========================================================================
# parameters:
n_var = 10; m = 4; g = 2. ; sp = 0.0

#mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx = np.array([m for i in range(n_var)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
mx_sum = mx.sum()
n_linear = n_var*(m-1)

w_true = EM.generate_interactions(n_var,m,g,sp,i1i2)

#plt.imshow(w_true,cmap='rainbow',origin='lower')

# generate data:
n_seq = 10000
w_true_ops = EM.convert_w_true_ops(w_true,n_var,m,i1i2)

seqs = EM.generate_seq(w_true_ops,n_seq,n_var,m,i1i2)

np.savetxt('seqs.txt',seqs,fmt='%i')
np.savetxt('w_true.txt',w_true,fmt='%f')