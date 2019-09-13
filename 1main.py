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

w_true = EM.generate_interactions(n_var,m,g,sp,i1i2)

#plt.imshow(w_true,cmap='rainbow',origin='lower')

# generate data:
n_seq = 10000
w_true_ops = EM.convert_w_true_ops(w_true,n_var,m,i1i2)

seqs = EM.generate_seq(w_true_ops,n_seq,n_var,m,i1i2)

ops = EM.operators(seqs,n_var,i1i2)

ij2d = EM.ij_2d_from_1d(n_var,m,i1i2)

#===========================================================================
# infer w:
n_ops = ops.shape[1]
eps_list = np.linspace(0.1,0.4,4)
E_eps = np.zeros(len(eps_list))
w_eps = np.zeros((len(eps_list),n_ops))
for i,eps in enumerate(eps_list):
    w_eps[i,:],E_eps[i] = EM.fit(ops,n_var,m,eps=eps,max_iter=100)
    print(eps,E_eps[i])

ieps = np.argmax(E_eps)
w = w_eps[ieps]
print('optimal eps:', eps_list[ieps])
#plt.plot(eps_list,E_eps)

np.savetxt('w_true.dat',w_true,fmt='%f')
np.savetxt('w_true_ops.dat',w_true_ops,fmt='%f')
np.savetxt('w.dat',w,fmt='%f')