# 2019.11.06: apply EM (binary) to predicting of protein structure
#
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from scipy.sparse import csr_matrix

#=========================================================================================
def ij_2d_from_1d(n_var,i1i2,mx):
    mx_sum = mx.sum()
    n_linear = mx_sum - n_var
    n_quad = int((mx_sum**2 - np.sum(mx**2))/2.)    
    n_ops = n_linear + n_quad

    #n_linear = int((m-1)*n_var)
    #n_quad = int(((m-1)**2)*n_var*(n_var-1)/2.)
    #n_ops = n_linear + n_quad
        
    ij_2d = np.zeros((n_ops,2))    
    iops = n_linear
    for i in range(n_var-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n_var):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            for ia in range(i1,i2-1):
                for jb in range(j1,j2-1):                    
                    ij_2d[iops,0] = ia
                    ij_2d[iops,1] = jb
                                        
                    iops += 1
                        
    return ij_2d.astype(int)

#=========================================================================================
# 2019.11.06: s1 = +/-1
def operators(s,n_var,i1i2,mx):
    """
    input: s[n_seq, n_var*m]: one hot   
    output: ops[n_seq,n_ops] : onehot, indepent variables
    ij_2d: convert ops index to 2d indices (ia,jb)
    """   
    n_seq,nm = s.shape

    mx_sum = mx.sum()
    n_linear = mx_sum - n_var
    n_quad = int((mx_sum**2 - np.sum(mx**2))/2.)
    
    n_ops = n_linear + n_quad       
    ops = np.zeros((n_seq,n_ops),dtype=np.int8)
    s1 = 2*s - 1  # convert 1,0 to 1,-1

    iops = 0
    # linear terms    
    for i in range(n_var):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for ia in range(i1,i2-1):
            ops[:,iops] = s1[:,ia]
            iops += 1

    # quadratic terms
    for i in range(n_var-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n_var):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            for ia in range(i1,i2-1):
                for jb in range(j1,j2-1):
                    ops[:,iops] = s1[:,ia]*s1[:,jb]                                    
                    iops += 1
                        
    return ops #,cov

#=========================================================================================
# find coupling w from sequences s
# input: ops[n_seq,n_ops], l1: Lasso
# output: w[n_ops], E_av
def fit(ops,l1,eps=0.1,max_iter=100,alpha=0.1):
    eps1 = eps - 1

    E_av = np.zeros(max_iter)
    n_ops = ops.shape[1]

    np.random.seed(13)
    w = np.random.rand(n_ops)-0.5    
    for i in range(max_iter):
        energy = ops.dot(w)

        energy_max = energy.max()  
        prob = np.exp((energy-energy_max)*eps1) # to avoid a too lager value

        #prob = np.exp(energy*eps1)
        z_data = prob.sum()
        prob /= z_data
        ops_ex = np.sum(prob[:,np.newaxis]*ops,axis=0)
        
        w += alpha*(ops_ex - w*eps - l1*np.sign(w))

        E_av[i] = energy.mean()    
      
    return w,-E_av
