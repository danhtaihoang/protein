import numpy as np
from sklearn.preprocessing import OneHotEncoder

def generate_interactions(n,m,g,sp,i1i2):
    """gerate interaction:
    n: number of variable
    m: number of categories of each variable
    g: coupling interaction variance
    sp: sparsity degree
    """
    nm = n*m
    w = np.random.normal(0.0,g/np.sqrt(nm),size=(nm,nm))
    
    """
    # sparse
    for i in range(n):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(n):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            if (np.abs(j-i) > 1) and (np.random.rand() < sp): 
                w[i1:i2,j1:j2] = 0.
    """

    # sum_j wji to each position i = 0                
    #for i in range(n):        
    #    i1,i2 = i1i2[i,0],i1i2[i,1]             
    #    w[:,i1:i2] -= w[:,i1:i2].mean(axis=1)[:,np.newaxis]            

    # no self-interactions
    #for i in range(n):
    #    i1,i2 = i1i2[i,0],i1i2[i,1]
    #    w[i1:i2,i1:i2] = 0.   # no self-interactions

    # symmetry
    #for i in range(nm):
    #    for j in range(i+1,nm):
    #        if j > i: w[i,j] = w[j,i]

    # set lower to be zeros
    for i in range(n):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(n):
            j1,j2 = i1i2[j,0],i1i2[j,1]

            if (i >= j): 
                w[i1:i2,j1:j2] = 0.                
            else:
                w[i1:i2,j2-1] = -np.sum(w[i1:i2,j1:j2-1],axis=1)
                w[i2-1,j1:j2] = -np.sum(w[i1:i2-1,j1:j2],axis=0)         
                        
    return w
#=========================================================================================
def convert_w_true_ops(w_true,n_var,m,i1i2):
    """
    convert w_true[n_quad,n_quand] from 2d to 1d as operators (ops) index, w_true_ops[n_ops]
    just for comparision
    """
    #input: s[n_seq, n_var*m]: one hot
    n_linear = int((m-1)*n_var)
    n_quad = int(((m-1)**2)*n_var*(n_var-1)/2.)
    n_ops = n_linear + n_quad
        
    w_true_ops = np.zeros(n_ops)    
    iops = n_linear   
    # quadratic terms
    for i in range(n_var-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n_var):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            for ia in range(i1,i2-1):
                for jb in range(j1,j2-1):
                    w_true_ops[iops] = w_true[ia,jb]
                                        
                    iops += 1
                        
    return w_true_ops

#=========================================================================================
def operators_for_simulation(s,n_var,i1i2):
    """
    input: s[n_seq, n_var*m]: one hot   
    output: ops[n_seq,n_ops] : onehot, indepent variables
    ij_tab: convert ops index to 2d indices (ia,jb)
    """   
    n_seq,nm = s.shape
    m = int(nm/float(n_var)) # numer of categories at each position

    n_linear = int((m-1)*n_var)
    n_quad = int(((m-1)**2)*n_var*(n_var-1)/2.)
    n_ops = n_linear + n_quad
        
    ops = np.zeros((n_seq,n_ops))    
    iops = 0
    # linear terms    
    for i in range(n_var):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for ia in range(i1,i2-1):
            ops[:,iops] = (s[:,ia] - s[:,i2-1])                   
            iops += 1
    
    # quadratic terms
    for i in range(n_var-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n_var):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            for ia in range(i1,i2-1):
                for jb in range(j1,j2-1):
                    ops[:,iops] = (s[:,ia] - s[:,i2-1])*(s[:,jb] - s[:,j2-1])
                                        
                    iops += 1
                        
    return ops

#=========================================================================================
def ij_2d_from_1d(n_var,m,i1i2):
    n_linear = int((m-1)*n_var)
    n_quad = int(((m-1)**2)*n_var*(n_var-1)/2.)
    n_ops = n_linear + n_quad
        
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
def operators(s,n_var,i1i2):
    """
    input: s[n_seq, n_var*m]: one hot   
    output: ops[n_seq,n_ops] : onehot, indepent variables
    ij_2d: convert ops index to 2d indices (ia,jb)
    """   
    n_seq,nm = s.shape
    m = int(nm/float(n_var)) # numer of categories at each position

    n_linear = int((m-1)*n_var)
    n_quad = int(((m-1)**2)*n_var*(n_var-1)/2.)
    n_ops = n_linear + n_quad
        
    ops = np.zeros((n_seq,n_ops))
    
    iops = 0
    # linear terms    
    for i in range(n_var):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for ia in range(i1,i2-1):
            ops[:,iops] = (s[:,ia] - s[:,i2-1])                   
            iops += 1
    
    # quadratic terms
    for i in range(n_var-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n_var):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            for ia in range(i1,i2-1):
                for jb in range(j1,j2-1):
                    ops[:,iops] = (s[:,ia] - s[:,i2-1])*(s[:,jb] - s[:,j2-1])                                        
                    iops += 1
                        
    return ops
#=========================================================================================
def generate_seq(w_true_ops,n_seq,n_var,m,i1i2,n_sample=30):
    """

    """
    samples = np.random.choice(np.arange(m),size=(n_seq*n_sample,n_var),replace=True)

    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    s = onehot_encoder.fit_transform(samples)

    ops = operators_for_simulation(s,n_var,i1i2)
    
    energy = ops.dot(w_true_ops)# linear and quadratic

    p = np.exp(energy)
    p /= np.sum(p)

    out_samples = np.random.choice(np.arange(n_seq*n_sample),size=n_seq,replace=True,p=p)

    return s[out_samples]
#=========================================================================================
def fit(ops,n_var,m,eps=0.1,max_iter=100,alpha=0.1):
    """
    input: ops[n_seq,n_ops], m: number of categories at each position
    """    
    n_linear = int((m-1)*n_var)
    n_quad = int(((m-1)**2)*n_var*(n_var-1)/2.)
    
    cov = np.hstack([np.full(n_linear,2./m),np.full(n_quad,4./(m**2))])
    
    E_av = np.zeros(max_iter)
    n_ops = ops.shape[1]

    np.random.seed(13)
    w = np.random.rand(n_ops)-0.5    
    for i in range(max_iter):              
        #energies_w = energy_ops(ops,w)
        energies_w = ops.dot(w)        
        energies_max = energies_w.max()
        
        # to avoid a too lager value:
        probs_w = np.exp((energies_w-energies_max)*(eps-1)) 
        z_data = np.sum(probs_w)
        probs_w /= z_data
        ops_expect_w = np.sum(probs_w[:,np.newaxis]*ops,axis=0)

        E_av[i] = energies_w.mean()  
        w += alpha*(ops_expect_w - eps*w*cov)
      
    return w,-E_av[-1]


