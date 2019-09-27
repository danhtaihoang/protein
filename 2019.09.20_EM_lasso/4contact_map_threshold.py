import sys
import numpy as np

seed = 1
np.random.seed(seed)

#pfam_id = 'PF00008'
#ipdb = 0
pfam_id = sys.argv[1]
ipdb = sys.argv[2]
ipdb = int(ipdb)

ext_name = '%s/%02d'%(pfam_id,ipdb)

try:
    ct = np.loadtxt('%s_ct.dat'%ext_name)
except:
    pass    
#=========================================================================================
thresholds = [2.,4.,6.,8.,10]

for threshold in thresholds:
    ct1 = ct.copy()
    np.fill_diagonal(ct1, 1000)

    # fill the top smallest to be 1, other 0
    top_pos = ct1 <= threshold
    #print(top_pos)
    ct1[top_pos] = 1.
    ct1[~top_pos] = 0.
    #print(ct1) 
    
    xy = np.argwhere(ct1==1)
    #print(xy)
    
    np.savetxt('%s_contact_%02d.dat'%(ext_name,threshold),xy,fmt='% i')
