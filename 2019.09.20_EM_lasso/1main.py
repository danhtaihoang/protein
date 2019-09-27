# 2019.01.28: Calculate only one half of W
# P(A) = P(A1|A2A3... An)*P(A2|A3... An)*P(A3|A4... An)*.....

import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import inference
from data_processing import data_processing
#from protein_validate import contact_map,direct_info
from protein_validate import direct_info

import sys, os, Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

data_path = '../../Pfam-A.full'

#=========================================================================================
np.random.seed(1)

l1 = 0.000 # regularization
nloop = 5

pfam_id = 'PF00186'
ipdb = 1

#pfam_id = sys.argv[1]
#print(pfam_id)
#ipdb = sys.argv[2]
ipdb = int(ipdb)
#print(ipdb)
      
ext_name = '%s/%02d'%(pfam_id,ipdb)

#--------------------------------------------
# read data
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
npdb = pdb.shape[0]

# data processing
s0,cols_removed = data_processing(data_path,pfam_id,ipdb,\
                        gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.8)
#np.savetxt('cols_removed.dat',cols_removed,fmt='%i')
    
#--------------------------------------------------------------------------
def contact_map(pdb,ipdb,cols_removed):
    pdb_id = pdb[ipdb,5]
    pdb_chain = pdb[ipdb,6]
    pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
    #print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)

    #print('download pdb file')
    pdb_file = pdb_list.retrieve_pdb_file(pdb_id,file_format='pdb')
    #pdb_file = pdb_list.retrieve_pdb_file(pdb_id)
    chain = pdb_parser.get_structure(pdb_id,pdb_file)[0][pdb_chain]
    coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
    coords = coords_all[pdb_start-1:pdb_end]
    #print('original pdb:')
    #print(coords.shape)

    coords_remain = np.delete(coords,cols_removed,axis=0)
    #print(coords_remain.shape)

    ct = distance_matrix(coords_remain,coords_remain)

    return ct

#--------------------------------------------------------------------------
# contact map
ct = contact_map(pdb,ipdb,cols_removed)    

#if s0.shape[1] != ct.shape[0]:
print('s0 and ct size:',ipdb,s0.shape[1],ct.shape[0])

# convert to onehot
onehot_encoder = OneHotEncoder(sparse=False)
s = onehot_encoder.fit_transform(s0) 

n = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
              
#-----------------------------------------------------------------------------------------
# inferring h0 and w
mx_sum = mx.sum()
my_sum = mx.sum() #!!!! my_sum = mx_sum

w = np.zeros((mx_sum,my_sum))
h0 = np.zeros(my_sum)
cost = np.zeros((n,nloop))

niter = np.zeros(n)

#for i0 in range(n):
for i0 in range(n-1):
    print('i0:',i0)       
    i1,i2 = i1i2[i0,0],i1i2[i0,1]
    #x = np.hstack([s[:,:i1],s[:,i2:]])
    # w one half:
    x = s[:,i2:]
    y = s[:,i1:i2]

    #w,h0,cost = inference.fit(x,y,nloop=10)
    w1,h01,cost1,niter1 = inference.fit_additive(x,y,l1,nloop)

    #w[:i1,i1:i2] = w1[:i1,:]

    # w one half:
    w[i2:,i1:i2] = w1[:,:]
    h0[i1:i2] = h01
    cost[i0,:] = cost1
    niter[i0] = niter1

np.savetxt('%s_ct.dat'%ext_name,ct,fmt='% f')
np.savetxt('%s_w.dat'%ext_name,w,fmt='% 3.8f')
np.savetxt('%s_h0.dat'%ext_name,h0,fmt='% 3.8f')
np.savetxt('%s_cost.dat'%ext_name,cost,fmt='% 3.8f')
np.savetxt('%s_niter.dat'%ext_name,niter,fmt='% i')

# 2019.01.28: fill another half of w
for i in range(w.shape[0]):
    for j in range(0,i):
        w[j,i] = w[i,j]

# direct information
di = direct_info(s0,w,h0)
np.savetxt('%s_di.dat'%ext_name,di,fmt='% 3.8f')

print('finished')
