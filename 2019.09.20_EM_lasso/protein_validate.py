import numpy as np
import sys, os, Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#===================================================================================================

#pfam_id = 'PF00186'
#print('read original aligned pfam data')
#s = np.load('../%s/msa.npy'%pfam_id).T
#print(s.shape)

# read parse_pfam data:
#print('read pdb references data')
#pdb = np.load('../%s/pdb_refs.npy'%pfam_id)
#print(pdb.shape)
#print(pdb[0])

#ipdb = 0
#===================================================================================================
def contact_map(pdb,ipdb,cols_removed):
    pdb_id = pdb[ipdb,5]
    pdb_chain = pdb[ipdb,6]
    pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])
    #print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)

    #print('download pdb file')
    #pdb_file = pdb_list.retrieve_pdb_file(pdb_id,file_format='pdb')
    pdb_file = pdb_list.retrieve_pdb_file(pdb_id)
    chain = pdb_parser.get_structure(pdb_id,pdb_file)[0][pdb_chain]
    coords_all = np.array([a.get_coord() for a in chain.get_atoms()])
    coords = coords_all[pdb_start-1:pdb_end]
    #print('original pdb:')
    #print(coords.shape)

    coords_remain = np.delete(coords,cols_removed,axis=0)
    #print(coords_remain.shape)

    ct = distance_matrix(coords_remain,coords_remain)
    #np.savetxt('contact_map.dat',ct,fmt='% f')

    #plt.figure(figsize=(3.2,3.2))
    #plt.title('contact map')
    #plt.imshow(ct,cmap='rainbow',origin='lower')
    #plt.xlabel('j')
    #plt.ylabel('i')
    #plt.clim(-0.5,0.5)
    #plt.colorbar()
    #plt.colorbar(fraction=0.045, pad=0.05,ticks=[-0.5,0,0.5])
    #plt.savefig('contact_map.pdf', format='pdf', dpi=100)

    return ct

#===================================================================================================
def direct_info(s0,w,h0):
    w = (w+w.T)/2. # make w to be symmetry

    n = s0.shape[1]
    mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
    mx_cumsum = np.insert(mx.cumsum(),0,0)
    i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
        
    di = np.zeros((n,n)) # direct information
    for i in range(n):
        i1,i2 = i1i2[i,0],i1i2[i,1]

        for j in range(n):
            j1,j2 = i1i2[j,0],i1i2[j,1]

            pij = np.exp(w[i1:i2,j1:j2] + h0[i1:i2,np.newaxis] + h0[np.newaxis,j1:j2])            
            pij /= pij.sum()

            pi,pj = pij.sum(axis=1),pij.sum(axis=0)
            di[i,j] = (pij*np.log(pij/np.outer(pi,pj))).sum()
            
    return di        
