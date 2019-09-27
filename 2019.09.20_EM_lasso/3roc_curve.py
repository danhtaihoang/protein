# 2018.12.27: Receiver operating characteristic (ROC) curve
import sys
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#=========================================================================================

pfam_id = 'PF00186'
#ipdb = 1
#pfam_id = sys.argv[1]
#ipdb = sys.argv[2]
#ipdb = int(ipdb)

#ext_name = '%s/%02d'%(pfam_id,ipdb)

"""
try:
    di = np.loadtxt('%s_di.dat'%ext_name).astype(float)
    ct = np.loadtxt('%s_ct.dat'%ext_name).astype(float)
except:
    pass    
"""

di = np.loadtxt('di.txt').astype(float)
ct = np.loadtxt('ct.txt').astype(float)

#===================================================================================================
def roc_curve(ct,di,ct_thres):    
    ct1 = ct.copy()
    
    ct_pos = ct1 < ct_thres           
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0],dtype=bool), k=1)
    order = di[mask].argsort()[::-1]

    ct_flat = ct1[mask][order]

    tp = np.cumsum(ct_flat, dtype=float)
    fp = np.cumsum(~ct_flat.astype(int), dtype=float)

    if tp[-1] !=0:
        tp /= tp[-1]
        fp /= fp[-1]
    
    return tp,fp
#===================================================================================================
# find optimal threshold of distance
ct_thres = np.linspace(0.5,20.5,41,endpoint=True)
n = ct_thres.shape[0]

auc = np.zeros(n)
for i in range(n):
    tp,fp = roc_curve(ct,di,ct_thres[i])
    auc[i] = tp.sum()/tp.shape[0]    
    #print(ct_thres[i],auc[i])
   
np.savetxt('auc.dat',(ct_thres,auc),fmt='%f')
#-----------------------------------------------------------------------------------------
# plot at optimal threshold:
i0 = np.argmax(auc)    
print('auc max:',ct_thres[i0],auc[i0])
tp0,fp0 = roc_curve(ct,di,ct_thres[i0])

iplot = [1,3,5,7,9,11,13]
plt.figure(figsize=(9.0,3.2))

plt.subplot2grid((1,3),(0,0))
plt.title('ROC various thres')
plt.plot([0,1],[0,1],'k--')
for i in iplot:
    tp,fp = roc_curve(ct,di,ct_thres[i])    
    plt.plot(fp,tp,label='thres = %s'%ct_thres[i])

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.subplot2grid((1,3),(0,1))
plt.title('ROC at thres = %3.2f'%(ct_thres[i0]))
plt.plot(fp0,tp0,'r-')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.subplot2grid((1,3),(0,2))
plt.title('AUC max = %f' %(auc[i0]))
plt.plot([ct_thres.min(),ct_thres.max()],[0.5,0.5],'k--')
plt.plot(ct_thres,auc,'r-')
plt.xlim([ct_thres.min(),ct_thres.max()])
plt.ylim([0,auc.max()+0.1])
plt.xlabel('distance threshold')
plt.ylabel('AUC')

plt.tight_layout(h_pad=1, w_pad=1.5)
#plt.savefig('%s_roc_curve.pdf'%ext_name, format='pdf', dpi=100)
plt.savefig('roc_curve.pdf', format='pdf', dpi=100)

#plt.show()
