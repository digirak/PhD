from callPCACrossCorr import SINFONI
import matplotlib as mpl
font = {'family' : 'serif','weight' : 'bold'}
mpl.rc('font', **font)
from matplotlib import pyplot as plt
import numpy as np
import sys

vels=np.arange(-2000,2000,20)
n_comps=1
wmin_max=[1.5,2.45]
s=SINFONI(wmin_max=wmin_max,vels=vels,
datpath='/Users/rakesh/Data/HD142527/HD142527-new/')
x_mins=[0,0,56,56]
y_mins=[0,56,0,56]
noise_comps=[]
noise_ims=np.zeros(n_comps*15*15).reshape(n_comps,15,15)
noise_finals=np.zeros(n_comps*71*71).reshape(n_comps,71,71)
for comp in range(n_comps):
    noise_im=np.zeros(s.cube[0].data[0,:,:].shape)
    for x_min,y_min in zip(x_mins,y_mins):
        sys.stdout.write("\rx= "+str(x_min)+"to "+str(x_min+15)+"y= "+str(y_min)+"to "+str(y_min+15))
        sys.stdout.flush()
        matrix,noise,snr=s.valentinPreProcessSINFONI(x_min,x_min+15,y_min,y_min+15,n_comps=comp
        ,temp_file="//Users/rakesh/Data/Templates/BT-Settl/lte3500-2.50-0.0a+0.0.BT-dusty-giant-2013.cf128.vo0.spid.fits",
        polyorder=3)
        maxmat=[]
        #for xx in range(matrix.shape[0]):
        #    for yy in range(matrix.shape[1]):
        #        maxmat.append(max(matrix[xx,yy]))
        #maxmat=np.reshape(maxmat,snr.shape)

        noise_im[x_min:x_min+15,y_min:y_min+15]=noise

        #print("Noise level is %3.2f"%(matrix/snr))
    noise_final=(noise_im[0:15,0:15]+noise_im[56:71,0:15]+noise_im[0:15,56:71]+noise_im[56:71,56:71])/4
    noise_ims[comp,:,:]=noise_final
    noise_finals[comp,:,:]=noise_finals

    noise_comps.append(np.mean(noise_final))
    print("For PC %d subtracted %3.2f noise"%(comp,noise_comps[comp]))
np.save("Noise_mat_%d.npy"%(n_comps),noise_ims)
np.save("Noise_full.npy",noise_finals)
plt.plot(np.arange(n_comps),noise_comps)
plt.savefig("comps_noise_%d.png"%n_comps,dpi=800)
plt.close()


#if(np.shape(matrix)[1]>=2):
    #plt.imshow(matrix[:,:,0])
    #plt.colorbar()
    #plt.savefig("vel_matrix.png",dpi=800)
    #plt.close()
 #   np.save("Meeting_18.11.2019/matrix_%d_%1.1f_%1.1f_new_wcorr.npy"%(n_comps,wmin_max[0],wmin_max[1]),matrix)
  #  np.save("Meeting_18.11.2019/snrmatrix_%d_%1.1f_%1.1f_new_wcorr.npy"%(n_comps,wmin_max[0],wmin_max[1]),snr)
#else:
  #  plt.plot(s.vels,matrix[0][0],'.')
   # plt.savefig("cross_Corr_matrix.png",dpi=800)
   # plt.close()
