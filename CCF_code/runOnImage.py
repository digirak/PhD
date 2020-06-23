import parallelCompareTemplates 
from parallelCompareTemplates import Template
from glob import glob
import numpy as np

vels=np.arange(-1000,1000,20)
vel_pixel_matrix=np.reshape(np.zeros(101*101*len(vels)),(101,101,len(vels)))
keys_matrix=np.reshape(np.chararray(101*101),(101,101))
files=glob("/mnt/disk4tb/Users/sabyasachi/BT_Settl/lte3*")
for x in range(40,50):
    for y in range(40,50):
        print("Computing for %d,%d"%(x,y))

        comp=Template(x,y,vels)
        #for f in files:
        f="/mnt/disk4tb/Users/sabyasachi/BT_Settl/lte4600-4.00-0.0a+0.0.BT-dusty-giant-2013.cf128.sc.spid.fits"
        crosscorr_dict=comp.compareTemplate(f)
        #if(crosscorr_dict==0):
         #   continue
        max_vals=[]
        keys=list(crosscorr_dict.keys())
        #for key in crosscorr_dict.keys():
         #   max_vals.append(abs(np.max(crosscorr_dict[key])-np.mean(crosscorr_dict[key])))
        #if not max_vals:
            #continue
        #ax_key=keys[np.argmax(max_vals)]
        vel_pixel_matrix[x,y,:]=crosscorr_dict[keys[0]]
       # keys_matrix[x,y]=max_key
np.save("Velocity_matrix.npy",vel_pixel_matrix)
#np.save("keys_matrix.npy",keys_matrix)

#plt.plot(vels,crosscorr_dict[max_key])
#plt.savefig("Trial1.png")



