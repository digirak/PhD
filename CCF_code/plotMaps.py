from matplotlib import pyplot as plt
import numpy as np
path="/mnt/diskss/home/rnath/HD142527/"
f="vmatrix_2_1.5_1.8.npy"


f=plt.figure()
f.set_size_inches(8,8)
ax=plt.gca()
plt.imshow(mats[5:45,5:45,100],origin='lower')
ax.set_aspect(1)
c1=plt.colorbar()
c1.set_label("Max coefficients")
plt.xticks(ax.axes.get_xticks()[1:-1],(np.arange(15,60,5)-35)*0.012)
plt.yticks(ax.axes.get_yticks()[1:-1],(np.arange(15,60,5)-35)*0.012)
plt.title("$v=$"+str(vels[100])+' km/s',fontweight='bold')
plt.xlabel("arcsecond")
plt.ylabel('arcsecond')
