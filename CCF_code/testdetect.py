from CCFcore.SyntheticData import SynData
import numpy as np
from matplotlib import pyplot as plt
from vip_hci.fits import write_fits
syndat=SynData("/Users/rakesh/Data/Templates/BT-Settl_M-0.0a+0.0/lte013.0-3.5-0.0a+0.0.BT-Settl.spec.fits.gz",
"/Users/rakesh/Data/Templates/BT-Settl_M-0.0a+0.0/lte070.0-5.5-0.0a+0.0.BT-Settl.spec.fits.gz")
def formatFunc(x):
    return '{0:1.0e}'.format(x)
def formatFunc2(x):
    return '{0:3.2f}'.format(x)


teff_c=np.float(syndat.companion_file.split('/')[-1].split('-')[0][-4:])*1e2
logg_c=np.float(syndat.companion_file.split('/')[-1].split('-')[1])
wmin = 1.0
wmax = 3.0  
Cmin = -6.0
Cmax = -1.3
Cnum = 30
dsnr_min = 2
dsnr_max = 8
dsnr_num = 30
mat,dsnrs,Rs = syndat.computeDetectionMatrix(wmin,wmax,Cmin,Cmax,Cnum,dsnr_min,dsnr_max,dsnr_num)



mask=np.ones_like(mat)*np.nan
mask[np.nonzero(mat)]=mat[np.nonzero(mat)]
plt.pcolor(np.log10(mask[::,::]),ec='k',cmap='Spectral')

R_labs=[]
for R in Rs[::]:
    R_labs.append(formatFunc(R))
snr_labs=[]
for snr in dsnrs[::]:
    snr_labs.append(formatFunc(snr))

    
plt.xticks(np.arange(len(Rs))+0.5,R_labs[::])
plt.yticks(np.arange(dsnr_num)+0.5,snr_labs[::])
#ax.set_xticklabels(cont_labs)
for x in range(len(R_labs))[::]:
    for y in range(len(dsnrs))[::]:
        if(np.isnan(mask[y,x])):
            #print("nan")
            plt.text(x+0.5,y+0.5,"--",ha='center',va='center',fontsize=14)
        else:
            plt.text(x+0.5,y+0.5,"{0:1.0e}".format(mat[y,x]),ha='center',va='center',fontsize=12)
plt.xlabel("$R$")
plt.ylabel("$\\rm{SNR}$")
plt.title("For a companion Teff={0:3.2f}, logg = {1:3.2f} between {2:3.2f} - {3:3.2f} $\mu$m".format(teff_c,logg_c,wmin,wmax))
plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_22.11.2021/detection_matrix_wmin_{0:3.1f}_wmax_{1:3.1f}_Teff_{2:3.2f}_logg_{3:3.2f}_dex.png"
.format(wmin,wmax,teff_c,logg_c),transparent=True,facecolor='white')
write_fits("/Users/rakesh/Documents/Monday_meetings/Plots_22.11.2021/FITS/detmatrix_wmin_{0:3.1f}_wmax_{1:3.1f}_Teff_{2:3.2f}_logg_{3:3.2f}_dex.fits"
.format(wmin,wmax,teff_c,logg_c),mat)