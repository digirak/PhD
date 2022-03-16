from CCFcore.SyntheticData import SynData
import numpy as np
from matplotlib import pyplot as plt
from vip_hci.fits import write_fits
from matplotlib.ticker import FuncFormatter
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

fmt = lambda x: '{:1.0e}'.format(x)
mask=np.ones_like(mat)*0
mask[np.nonzero(mat)]=mat[np.nonzero(mat)]
ax=plt.gca()
plt.pcolormesh(Rs,dsnrs[::],np.log10(mask)[0::,::],cmap='Spectral')
ax.set_xscale('log')
ax.set_yscale('log')

plt.xlabel("$R$")
plt.ylabel("$\\rm{SNR}$")
cb =plt.colorbar()
cbticks =cb.get_ticks()
ticklabs =["{0:1.0e}".format(10**tick) for tick in (cbticks)]
cb.set_ticks(cb.get_ticks())
cb.set_ticklabels(ticklabs)
cb.ax.set_title("$C$")

plt.xlabel("$R$")
plt.ylabel("$\\rm{SNR}$")
plt.title("For a companion Teff={0:3.2f}, logg = {1:3.2f} between {2:3.2f} - {3:3.2f} $\mu$m".format(teff_c,logg_c,wmin,wmax))
plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_20.12.2021/detection_matrix_wmin_{0:3.1f}_wmax_{1:3.1f}_Teff_{2:3.2f}_logg_{3:3.2f}_dex_lowres.png"
.format(wmin,wmax,teff_c,logg_c),transparent=True,facecolor='white')
write_fits("/Users/rakesh/Documents/Monday_meetings/Plots_20.12.2021/FITS/detmatrix_wmin_{0:3.1f}_wmax_{1:3.1f}_Teff_{2:3.2f}_logg_{3:3.2f}_dex_lowres.fits"
.format(wmin,wmax,teff_c,logg_c),mat)