from matplotlib import pyplot as plt
import numpy as np
import glob
from vip_hci.fits import open_fits, write_fits
from scipy.signal import convolve2d
path="/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/Telluricsadded/TelluricsIncl/Corners_parmap/SNR/"
files=glob.glob(path+"*.fits")
files.sort()
print(len(files))
plt.style.use("seaborn-poster")
import matplotlib as mpl
import warnings
font = {'family' : 'serif',
        'weight' : 'bold'}

mpl.rc('font', **font)
#f="vmatrix_2_1.5_1.8.npy"
psf=open_fits("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/normalize_psf.fits")
def create_circle(loc,r,color,fill=False,label=None):
	circle= plt.Circle(loc, radius= r,fill=fill,color=color,lw=2.0,label=label)
	return circle
def calc_astrometry(snr,tol=80):
    loc=np.unravel_index(np.argmax(snr),snr.shape)
    print("Maximum at {0} of {1} snr".format(loc,snr[loc]))
    maxsnr=np.max(snr)
    locs_snr=np.where((snr)>=tol*maxsnr/100)
    #print(snr[locs_snr])
    wt_average=np.average(locs_snr,axis=1,weights=snr[locs_snr])
    return wt_average
#fname=files[10]
for fname in files[:]:
    f=plt.figure()
    #f.set_size_inches(6,6)
    ax=plt.gca()
    snr=open_fits(fname)
    if(len(snr.shape)>2):
        snrplot=snr[100,10:61,10:61]
        snrplot=convolve2d(snrplot,psf,mode='same')
    else:
        snrplot=snr[:,:]
        snrplot=convolve2d(snr,psf,mode='same')
    vloc=[5,3]
    plt.pcolormesh(snrplot.T,cmap='rainbow')
    plt.colorbar()
    p=create_circle([25-vloc[0],25-vloc[1]],r=2.65,color='darkcyan',fill=False,label='Valentin')
    ax.add_patch(p)
    plt.text(25-vloc[0],25-vloc[1],"(-{0},-{1})".format(np.round(((vloc[0])*12.5/1000),3),np.round(((vloc[1])*12.5/1000),3)),
    ha='center',va='center',fontsize=10)
    loc_derived=calc_astrometry(snrplot,85)
    p=create_circle(loc_derived,r=2.65,color='indianred',fill=False,label='Rakesh')
    ax.add_patch(p)
    plt.text(loc_derived[0],loc_derived[1],"(-{0},-{1})".format(np.round(((25-loc_derived[0])*12.5/1000),3),np.round(((25-loc_derived[1])*12.5/1000),3)),
    ha='center',va='center',fontsize=10)
    plt.legend()
    plt.xticks(ax.get_xticks()[0:-1],np.round((np.arange(10,61,10)-35)*12.5/1000,2))
    plt.yticks(ax.get_yticks()[0:-1],np.round((np.arange(10,61,10)-35)*12.5/1000,2))
    #plt.tight_layout()
    fname=fname.split("/")[-1]
    print(fname)
    PCs=int(fname.split("_")[10])
    wmin=float(fname.split("_")[12])
    wmax=float(fname.split("_")[14])
    Teff=float(fname.split("_")[16])
    logg=float(fname.split("_")[18][0:4])
    plt.title("wmin$={1}$ $\mu$m wmax$={2}$ $\mu$m PCs$={0}$".format(PCs,wmin,wmax),fontweight='bold')
    plt.xlabel("arcsecond")
    plt.ylabel('arcsecond')
    plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_12.10.2020/snr_pcs_{0}_wmin_{1}_wmax_{2}_Teff_{3}_logg_{4}.png".format(PCs,wmin,wmax,Teff,logg),dpi=300)
    plt.close()