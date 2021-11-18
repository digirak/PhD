from matplotlib import pyplot as plt
import numpy as np
import glob
from vip_hci.fits import open_fits, write_fits
from scipy.signal import convolve2d
from CCFcore.PreProcess import SINFONI
from CCFcore.PreProcess import applyFilter
from CCFcore import removeTelluric
from CCFcore.CrossCorr import CrossCorr
from CCFcore.PreProcess import measureSpatialSpec
from CCFcore._utils import find_nearest
from vip_hci.var.shapes import frame_center
import pandas as pd
from astropy.table import Table


path="/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/Telluricsadded/TelluricsIncl/TelluricsIncl/SNR/"
files=glob.glob(path+"*.fits")
files.sort()
print(len(files))
plt.style.use("seaborn-poster")
import matplotlib as mpl
import warnings
font = {'family' : 'serif',
        'weight' : 'bold'}

mpl.rc('font', **font)
window_size=101
order=1
def computeParMap(files,waves_dat,spec):
  
    Teff=[]
    logg=[]
    files.sort()
    ccfs=[]
    snrs=[]
    i=0
    for fl in files:
        temp_flux,temp_wavs=CC.processTemplate(fl)
        Teff.append(fl.split('/')[-1].split('-')[0][-4:])
        logg.append(fl.split('/')[-1].split('-')[1])
        
        print("Teff,logg is %3.2f,%3.2f"%(float(Teff[i]),float(logg[i])))
        ccf_nopc,noise_nopc,snr=CC.compareFluxes(waves_dat,
                                             spec,
                                             temp_wavs,
                                             temp_flux,
                                             window_size=window_size,
                                             order=order,
                                             wmin_wmax_tellurics=[1.8,1.9])
        ccfs.append(ccf_nopc)
        snrs.append(snr)
        i=i+1
    teff_labs=np.unique(Teff)
    logg_labs=np.unique(logg)
    
    snrs=np.reshape(snrs,(teff_labs.size,logg_labs.size))

    ccfs=np.reshape(ccfs,(teff_labs.size,logg_labs.size,vels.size))
    return snrs,ccfs,teff_labs,logg_labs
vels=np.arange(-2000,2000,20)
CC=CrossCorr(vels)
t1=Table.read("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/astrometry.fits")

p=t1.to_pandas()

#p=p.drop_duplicates()

p.loc[((p['wmin']==1.9)& (p['wmax']==2.5)),'band']='K'
p.loc[((p['wmin']==1.9)& (p['wmax']==2.3)),'band']='K (1.9-2.3)'
p.loc[((p['wmin']==2.0)& (p['wmax']==2.4)),'band']='K (2.0-2.4)'

p.loc[((p['wmin']==1.4) &(p['wmax']==2.5)),'band']='Whole'
p.loc[((p['wmin']>=1.4) &(p['wmax']<=1.9)),'band']='H'
p.loc[((p['wmin']>=1.4) &(p['wmax']<1.9)),'band']='H short'
p=p.drop_duplicates()
filenames=[]
for fi in files:
    filenames.append(fi.split('/')[-1][-55::])
for i in range(len(filenames)):
    PCs=int(filenames[::][i].split('_')[3])
    if(PCs==5):
        continue
    wmin=np.float64(filenames[::][i].split('_')[5])
    wmax=np.float64(filenames[::][i].split('_')[7])
    #print("Computing the values between {0} and {1} for {2} PCs".format(wmin,wmax,PCs))
    datapath="/Users/rakesh/Data/Final_derot_residual_cubes_HD142527/Rakesh_derot/"
    s=SINFONI(datpath=datapath,filename="residuals_%dPCs.fits"%PCs,
          wavelen="lbda_vec_avg.fits",#"good_lambdas_WLcorr.fits",
          fwhm="fwhm_vec_WLcorr.fits",#"good_fwhm_WLcorr.fits",
          sz=70,
          wmin_max=[wmin,wmax])
    _=s.preProcessSINFONI(n_comps=0,window_size=window_size,polyorder=order)
    loc_low=find_nearest(s.wavelen[0].data,s.wmin_max[0])#np.argmin(abs(s.waves_dat-s.wmin_max[0]))
    loc_high=find_nearest(s.wavelen[0].data,s.wmin_max[1])#np.argmin(abs(s.waves_dat-s.wmin_max[1]))
    #print(s.wavelen[0].data[loc_low],s.wavelen[0].data[loc_high])
    im_pc=s.cube[0].data[loc_low:loc_high]
    fwhm=s.fwhm[0].data[loc_low:loc_high]
    cent=frame_center(im_pc)
    
    xx=p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'xx_valentin']
    yy=p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'yy_valentin']
    
    loc=[xx+int(cent[0]),yy+int(cent[1])]
    spec=measureSpatialSpec(im_pc,loc,fwhm)
    temp_files=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte3*")
    snrs,ccfs,Teffs,loggs=computeParMap(temp_files,s.waves_dat,spec)
    ax=plt.gca()
    plt.pcolormesh(snrs,cmap='plasma')
    cb=plt.colorbar()
    cb.set_label("SNR")
    plt.xticks(ax.get_xticks(),loggs)
    plt.yticks(ax.get_yticks(),Teffs[::2])
    plt.grid(True)
    plt.xlabel("$\log(g)$")
    plt.ylabel("$T_{eff}$")
    plt.title("Parmap Ncomps {0} from {1}$\mu$m to {2}$\mu$m".format(PCs,wmin,wmax),fontweight='bold')
    plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_12.10.2020/Parametermap/parmap_valentin_{}_{}_{}.png".format(PCs,wmin,wmax))
    plt.close()
    
    xx=p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'xx_rakesh_psf']
    yy=p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'yy_rakesh_psf']
    
    loc=[xx+int(cent[0]),yy+int(cent[1])]
    spec=measureSpatialSpec(im_pc,loc,fwhm)
    temp_files=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte3*")
    snrs,ccfs,Teffs,loggs=computeParMap(temp_files,s.waves_dat,spec)
    ax=plt.gca()
    plt.pcolormesh(snrs,cmap='plasma')
    cb=plt.colorbar()
    cb.set_label("SNR")
    plt.xticks(ax.get_xticks(),loggs)
    plt.yticks(ax.get_yticks(),Teffs[::2])
    plt.grid(True)
    plt.xlabel("$\log(g)$")
    plt.ylabel("$T_{eff}$")
    plt.title("Parmap Ncomps {0} from {1}$\mu$m to {2}$\mu$m".format(PCs,wmin,wmax),fontweight='bold')
    plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_12.10.2020/Parametermap/parmap_psf_{}_{}_{}.png".format(PCs,wmin,wmax))
    plt.close()

