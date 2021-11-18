
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
def computeParPlot(files,waves_dat,spec):
  
    Teff=[]
    logg=[]
    files.sort()
    ccfs=[]
    snrs=[]
    spectra=[]
    template_spec=[]
    i=0
    vels=np.arange(-1,1,1)
    CC_obj=CrossCorr(vels)
    snrs=[]
    vels=np.arange(-2000,2000,20)
    CC=CrossCorr(vels)
    for fl in files:
        temp_flux,temp_wavs=CC_obj.processTemplate(fl)
        Teff.append(fl.split('/')[-1].split('-')[0][-4:])
        logg.append(fl.split('/')[-1].split('-')[1])
        
        print("Teff,logg is %3.2f,%3.2f"%(float(Teff[i]),float(logg[i])))
        ccf_nopc,noise_nopc,snr=CC_obj.compareFluxes(waves_dat,
                                             spec,
                                             temp_wavs,
                                             temp_flux,
                                             window_size=window_size,
                                             order=order,
                                             wmin_wmax_tellurics=[1.8,1.9])
        spectra.append(CC_obj.f1)
        template_spec.append(CC_obj.f2)
        ccf_nopc,noise_nopc,snr=CC.compareFluxes(waves_dat,
                                             spec,
                                             temp_wavs,
                                             temp_flux,
                                             window_size=window_size,
                                             order=order,
                                             wmin_wmax_tellurics=[1.8,1.9])
        snrs.append(snr)
        
        i=i+1

    return spectra, template_spec, Teff, logg,snrs
def scaleByStd(vector):
    vector=np.asarray(vector,dtype=np.float64)
    return vector/vector.std()
files=glob.glob("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/Telluricsadded/TelluricsIncl/TelluricsIncl/SNR/*")
filenames=[]
window_size=101
order=1
t1=Table.read("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/astrometry.fits")

p=t1.to_pandas()

p=p.drop_duplicates()

p.loc[((p['wmin']==1.9)& (p['wmax']==2.5)),'band']='K'
p.loc[((p['wmin']==1.9)& (p['wmax']==2.3)),'band']='K (1.9-2.3)'
p.loc[((p['wmin']==2.0)& (p['wmax']==2.4)),'band']='K (2.0-2.4)'

p.loc[((p['wmin']==1.4) &(p['wmax']==2.5)),'band']='Whole'
p.loc[((p['wmin']>=1.4) &(p['wmax']<=1.9)),'band']='H'
p.loc[((p['wmin']>=1.4) &(p['wmax']<1.9)),'band']='H short'
print(p.columns)
p.dropna(inplace=True)
datapath="/Users/rakesh/Data/Final_derot_residual_cubes_HD142527/Rakesh_derot/"
for fi in files:
    filenames.append(fi.split('/')[-1][-55::])
for i in range(20,30):
    print(filenames[i])
    PCs=int(filenames[::][i].split('_')[3])
    wmin=np.float64(filenames[::][i].split('_')[5])
    wmax=np.float64(filenames[::][i].split('_')[7])
    print("Computing the values between {0} and {1} for {2} PCs".format(wmin,wmax,PCs))
    xx=p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'xx_rakesh_nopsf']
    yy=p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'yy_rakesh_nopsf']
    if(p.loc[((p['wmin']==wmin) &(p['wmax']==wmax) &(p['PCs']==PCs)),'band' ].values=='Whole'):
        print("skipping because we don't want the whole band")
        continue
    wmin_max_ranges=np.arange(wmin, wmax+0.1, 0.1)
    for w in range(0,len(wmin_max_ranges)-1):
        wmin=wmin_max_ranges[w]
        wmax=wmin_max_ranges[w+1]
        if((wmin==1.8) &(wmax==1.9) ):
            print("Just tellurics")
            continue
        
        s=SINFONI(datpath=datapath,filename="residuals_%dPCs_includetell.fits"%PCs,
              wavelen="lbda_vec_avg.fits",#"good_lambdas_WLcorr.fits",
              fwhm="fwhm_vec_WLcorr.fits",#"good_fwhm_WLcorr.fits",
              sz=70,
              wmin_max=[wmin,wmax])
        
        print("Running between {0} mum and {1} mum".format(wmin,wmax))
        if((s.wmin_max[1]>np.max(s.wavelen[0].data))|(s.wmin_max[0]<np.min(s.wavelen[0].data))):
            print("Skip because of wavelength range error")
            continue
        print(np.min(s.wavelen[0].data))
        _=s.preProcessSINFONI(n_comps=0,window_size=window_size,polyorder=order)
        loc_low=find_nearest(s.wavelen[0].data,s.wmin_max[0])#np.argmin(abs(s.waves_dat-s.wmin_max[0]))
        loc_high=find_nearest(s.wavelen[0].data,s.wmin_max[1])#np.argmin(abs(s.waves_dat-s.wmin_max[1]))
        #print(s.wavelen[0].data[loc_low],s.wavelen[0].data[loc_high])
        im_pc=s.cube[0].data[loc_low:loc_high]
        fwhm=s.fwhm[0].data[loc_low:loc_high]
        cent=frame_center(im_pc)



        loc=[xx+int(cent[0]),yy+int(cent[1])]
        spec=measureSpatialSpec(im_pc,loc,fwhm)

        temp_files=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte3*-5.00-*")
        #temp_files=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte3000*")
        spec_orig, temp_spec, teff, logg,snrs = computeParPlot(temp_files,s.waves_dat,spec)
        plt.plot(s.waves_dat[:],scaleByStd(spec_orig[0][:]),c='k')
        for i in range(len(logg[0:])):
        #pos=np.argmax(snrs)
            plt.plot(s.waves_dat[:],
                     scaleByStd(temp_spec[i])[:],label='teff {0} snr {1:2.1f}'.format(teff[i],snrs[i]),alpha=0.5)
        plt.legend(loc='best')
        plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_12.10.2020/SpectraPlots/Teffplots_wmin_{0:1.1f}_wmax_{1:1.1f}_PCs_{2:1d}.png".format(wmin,wmax,PCs)
                    ,dpi=800)
        plt.close()
        print("Doing The loggs")
        temp_files=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte3500*")
        #temp_files=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte3000*")
        spec_orig, temp_spec, teff, logg,snrs = computeParPlot(temp_files,s.waves_dat,spec)
        plt.plot(s.waves_dat[:],scaleByStd(spec_orig[0][:]),c='k')
        for i in range(len(logg[0:])):
        #pos=np.argmax(snrs)
            plt.plot(s.waves_dat[:],
                     scaleByStd(temp_spec[i])[:],label='logg {0} snr {1:2.1f}'.format(logg[i],snrs[i]),alpha=0.5)
        plt.legend(loc='best')
        plt.savefig("/Users/rakesh/Documents/Monday_meetings/Plots_12.10.2020/SpectraPlots/loggplots_wmin_{0:1.1f}_wmax_{1:1.1f}_PCs_{2:1d}.png".format(wmin,wmax,PCs)
                    ,dpi=800)
        plt.close()