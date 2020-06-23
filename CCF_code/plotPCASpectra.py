from matplotlib import pyplot as plt
from callPCACrossCorr import SINFONI
from parallelCompareTemplates import CrossCorr
from computeRadialNoise import computeRadialNoise
import matplotlib as mpl
font = {'family' : 'serif','weight' : 'bold'}
mpl.rc('font', **font)
import numpy as np
from glob import glob
from scipy.interpolate import interp1d
files=glob("/Users/rakesh/Data/Templates/BT-Settl/lte15*")
files.sort()
fl=files[6]
Teff=fl.split('/')[-1].split('-')[0][-4:]
logg=fl.split('/')[-1].split('-')[1]
vels=np.arange(-2000,2000,20)
datapath="/Users/rakesh/Data/PDS70v2020/"
wmin_max=[2.2,2.5]
s=SINFONI(datpath=datapath,filename="good_ASDI_cube_skysub_derot_med_WLcorr.fits",
          wavelen="good_lambdas_WLcorr.fits",
          fwhm="good_fwhm_WLcorr.fits",
          sz=71,
          wmin_max=wmin_max,
          wmin_wmax_tellurics=[1.75,2.15])
window_size=101
order=1
noise_rn_comp_pix=[]
noise_sn_comp_pix=[]
CC=CrossCorr(vels)
temp_flux,temp_wavs=CC.processTemplate(fl)
for i in range(30):
    im_pc=s.preProcessSINFONI(n_comps=i,window_size=window_size,polyorder=order)
    #spec=measureSpatialSpec(im_pc,[28,30])
    spec=im_pc[:,32,16]
    print("With %d PCs removed"%i)
    import io
    from contextlib import redirect_stdout
    trap = io.StringIO()
    with redirect_stdout(trap):


        ccf_pc,noise_pc,snr=CC.compareFluxes(s.waves_dat,spec,temp_wavs,temp_flux,
                                              window_size=window_size,order=order)
        m_pc,mnn,msks=computeRadialNoise(im_pc,[32,16],vels,
                                      s.waves_dat,temp_wavs,
                                      temp_flux,window_size,
                                      order,
                                      aperture_size=[5])
       # ccf_rn_pc,mean_n_pc,snr_rn=CC.compareFluxes(s.waves_dat,spec,temp_wavs,temp_flux,noise=np.std(m_pc),
        #                                             window_size=window_size,order=order)
    noise_rn_comp_pix.append(m_pc)
    noise_sn_comp_pix.append(ccf_pc)
fname=s.datpath.split('/')[-2]
np.save("../Results_CCF/noise_rn_for_%s_wmin_%1.1f_wmax_%1.1f_Teff_%d_logg_%3.2f.npy"%(fname,wmin_max[0],wmin_max[1],int(Teff),float(logg)),noise_rn_comp_pix)
np.save("../Results_CCF/CCF_for_%s_wmin_%1.1f_wmax_%1.1f_Teff_%d_logg_%3.2f.npy"%(fname,wmin_max[0],wmin_max[1],int(Teff),float(logg)),noise_sn_comp_pix)
#plt.plot(n_comps,snrs,c='k',marker='o',mfc='k',ms=6)
#plt.axhline(y=5,c='k',ls='--')
#plt.text(30,5.05,'$5\sigma$',ha='center',va='center')
#plt.axhline(y=3,c='k',ls='--')
#plt.text(30,3.05,'$3\sigma$',ha='center',va='center')
#plt.xlabel("components")
#plt.ylabel("SNR")
