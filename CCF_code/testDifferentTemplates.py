from callPCACrossCorr import SINFONI
from parallelCompareTemplates import CrossCorr
from photutils import CircularAperture,aperture_photometry
def measureSpatialSpec(datcube,loc):
    slices=[]
    for wl in range(datcube.shape[0]):
        apertures=CircularAperture([loc[0],loc[1]],r=s.fwhm[0].data[wl]/2)
        #print(aperture_p)
        slices.append(np.float64(aperture_photometry(datcube[wl,:,:],apertures)['aperture_sum']))
    return np.asarray(slices)

import matplotlib as mpl
import glob
font = {'family' : 'serif','weight' : 'bold'}
mpl.rc('font', **font)
from matplotlib import pyplot as plt
import numpy as np
import os
temp_list=glob.glob("/Users/rakesh/Data/Templates/BT-Settl/lte1*")
temp_list.sort()
pars=np.reshape(np.zeros(2*len(temp_list)),(2,len(temp_list)))
i=0
for fi in temp_list:
   # matrix,snr=s.valentinPreProcessSINFONI(5,45,5,45,n_comps=n_comps,temp_file=fi,polyorder=3)
    Teff=fi.split('/')[-1].split('-')[0][-4:]
    logg=fi.split('/')[-1].split('-')[1]
    pars[0][i]=Teff
    pars[1][i]=logg
    i+=1
Teffs=(np.unique(pars[0][:]))
loggs=(np.unique(pars[1][:]))
prefix="/Users/rakesh/Data/Templates/BT-Settl/lte"
suffix_1="0-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0sc2.spid.fits"
suffix_2="0-0.0a+0.0.BT-settl-giant-2013.cf250.tm1.0-0.0.R1-g12p0vo1.spid.fits"
max_matrix=np.reshape(np.zeros(len(Teffs)*len(loggs)),(len(Teffs),len(loggs)))
snr_matrix=np.reshape(np.zeros(len(Teffs)*len(loggs)),(len(Teffs),len(loggs)))
i=0
wmin_max=[2.2,2.5]
datapath="/Users/rakesh/Data/PDS70v2020/"
wmin_max=[2.2,2.5]
n_comps=3
s=SINFONI(datpath=datapath,filename="good_ASDI_cube_skysub_derot_med_WLcorr.fits",
          wavelen="good_lambdas_WLcorr.fits",
          fwhm="good_fwhm_WLcorr.fits",
          sz=71,
          wmin_max=wmin_max,
          wmin_wmax_tellurics=[1.75,2.15])
window_size=101
order=1
im_pc=s.preProcessSINFONI(n_comps=n_comps,window_size=window_size,polyorder=order)
loc=[32,16]
spec=measureSpatialSpec(im_pc,loc)
vels=np.arange(-2000,2000,20)
CC=CrossCorr(vels)
for Teff in Teffs:
    j=0
    for logg in loggs:
        temp_file_1=str(prefix+str(int(Teff))+'-'+str(logg)+suffix_1)
        temp_file_2=str(prefix+str(int(Teff))+'-'+str(logg)+suffix_2)
        if(os.path.isfile(temp_file_1)):
            temp_flux, temp_wavs = CC.processTemplate(temp_file_1)
            ccf_nopc, noise_nopc, snr = CC.compareFluxes(s.waves_dat,
                                                         spec,
                                                         temp_wavs,
                                                         temp_flux,
                                                         window_size=window_size,
                                                         order=order)
            #final,snr=s.valentinPreProcessSINFONI(28,29,30,31,temp_file=temp_file_1,polyorder=3,n_comps=n_comps)
            max_matrix[i,j]=np.max(ccf_nopc)
            snr_matrix[i,j]=snr
        elif(os.path.isfile(temp_file_2)):
            temp_flux, temp_wavs = CC.processTemplate(temp_file_2)
            ccf_nopc, noise_nopc, snr = CC.compareFluxes(s.waves_dat,
                                                         spec,
                                                         temp_wavs,
                                                         temp_flux,
                                                         window_size=window_size,
                                                         order=order)
            max_matrix[i,j]=np.max(ccf_nopc)
            snr_matrix[i,j]=snr
        else:
            print("Neither %s nor %s exists"%(temp_file_1,temp_file_2))
            max_matrix[i,j]=0.
            snr_matrix[i,j]=0.
        j+=1
    i+=1
fname=s.datpath.split('/')[-2]
np.save("../Results_CCF/max_mat_for_%s_wmin_%1.1f_wmax_%1.1f_PCs_%d_at_%d_%d.npy"%(fname,wmin_max[0],wmin_max[1],n_comps,loc[0],loc[1]),max_matrix)
np.save("../Results_CCF/snr_test_for_%s_wmin_%1.1f_wmax_%1.1f_PCs_%d_at_%d_%d.npy"%(fname,wmin_max[0],wmin_max[1],n_comps,loc[0],loc[1]),snr_matrix)
