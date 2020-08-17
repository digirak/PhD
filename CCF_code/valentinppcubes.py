import matplotlib as mpl
from CCFcore.PreProcess import SINFONI
from CCFcore.PreProcess import applyFilter
from CCFcore import removeTelluric
from CCFcore.CrossCorr import CrossCorr
from CCFcore.PreProcess import measureSpatialSpec
from photutils import CircularAperture,aperture_photometry
from astropy.io import fits
import vip_hci
import sys
from vip_hci.fits import open_fits, write_fits
font = {'family' : 'serif','weight' : 'bold'}
mpl.rc('font', **font)
from matplotlib import pyplot as plt
import numpy as np
import sys
vmin=-2000
vmax=2000
dv=20
vels=np.arange(vmin,vmax,dv)
CC=CrossCorr(vels)
from glob import glob
from scipy.interpolate import interp1d
files=glob("/Users/rakesh/Data/Templates/BT-Settl/lte14*")
files.sort()
fl=files[2]
Teff=np.float(fl.split('/')[-1].split('-')[0][-4:])
logg=np.float(fl.split('/')[-1].split('-')[1])
temp_flux,temp_wavs=CC.processTemplate(fl)
window_size=101
order=1
npc=5
filename="/Users/rakesh/Data/Final_derot_residual_cubes/final_derot_res_cube_npc{}_H_good0-28.fits".format(npc)
fits_h=open_fits(filename)
filename="/Users/rakesh/Data/Final_derot_residual_cubes/final_derot_res_cube_npc{}_K_good0-28.fits".format(npc)
fits_k=open_fits(filename)
filename="/Users/rakesh/Data/Final_derot_residual_cubes/lbda_vec_avg_K.fits"
waves_k=open_fits(filename)
filename="/Users/rakesh/Data/Final_derot_residual_cubes/lbda_vec_avg_H.fits"
waves_h=open_fits(filename)
fwhm_h=open_fits("/Users/rakesh/Data/Final_derot_residual_cubes/fwhm_vec_avg_H.fits")
fwhm_k=open_fits("/Users/rakesh/Data/Final_derot_residual_cubes/fwhm_vec_avg_K.fits")
cropped_k=vip_hci.hci_dataset.cube_crop_frames(fits_k,71)
cropped_h=vip_hci.hci_dataset.cube_crop_frames(fits_h,71)
wavs=np.array(list(waves_h)+list(waves_k))
wmin=np.min(wavs)
wmax=np.max(wavs)
print("On baseline from {00:03f} mu m to {01:03f} mu m".format(wmin,wmax))
snr_mat=np.zeros_like(cropped_h[0,:,:])
ccf_mat=np.reshape(np.zeros(snr_mat.shape[1]*snr_mat.shape[0]*vels.shape[0]),(vels.shape[0],snr_mat.shape[1],snr_mat.shape[0]))
for xx in range(10,61):
    for yy in range(10,61):
        #print("Working on %d ,%d "%(xx,yy))
        spec=np.array(list(measureSpatialSpec(cropped_h,[yy,xx],fwhm_h))
        +list(measureSpatialSpec(cropped_k,[yy,xx],fwhm_k)))
        import io
        from contextlib import redirect_stdout
        trap = io.StringIO()
        with redirect_stdout(trap):
            ccf_nopc,noise_nopc,snr=CC.compareFluxes(wavs,
                                     spec,
                                     temp_wavs,
                                     temp_flux,
                                     window_size=window_size,
                                     order=order)
        snr_mat[xx,yy]=snr
        ccf_mat[:,xx,yy]=ccf_nopc
        sys.stdout.flush()
        sys.stdout.write("Completed %d %d in pixels\r"%(xx,yy))
#np.save("/Users/rakesh/Code/Results_CCF/valentin_residuals.npy",snr_mat)
write_fits("/Users/rakesh/Results/Results_CCF/valentin_residuals_{00:d}_pcs_Teff_{01:3.2f}_logg_{02:3.2f}_wmin_{03:3.2f}_wmax{04:3.2f}_SNR_PDS70.fits".format(npc,Teff,logg,wmin,wmax),snr_mat)
write_fits("/Users/rakesh/Results/Results_CCF/valentin_residuals{00:d}_pcs_Teff_{01:3.2f}_logg_{02:3.2f}_wmin_{03:3.2f}_wmax{04:3.2f}_CCF_PDS70.fits".format(npc,Teff,logg,wmin,wmax),ccf_mat)

