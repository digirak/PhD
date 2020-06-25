from callPCACrossCorr import SINFONI
import matplotlib as mpl
from photutils import CircularAperture,aperture_photometry
from astropy.io import fits
import vip_hci
import sys
def measureSpatialSpec(datcube,loc,fwhm):
    slices=[]
    for wl in range(datcube.shape[0]):
        apertures=CircularAperture([loc[0],loc[1]],r=fwhm[wl]/2)
        #print(aperture_p)
        slices.append(np.float64(aperture_photometry(datcube[wl,:,:],apertures)['aperture_sum']))
    return np.asarray(slices)
def read_fits(filename):
    return fits.open(filename)[0].data
font = {'family' : 'serif','weight' : 'bold'}
mpl.rc('font', **font)
from matplotlib import pyplot as plt
import numpy as np
import sys
from parallelCompareTemplates import CrossCorr
vmin=-2000
vmax=2000
dv=20
vels=np.arange(vmin,vmax,dv)
CC=CrossCorr(vels)
from glob import glob
from scipy.interpolate import interp1d
files=glob("/Users/rakesh/Data/Templates/BT-Settl/lte15*")
files.sort()
fl=files[2]
Teff=fl.split('/')[-1].split('-')[0][-4:]
logg=fl.split('/')[-1].split('-')[1]
temp_flux,temp_wavs=CC.processTemplate(fl)
window_size=101
order=1
filename="/Users/rakesh/Data/Final_derot_residual_cubes/final_derot_res_cube_npc5_H_good0-28.fits"
fits_h=read_fits(filename)
filename="/Users/rakesh/Data/Final_derot_residual_cubes/final_derot_res_cube_npc5_K_good0-28.fits"
fits_k=read_fits(filename)
filename="/Users/rakesh/Data/Final_derot_residual_cubes/lbda_vec_avg_K.fits"
waves_k=read_fits(filename)
filename="/Users/rakesh/Data/Final_derot_residual_cubes/lbda_vec_avg_H.fits"
waves_h=read_fits(filename)
fwhm_h=read_fits("/Users/rakesh/Data/Final_derot_residual_cubes/fwhm_vec_avg_H.fits")
fwhm_k=read_fits("/Users/rakesh/Data/Final_derot_residual_cubes/fwhm_vec_avg_K.fits")
cropped_k=vip_hci.hci_dataset.cube_crop_frames(fits_k,71)
cropped_h=vip_hci.hci_dataset.cube_crop_frames(fits_h,71)
wavs=np.array(list(waves_h)+list(waves_k))
snr_mat=np.zeros_like(cropped_h[0,:,:])
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
        sys.stdout.flush()
        sys.stdout.write("Completed %d %d in pixels\r"%(xx,yy))
np.save("/Users/rakesh/Code/Results_CCF/valentin_residuals.npy",snr_mat)
