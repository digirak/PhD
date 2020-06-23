from callPCACrossCorr import SINFONI
import matplotlib as mpl
from photutils import CircularAperture,aperture_photometry
def measureSpatialSpec(datcube,loc):
    slices=[]
    for wl in range(datcube.shape[0]):
        apertures=CircularAperture([loc[0],loc[1]],r=s.fwhm[0].data[wl]/2)
        #print(aperture_p)
        slices.append(np.float64(aperture_photometry(datcube[wl,:,:],apertures)['aperture_sum']))
    return np.asarray(slices)
font = {'family' : 'serif','weight' : 'bold'}
mpl.rc('font', **font)
from matplotlib import pyplot as plt
import numpy as np
import sys
from parallelCompareTemplates import CrossCorr
import configparser
configpath=sys.argv[1]
if(configpath==[]):
    configpath="datconfig.ini"
    print("Taking datconfig.ini")
config = configparser.ConfigParser()
config.read(configpath)
vmin=np.float(config.get("wavelengthparams","vmin"))
vmax=np.float(config.get("wavelengthparams","vmax"))
dv=np.float(config.get("wavelengthparams","dv"))

vels=np.arange(vmin,vmax,dv)
CC=CrossCorr(vels)
from glob import glob
from scipy.interpolate import interp1d
#files=glob("/Users/rakesh/Data/Templates/BT-Settl/lte15*")
#files.sort()
#fl=files[5]
fl=config.get("paths","template_path")
Teff=fl.split('/')[-1].split('-')[0][-4:]
logg=fl.split('/')[-1].split('-')[1]
temp_flux,temp_wavs=CC.processTemplate(fl)
n_comps=int(config.get("wavelengthparams",'n_comps'))
wmin_max=[np.float(config.get("wavelengthparams","wmin")),
          np.float(config.get("wavelengthparams","wmax"))]#[2.2,2.5]
datapath=config.get("paths","datapath")#"/Users/rakesh/Data/PDS70v2020/"
s=SINFONI(datpath=datapath,filename=config.get("paths","cubename"),#"good_ASDI_cube_skysub_derot_med_WLcorr.fits",
          wavelen=config.get("paths","wavelen_file"),#"good_lambdas_WLcorr.fits",
          fwhm=config.get("paths","fwhm_file"),#"good_fwhm_WLcorr.fits",
          sz=int(config.get("imageparams","crop_size")),
          wmin_max=wmin_max,
          wmin_wmax_tellurics=[np.float(config.get("wavelength_params","tell_wmin",fallback=1.75)),
                               np.float(config.get("wavelength_params","tell_wmax",fallback=2.15))])
window_size=int(config.get("wavelengthparams","window_size",fallback=101))
order=int(config.get("wavelengthparams","order",fallback=1))
im_pc=s.valentinPreProcessSINFONI(n_comps=n_comps,window_size=window_size,polyorder=order)
snrmatrix=np.reshape(np.zeros(s.crop_sz*s.crop_sz),(s.crop_sz,s.crop_sz))
noisemat=np.zeros(snrmatrix.shape)
xmin=int(config.get("imageparams","xmin"))
xmax=int(config.get("imageparams","xmax"))
ymin=int(config.get("imageparams","ymin"))
ymax=int(config.get("imageparams","ymax"))
for xx in range(xmin,xmax):
    for yy in range(ymin,ymax):
        spec=measureSpatialSpec(im_pc,[xx,yy])
        import io
        from contextlib import redirect_stdout
        trap = io.StringIO()
        with redirect_stdout(trap):
            ccf_nopc, noise_nopc, snr = CC.compareFluxes(s.waves_dat,
                                                         spec,
                                                         temp_wavs,
                                                         temp_flux,
                                                         window_size=window_size,
                                                         order=order)
        snrmatrix[xx,yy]=snr
        noisemat[xx,yy]=noise_nopc
        sys.stdout.flush()
        sys.stdout.write("Completed %d %d in pixels\r"%(xx,yy))
        #print(snr)
        #print("Completed %d %d in pixels"%(xx,yy))

    #plt.imshow(matrix[:,:,0])
    #plt.colorbar()
    #plt.savefig("vel_matrix.png",dpi=800)
    #plt.close()
fname=s.datpath.split('/')[-2]
frame_size=xmax-xmin
np.save("../Results_CCF/valentinsnrmatrix_for_%s_framesize_%d_PCs_%d_wmin_%1.1f_wmax_%1.1f_Teff_%d_logg_%3.2f.npy"%(fname,frame_size,n_comps,wmin_max[0],wmin_max[1],int(Teff),float(logg)),snrmatrix)
#np.save("noisematrix_%d_%1.1f_%1.1f_Teff_%d_logg_%3.2f.npy"%(n_comps,wmin_max[0],wmin_max[1],int(Teff),float(logg)),noisemat)
