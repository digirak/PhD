import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from photutils import CircularAperture,aperture_photometry
import vip_hci
from scipy.signal import savgol_filter
from pycorrelate import pcorrelate,ucorrelate
from vip_hci import pca
from sklearn.decomposition import pca
import matplotlib as mpl
from parallelCompareTemplates import Template
font = {'family' : 'serif',
                'weight' : 'bold'}

mpl.rc('font', **font)


datpath="/mnt/disk4tb/Users/rnath/Data/HD142527/"
cube=fits.open(datpath+"ASDI_cube_derot_med.fits")
fwhm=fits.open(datpath+"fwhm_vec.fits")
wavelen=fits.open(datpath+"lbda_vec.fits")
sum_spax=[]
for wl in range(2000):
    sum_spax.append(np.nansum(cube[0].data[wl,:,:]))
for spax in range(len(sum_spax)): 
    cube[0].data[spax,:,:]=cube[0].data[spax,:,:]/sum_spax[spax]
ref_spec=[]
for wl in range(2000):
    ref_spec.append(np.median(cube[0].data[wl,:,:]))
normalized_cube=np.reshape(np.zeros(cube[0].data.size),cube[0].data.shape)
for i in range(101):
    for j in range(101):
        filt=savgol_filter(ref_spec,101,polyorder=3)
        normalized_cube[:,i,j]=cube[0].data[:,i,j]/filt
new_matrix=np.reshape(np.zeros(2000*101*101),(2000,(101*101)))
for i in range(2000):
    for j in range(101):
        for k in range(101):
            new_matrix[i,101*j+k]=normalized_cube[i,j,k]
mat=new_matrix[:,:]
pca_sklearn=pca.PCA(n_components=6)
comps=pca_sklearn.fit(np.transpose(np.reshape(mat,[mat.shape[0],mat.shape[1]])))
mat=np.reshape(mat,(mat.shape[0],10201))
red2=comps.transform(np.transpose(mat))
red3=comps.inverse_transform(red2)
im_cube_new=np.reshape(np.transpose(red3),(2000,101,101))

#waves=np.load("/mnt/diskss/home/rnath/Numpy_PDS70/wavelens_new28.npy")
locs=np.where((wavelen[0].data>=1.6) & (wavelen[0].data<=2.4))
waves=wavelen[0].data[locs]
planet_loc=[20,80]
star_slices=[]
planet_slices=[]
im=im_cube_new
for wl in range(len(locs)):
    aperture_p=CircularAperture(planet_loc,r=fwhm[0].data[locs[0][wl]])
    planet_slices.append(np.float64(aperture_photometry(im[wl,:,:],aperture_p)['aperture_sum']))
vels=np.arange(-2000,2000,20)
comp=Template(planet_loc[0],planet_loc[1],vels)
f="/mnt/disk4tb/Users/sabyasachi/BT_Settl/lte3600-5.00-0.0a+0.0.BT-dusty-giant-2013.cf128.vo0.spid.fits"

crosscorr_dict=comp.compareTemplate(f,im=im)
max_vals=[]
keys=list(crosscorr_dict.keys())
for key in crosscorr_dict.keys():
    max_vals.append(abs(np.max(crosscorr_dict[key])-np.mean(crosscorr_dict[key])))
    vals=crosscorr_dict[key]
np.save("normalized_flux.npy",im)
plt.plot(vels,vals,'.')
plt.savefig("trial.png",dpi=800)
plt.close()
