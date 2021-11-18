from matplotlib import pyplot as plt
from CCFcore._utils import find_nearest
import numpy as np
import glob
from vip_hci.fits import open_fits, write_fits
from scipy.signal import convolve2d
import pandas as pd
from astropy.table import Table
import glob
from vip_hci.fits import open_fits, write_fits
from vip_hci.var.shapes import frame_center
from vip_hci.var.fit_2d import fit_2dgaussian

path="/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/Centered/SNR/"
files=glob.glob(path+"derot_*.fits")
files.sort()
normalized_psf=open_fits("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/normalize_psf.fits")

def calc_astrometry(snr,tol=80):
    loc=np.unravel_index(np.argmax(snr),snr.shape)
    print("Maximum at {0} of {1} snr".format(loc,snr[loc]))
    maxsnr=np.max(snr)
    locs_snr=np.where((snr)>=tol*maxsnr/100)
    #print(snr[locs_snr])
    wt_average=np.average(locs_snr,axis=1,weights=snr[locs_snr])
    return wt_average
vlocs=open_fits("/Users/rakesh/Data/HD142527/Final_median_params_HD142527B_simplex.fits")
vwavs=open_fits("/Users/rakesh/Data/HD142527/lbda_vec_corr_avg_detections.fits")
wavelen=open_fits("/Users/rakesh/Data/HD142527_fulldata/lbda_vec_avg.fits")
vlocs_whole=np.zeros(3*len(wavelen)).reshape(3,len(wavelen))
for i in range(len(wavelen)):
    vlocs_whole[:,i] = vlocs[:,find_nearest(vwavs,wavelen[i])]
df=pd.DataFrame(data=None, columns=['PCs','wmin','wmax','xx_rakesh_nopsf','yy_rakesh_nopsf','xx_rakesh_psf','yy_rakesh_psf','snr','snr_psf','xx_valentin','yy_valentin','xx_gauss','yy_gauss'])
filenames=[]
for fi in files:
    filenames.append(fi.split('/')[-1][-55::])
loc_xx_yy=np.zeros(2*len(files[0::])).reshape(2,len(files[0::]))
loc_xx_yy_psf=np.zeros_like(loc_xx_yy)

for i in range(len(filenames)):
    print(filenames[i])
    PCs=int(filenames[::][i].split('_')[3])
    wmin=np.float64(filenames[::][i].split('_')[5])
    wmax=np.float64(filenames[::][i].split('_')[7])
    print("Computing the values between {0} and {1} for {2} PCs".format(wmin,wmax,PCs))


    loc_low=find_nearest(wavelen,wmin)#np.argmin(abs(s.waves_dat-s.wmin_max[0]))
    loc_high=find_nearest(wavelen,wmax)
    r,theta,cont=np.average(vlocs_whole[:,loc_low:loc_high],axis=1)
    xx_v,yy_v=(r*np.cos(np.deg2rad(theta)),r*np.sin(np.deg2rad(theta)))
    print("Valentin's location for this is {}".format([xx_v,yy_v]))

    fi=files[i]
    f=open_fits(fi)
    print(fi.split('/')[-1])
    #print(fi)
    #i=0#REMOVE THIS
    f_psf=convolve2d(f,normalized_psf)
    #loc_xx_yy[:,i]=calc_astrometry(f,80)-frame_center(f)

    xx,yy=calc_astrometry(f,80)-frame_center(f)
    max_loc=np.unravel_index(np.argmax(f_psf),f_psf.shape)

    val=fit_2dgaussian(np.transpose(f_psf),full_output=True,debug=False,crop=True,
                   fwhmx=2.5,fwhmy=2.5,cent=max_loc,cropsize=11)
    #val=fit_2dgaussian(f_psf.T,full_output=True,debug=False)
    #xx,yy=np.where(f==np.max(f))-frame_center(f)
    cen =frame_center(f_psf)
    snr=f_psf[np.int(xx)+int(cen[0]),np.int(yy)+int(cen[1])]
    #loc_xx_yy_psf[:,i]=calc_astrometry(f_psf,80)-frame_center(f_psf)
    xx_psf,yy_psf=calc_astrometry(f_psf,80)-frame_center(f_psf)
    x,y=calc_astrometry(f_psf,80)
    loc=[int(x),int(y)]
    #xx_psf=np.float(loc[0]-frame_center(f_psf)[0])
    #yy_psf=np.float(loc[1]-frame_center(f_psf)[1])
    snr_psf=f_psf[np.int(xx_psf)+int(cen[0]),np.int(yy_psf)+int(cen[1])]
    xx_2d=np.float(val['centroid_x'].values-cen[0])
    yy_2d=np.float(val['centroid_y'].values-cen[1])
    
    #xx,yy=np.mean(loc_xx_yy,axis=1)
    #xx_psf,yy_psf=np.mean(loc_xx_yy_psf,axis=1)
    #xx_std,yy_std=np.std(loc_xx_yy,axis=1)


    #print(np.mean(loc_xx_yy,axis=1))
    #print("Best location is {0} with a standard dev of {1} in RA".format(xx*12.5/1000,xx_std*12.5/1000))
    #print("Best location is {0} with a standard dev of {1} in Dec".format(yy*12.5/1000,yy_std*12.5/1000))
    #print("Best location is {0} with a standard dev of {1} in RA with psf".format(xx_psf*12.5/1000,xx_std*12.5/1000))
    #print("Best location is {0} with a standard dev of {1} in Dec with psf".format(yy_psf*12.5/1000,yy_std*12.5/1000))

    s1=[PCs,wmin,wmax,xx,yy,xx_psf,yy_psf,snr,snr_psf,xx_v,yy_v,xx_2d,yy_2d]
    df.loc[i]=s1
print(df)

t=Table.from_pandas(df)
t.write("/Users/rakesh/Results/Results_CCF/HD142527v2020_residualsFullbaseline/HD142527v2020_fromalan/astrometry_correctrot.fits",overwrite=True)
