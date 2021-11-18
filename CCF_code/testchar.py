from CCFcore.SyntheticData import SynData
import glob
import numpy as np
files=[]
files_trial=["/Users/rakesh/Data/Templates/BT-Settl_M-0.0a+0.0/lte0{}*".format(i) for i in range(12,20)]
from vip_hci.fits import write_fits
for f in files_trial:
    temp_files=glob.glob(f)
    for t in temp_files:
        files.append(t)
syndat=SynData("/Users/rakesh/Data/Templates/BT-Settl_M-0.0a+0.0/lte013.0-3.5-0.0a+0.0.BT-Settl.spec.fits.gz",
"/Users/rakesh/Data/Templates/BT-Settl_M-0.0a+0.0/lte070.0-5.5-0.0a+0.0.BT-Settl.spec.fits.gz")
dsnr_min = 3
dsnr_max = 8
dsnr_num = 5
Rmin = 3
Rmax = 5
Rnum = 5
dsnrs = np.logspace(dsnr_min,dsnr_max,dsnr_num)
Rs = np.logspace(Rmin,Rmax,Rnum)
print(Rs[-1])
est_teffs = np.zeros(Rnum*dsnr_num).reshape(Rnum,dsnr_num)
est_loggs = np.zeros_like(est_teffs)
est_terrs = np.zeros_like(est_teffs)
est_gerrs = np.zeros_like(est_teffs)
for i in range(dsnr_num):
    for j in range(Rnum):
        if(Rs[j]>2e5):
            print("R is too high")
            break
        rets = syndat.constrainTeffLogg(files,Rs[j],1e-2,dsnrs[i],1.0,3.0)
        teff = rets[0]
        teff_err = rets[1]
        logg = rets[2]
        logg_err = rets[3]
        est_teffs[j,i] = teff
        est_loggs[j,i] = logg
        est_terrs[j,i] = teff_err
        est_gerrs[j,i] = logg_err
write_fits("/Users/rakesh/Documents/Monday_meetings/Plots_08.11.2021/FITS/teff_char_star_new.fits",est_teffs)
write_fits("/Users/rakesh/Documents/Monday_meetings/Plots_08.11.2021/FITS/logg_char_star_new.fits",est_loggs)
write_fits("/Users/rakesh/Documents/Monday_meetings/Plots_08.11.2021/FITS/teff_errs_star_new.fits",est_terrs)
write_fits("/Users/rakesh/Documents/Monday_meetings/Plots_08.11.2021/FITS/logg_errs_star_new.fits",est_gerrs)
