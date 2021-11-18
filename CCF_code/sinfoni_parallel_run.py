import matplotlib as mpl
from photutils import CircularAperture,aperture_photometry
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np
import sys
import os
from CCFcore.PreProcess import SINFONI
from CCFcore.PreProcess import applyFilter
from CCFcore import removeTelluric
from CCFcore.CrossCorr import CrossCorr
from CCFcore.PreProcess import measureSpatialSpec
from vip_hci.fits import open_fits, write_fits
from CCFcore._utils import find_nearest
import configparser
import multiprocessing as mp
import pandas as pd
if __name__ == '__main__':
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
    fl=config.get("paths","template_path")
    Teff=fl.split('/')[-1].split('-')[0][-4:]
    logg=fl.split('/')[-1].split('-')[1]
    temp_flux,temp_wavs=CC.processTemplate(fl)
    fwhm=open_fits("/Users/rakesh/Data/Final_derot_residual_cubes_HD142527/Rakesh_derot/fwhm_vec_WLcorr")
    n_comps=int(config.get("wavelengthparams",'n_comps'))
    wmin_max=[np.float(config.get("wavelengthparams","wmin")),
            np.float(config.get("wavelengthparams","wmax"))]#[2.2,2.5]
    datapath=config.get("paths","datapath")#"/Users/rakesh/Data/PDS70v2020/"
    pp=(config.get("imageparams",'preprocess',fallback=True))
    window_size=int(config.get("wavelengthparams","window_size",fallback=101))
    order=int(config.get("wavelengthparams","order",fallback=1))
    s=SINFONI(datpath=datapath,filename=config.get("paths","cubename"),
            wavelen=config.get("paths","wavelen_file"),#"good_lambdas_WLcorr.fits",
            fwhm=config.get("paths","fwhm_file"),#"good_fwhm_WLcorr.fits",
            sz=int(config.get("imageparams","crop_size")),
            wmin_max=wmin_max,
            wmin_wmax_tellurics=[np.float(config.get("wavelengthparams","tell_wmin",fallback=1.75)),
            np.float(config.get("wavelengthparams","tell_wmax",fallback=2.15))])

    if(pp=="True"):
        print("Now running preprocess")
        im_pc=s.valentinPreProcessSINFONI(perc=99.5,n_comps=n_comps,window_size=window_size,polyorder=order)
    else:
        print("Using raw data")
        _=s.preProcessSINFONI(n_comps=0,window_size=window_size,polyorder=order)
        loc_low=find_nearest(s.wavelen[0].data,s.wmin_max[0])#np.argmin(abs(s.waves_dat-s.wmin_max[0]))
        loc_high=find_nearest(s.wavelen[0].data,s.wmin_max[1])#np.argmin(abs(s.waves_dat-s.wmin_max[1]))
        #print(s.wavelen[0].data[loc_low],s.wavelen[0].data[loc_high])
        im_pc=s.cube[0].data[loc_low:loc_high]
        fwhm=fwhm[loc_low:loc_high]
    print(im_pc.shape)

    snrmatrix=np.reshape(np.zeros(s.crop_sz*s.crop_sz),(s.crop_sz,s.crop_sz))
    noisemat=np.zeros(snrmatrix.shape)
    prefix=config.get("result","prefix")
    print("We are prefixing %s"%prefix)
    xmin=int(config.get("imageparams","xmin"))
    xmax=int(config.get("imageparams","xmax"))
    ymin=int(config.get("imageparams","ymin"))
    ymax=int(config.get("imageparams","ymax"))
    res_dir=config.get("result","results_dir",fallback="../Results_CCF/")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print("Results will be saved in %s"%res_dir)
    ccfmat=np.zeros(snrmatrix.shape[0]*snrmatrix.shape[1]*vels.shape[0])
    ccfmat=np.reshape(ccfmat,(vels.shape[0],snrmatrix.shape[0],snrmatrix.shape[1]))
   # pool=mp.Pool(8)
    #for xx in range(xmin,xmax):
    #   for yy in range(ymin,ymax):
    #      spec=measureSpatialSpec(im_pc,[xx,yy],fwhm)
    import io
    from contextlib import redirect_stdout
    specs=[]
    import timeit
    starttime = timeit.default_timer()
    pool=mp.Pool(8)
    print("Start the specs")
    print("We're running {} processes".format(pool._processes))
    specs = pool.starmap(measureSpatialSpec,[(im_pc,[xx,yy],fwhm) for xx in range(xmin,xmax) for yy in range(ymin,ymax)])
    pool.close()
    #for xx in range(xmin,xmax):
     #   for yy in range(ymin,ymax):
      #      specs.append(measureSpatialSpec(im_pc,[xx,yy],fwhm))

    #trap = io.StringIO()
    #with redirect_stdout(trap):
            #  ccf_nopc, noise_nopc, snr = CC.compareFluxes(s.waves_dat,
            #                                                 spec,
                #                                                temp_wavs,
                #                                                temp_flux,
                #                                               window_size=window_size,
                #                                              order=order,
                #                                             wmin_wmax_tellurics=s.wmin_wmax_tellurics)
    pool=mp.Pool(8)
    print("Starting parallel")
    trap = io.StringIO()
    with redirect_stdout(trap):
        results = pool.starmap(CC.compareFluxes,[(s.waves_dat,
        spec,
        temp_wavs,
        temp_flux,
        window_size,order,
        0,
        s.wmin_wmax_tellurics) for spec in specs])
    pool.close()
    results=np.asarray(results)
    snrs=np.asarray(results[:,2])
    noises=np.asarray(results[:,1])
    ccfs=np.asarray(results[:,0])

    snrs=pd.DataFrame(snrs)
    snrmatrix=snrs.to_numpy(dtype="float64")
    snrmatrix=snrmatrix.reshape(xmax-xmin,ymax-ymin)
    fname=s.datpath.split('/')[-2]
    frame_size=xmax-xmin
    write_fits(os.path.join(res_dir,"%s_snrmatrix_for_%s_framesize_%d_PCs_%d_wmin_%1.1f_wmax_%1.1f_Teff_%d_logg_%3.2f.fits"
    %(prefix,fname,frame_size,n_comps,wmin_max[0],wmin_max[1],int(Teff),float(logg))),snrmatrix)#ite()
    ccfs=np.reshape(ccfs,(xmax-xmin,ymax-ymin))
    ccfs=pd.DataFrame(ccfs)
    #ccfs=ccfs.to_numpy(dtype="float64")

    for xx in range(xmin,xmax):
        for yy in range(ymin,ymax):
            ccfmat[:,xx,yy]=np.ravel(ccfs.iloc[xx-xmin,yy-ymin])
    



#    np.save(os.path.join(res_dir,"resmatrix_for_parallel.npy"),results)

    #np.save(os.path.join(res_dir,"%s_snrmatrix_for_%s_framesize_%d_PCs_%d_wmin_%1.1f_wmax_%1.1f_Teff_%d_logg_%3.2f.npy"
    write_fits(os.path.join(res_dir,"%s_ccfmatrix_for_%s_framesize_%d_PCs_%d_wmin_%1.1f_wmax_%1.1f_Teff_%d_logg_%3.2f.fits"
    %(prefix,fname,frame_size,n_comps,wmin_max[0],wmin_max[1],int(Teff),float(logg))),ccfmat)
    #np.sav("noisematrix_%d_%1.1f_%1.1f_Teff_%d_logg_%3.2f.npy"%(n_comps,wmin_max[0],wmin_max[1],int(Teff),float(logg)),noisemat)
    print("The time difference is :", timeit.default_timer() - starttime)
