__author__='Rakesh Nath'
import numpy as np
from astropy.io import fits
from scipy.constants import c
from glob import glob
from scipy.interpolate import interp1d
import sys
import os
import json
from photutils import CircularAperture,aperture_photometry
import vip_hci
from scipy.signal import savgol_filter
from .removeTelluric import removeTelluric
from .PreProcess import applyFilter
from ._utils import find_nearest
#from pycorrelate import pcorrelate,ucorrelate
from astropy.convolution import Gaussian1DKernel, convolve_fft
from astropy.stats import gaussian_fwhm_to_sigma
import warnings
from matplotlib import pyplot as plt
warnings.simplefilter('ignore', np.RankWarning)
class CrossCorr:
    """
    This is a class for computing cross correlations

    Attributes
    ----------
    vels (array):
        Velocity array should be greater 1000 km/s
    Methods
    --------
    processTemplate(temp_file,instru_fwhm_nm):
        Downsamples and broadens the template according to instru_fwhm in microm
    compareFluxes(data_wavs, data_flux, model_wavs, model_flux,window_size, order, noise(optional),wmin_wmax_tellurics(optional)):
        Performs cross correlations based on data_flux and model_flux for finite wavelengths defined by data_waves over vels
    """
    def __init__(self,vels):
        self.vels=vels
        self.f1=0
        self.f2=0
    def processTemplate(self,temp_file,instru_fwhm_nm=4.998446e-04):
        """
        Performs broadening of the template based on instrumental fwhm.

        Parameters
        ----------
        temp_file (string):
            Full absolute path of the template file 
        instru_fwhm_nm (float):
            The instrument wavelength resolution (in same units as temp_file)
        
        Returns
        --------
            float, float:
                Returns wavelengths from the temp_file and flux now broadened
        """
        print("Now reading {}".format(str(temp_file)))
        #f=str(f)
        temp=fits.open(temp_file)
        wave_temp=temp[1].data['Wavelength']
        flux=temp[1].data['flux']
        #flux=(flux-np.min(flux))/(np.max(flux)-np.min(flux))
        Teff=temp_file.split('/')[-1].split('-')[0][-4:]
        logg=temp_file.split('/')[-1].split('-')[1]
        instru_fwhm_nm = instru_fwhm_nm#4.998446e-04#0.67 #mum

        BT_SETTL_res = wave_temp[1]-wave_temp[0]#0.005   #mum
        instru_fwhm_BTSETTL = instru_fwhm_nm/BT_SETTL_res
        gaus_BTSETTL = Gaussian1DKernel(stddev=instru_fwhm_BTSETTL*gaussian_fwhm_to_sigma)

        kernel = np.array(gaus_BTSETTL)
        temp_conv = np.convolve(flux,kernel,'same') / sum(kernel)
        #filt=savgol_filter(temp_conv,101,polyorder=3)
        return temp_conv,wave_temp
    def compareFluxes(self
                     ,data_wavs
                     ,data_flux
                     ,model_wavs
                     ,model_flux
                     ,window_size
                     ,order
                     ,noise=0
                     ,wmin_wmax_tellurics=0):
        """
        Performs crosscorrelations between data and model fluxes.

        Parameters
        ----------
        data_wavs (float array) :
            wavelengths of the data 
        data_flux (float array) :
            data flux that needs to be correlated
        model_wavs (float array):
            wavelengths of the model
        model_flux (float array):
            Model flux to compare data against
        window_size (int):
            The size of the Savitzky-Golay filter
        order (int):
            The polynomial order of the filter
        noise (float):
            The noise standard deviation to compare the peak of CCF with
        wmin_max_tellurics (list):
            The wavelength of tellurics to mask out
        
        Returns
        -------
        array, float, float:
            cross correlation coefficients, noise floor, SNR
        """
        vels=self.vels
        df=vels[1]-vels[0]
        dataflux=data_flux
        final=np.zeros(len(vels))
        self.f1=dataflux-dataflux.mean()
        if wmin_wmax_tellurics==0:
            wmin_wmax_tellurics=[1.75,2.05]
        print(wmin_wmax_tellurics)

        #self.f1=removeTelluric(data_wavs,self.f1,wmin_wmax_tellurics[0],wmin_wmax_tellurics[1])

        for i in range(len(final)):
            #flux=np.convolve(flux,fwhm[0].data)
            inter=interp1d(model_wavs*(1+vels[i]/3e5),model_flux)
            temp=inter(data_wavs)
            temp_filt=applyFilter(temp
                                 ,window_size=window_size
                                 ,order=order)
            #filt=savgol_filter(temp,101,1)
            #temp_filt=temp-filt
            temp_filt=removeTelluric(data_wavs,temp_filt,wmin_wmax_tellurics[0],wmin_wmax_tellurics[1])

            self.f2=temp_filt-temp_filt.mean()
            ccov = np.correlate(self.f1,self.f2,mode='valid')
            #ccov=np.corrcoef(f1,f2)[0][1]
            cf = ccov / (self.f1.std()*self.f2.std())
            final[i]=(cf[0])
        neg_edge=find_nearest(vels,-500)
        pos_edge=find_nearest(vels,500)
        max_val=np.max((final[neg_edge:pos_edge]))

       # max_vel=vels[np.argmax((final[neg_edge:pos_edge]))]
        #print(final[max_val])

        if (noise==0):


            locs_noise_l=[(np.where(((vels>-2000)) & (vels<-1000)))]
            locs_noise_h=[(np.where(((vels>1000)) & (vels<2000)))]
            noise_floor=np.sqrt(np.std(final[locs_noise_l])**2\
            +np.std(final[locs_noise_h])**2)
        else:
            noise_floor=noise

        #print("SNR is %3.2f"%(np.max(final)/noise_floor))
        if(np.max(vels)<1000):
            print("Use Ruffio SNR")
        else:
            print("SNR is %3.2f"%(max_val/noise_floor))

        return final,noise_floor,((max_val/noise_floor))

#if __name__ == '__main__':
 #   from multiprocessing import Pool
  #  files=glob("/mnt/disk4tb/Users/sabyasachi/BT_Settl/*")
   # x=(int(sys.argv[1]))
   # y=(int(sys.argv[2]))
    #T=Template(x,y)


    #p = Pool(5)
    #crosscorr_dict = p.map( T.compareTemplate, (item for item in files[0:50]) )
    #print(crosscorr_dict)
    #np.save("crosscorr_%d.%d.npy"%(x,y),crosscorr_dict)
def computeSNRLL(ccf,acf,sigma):
    """
    This method computes the loglikelihood and SNR based on Ruffio et al 2019

    Parameters:
    ------------
    ccf: Cross correlation of the template with the target spectrum
    acf: Auto correlation of the template with itself
    sigma: Variance of the noise that is inserted

    Returns:
    -----------
    float,float:
    Returns the log likelihood and SNR
    """
    log_like = -(ccf/sigma**2)**2/(acf/sigma**2)
    snr = (ccf/sigma**2)/np.sqrt(acf/sigma**2)
    return log_like, snr