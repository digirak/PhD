import numpy as np
from scipy.interpolate import interp1d
from vip_hci.fits import open_fits, write_fits
from scipy.signal import savgol_filter
from .removeTelluric import removeTelluric
from .PreProcess import applyFilter
from ._utils import find_nearest
class SynData():
    """
    Class to generate synthetic data. This class will initialize an object to insert a companion orbitting a star
    with their spectra extracted.

    Attributes:
    -----------


    Methods:
    ---------- 

    """
    def __init__(self,companion_file, stellar_file):
        self.companion_file = companion_file
        self.stellar_file = stellar_file

    def insertCompanion(self,
                        R,
                        C,
                        dSNR,
                        wmin_w,
                        wmax_w):
        dat=fits.open(self.stellar_file)[1].data
        teff_s=np.float(self.stellar_file.split('/')[-1].split('-')[0][-4:])*1e2
        logg_s=np.float(self.stellar_file.split('/')[-1].split('-')[1])
        print("Starting star {0:3.2f} K and logg {1:3.2f}".format(teff_s,logg_s))
        
        wmin=find_nearest(dat['Wavelength'],wmin_w)
        wmax=find_nearest(dat['Wavelength'],wmax_w)
        stellar_waves=dat['Wavelength'][wmin:wmax]
        stellar_flux=dat['Flux'][wmin:wmax]
        
        planet_data=fits.open(self.companion_file)[1].data
        teff_c=np.float(self.companion_file.split('/')[-1].split('-')[0][-4:])*1e2
        logg_c=np.float(companion_file.split('/')[-1].split('-')[1])
        wmin_p=find_nearest(planet_data['Wavelength'],wmin_w)

        wmax_p=find_nearest(planet_data['Wavelength'],wmax_w)
        waves_p=planet_data['Wavelength'][wmin_p:wmax_p]
        planet_spec=removeTelluric(waves_p,planet_data['Flux'][wmin_p:wmax_p],1.81,1.93)

        stellar_spec=removeTelluric(stellar_waves,stellar_flux,1.81,1.93)
        
        waves_low=np.arange(wmin_w,wmax_w,1.4/R)

        stellar_spec_resampled=maxMinNorm(np.interp(waves_low,stellar_waves,stellar_spec))*100
        planet_spec_resampled=maxMinNorm(np.interp(waves_low,waves_p,planet_spec))*100
        stellar_noise=np.zeros_like(stellar_spec_resampled)
        planetary_noise=np.zeros_like(stellar_spec_resampled)
        s_spec=stellar_spec_resampled
        st_sum=np.sum(s_spec)
        p_sum=np.sum(planet_spec_resampled)
        multiplier=np.sum(s_spec)*desired_contrast/np.sum(planet_spec_resampled)
        if(C==1):
            sum_spec=multiplier*planet_spec_resampled
        
        else:
            sum_spec=s_spec+multiplier*(planet_spec_resampled)
        print("Inserting {0:3.2f}K {1:3.2f} logg companion at contrast {2:1.0e}"
            .format(teff_c,logg_c,multiplier*(p_sum/st_sum)))
        print("Current resolution is {0:2.2f}".format(waves_low[0]/(waves_low[1]-waves_low[0])))
        
        for i in range(len(waves_low)):
                multiplier_snr_star=0
                if(sum_spec[i]==0):
                    multiplier_snr_star=1
                else:
                    multiplier_snr_star=sum_spec[i]/(np.sqrt(sum_spec[i])*desired_snr)
                stellar_noise[i]=np.random.poisson(np.sqrt(sum_spec[i])*multiplier_snr_star)
                sum_spec[i]=np.sqrt(sum_spec[i]**2+stellar_noise[i]**2)
        sum_spec=np.sqrt(sum_spec**2+stellar_noise**2)
        snr=st_sum/(np.sum(stellar_noise))
        st_sum=np.sum(sum_spec)
        ref_star=medfilt(s_spec/st_sum)
        st_norm=s_spec/(st_sum*ref_star)
        print("SNR is {0:1.0e}".format(snr))
        
        sum_norm=sum_spec/(st_sum*ref_star)
        return sum_norm,waves_low,stellar_noise
    

        
