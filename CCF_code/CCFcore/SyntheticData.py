import numpy as np
from scipy.interpolate import interp1d
from vip_hci.fits import open_fits, write_fits
from astropy.io import fits
from scipy.signal import savgol_filter
from .removeTelluric import removeTelluric
from .PreProcess import applyFilter
from ._utils import find_nearest,maxMinNorm
from .CrossCorr import CrossCorr,computeSNRLL
from scipy.signal import medfilt
from contextlib import contextmanager
import sys, os
from scipy.optimize import curve_fit
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
def computeLLSNRACF(vels,waves,spec,temp_wavs,temp_flux,sigma):
    temp_flux_downsampled = np.interp(waves,temp_wavs,temp_flux)
    #other_temp_spec=applyFilter(temp_flux_downsampled,101,1)
    #other_temp_spec=temp_flux_downsampled-temp_flux_downsampled.mean()#applyFilter(temp_flux_downsampled,101,1)
    m_waves = temp_wavs
    m_flux = temp_flux
    CC = CrossCorr(vels)
    with suppress_stdout():
        ccf,_,_ = CC.compareFluxes(waves,
        spec,
        m_waves,
        m_flux,
        101,
        1,
        noise=0,
        wmin_wmax_tellurics=[1.81,1.93]
        )
        acf,_,_ = CC.compareFluxes(waves,
        temp_flux_downsampled,
        m_waves,
        m_flux,
        101,
        1,
        noise=0,
        wmin_wmax_tellurics=[1.81,1.93]
        )
        #sigma= np.nanstd(noise)

    LL,snr = computeSNRLL(ccf,acf,sigma=sigma)
    return LL,snr
def computeErrBarsUsingLL(y,x,x_est,xerr,xind_low=0,xind_high=-1):
    def func_FitLL(x,A,mu,sigma):
        return (A/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sigma**2))
    xmin = np.min(x)
    xmax= np.max(x)
    x_full = np.linspace(xmin,xmax,2000)
    y_vals = np.interp(x_full,x,y)
    p0=[1.0,x_est,xerr]
    try:
        popt,fit_errs=curve_fit(func_FitLL,xdata=x_full[xind_low:xind_high],ydata=y_vals[xind_low:xind_high],
                p0=p0,method='lm',maxfev=100000)
    except:
        popt =[np.nan,np.nan,np.nan]
        fit_errs =np.ones(4).reshape(2,2)*np.nan
    
    return popt,fit_errs



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
        self.log_likes =0.
        self.snrs_char =0.
        self.sum_spec =0.
        self.stellar_noise=0.
        self.noisy =0.

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
        logg_c=np.float(self.companion_file.split('/')[-1].split('-')[1])
        wmin_p=find_nearest(planet_data['Wavelength'],wmin_w)

        wmax_p=find_nearest(planet_data['Wavelength'],wmax_w)
        waves_p=planet_data['Wavelength'][wmin_p:wmax_p]
        #planet_spec=removeTelluric(waves_p,planet_data['Flux'][wmin_p:wmax_p],1.81,1.93)

        #stellar_spec=removeTelluric(stellar_waves,stellar_flux,1.81,1.93)
        
        waves_low=np.arange(wmin_w,wmax_w,wmin_w/R)

        stellar_spec_resampled=(np.interp(waves_low,stellar_waves,stellar_flux))
        planet_spec_resampled=(np.interp(waves_low,waves_p,planet_data['Flux'][wmin_p:wmax_p]))

       # print(sum(stellar_spec_resampled)/sum(planet_spec_resampled))
        stellar_noise=np.zeros_like(stellar_spec_resampled)
        planetary_noise=np.zeros_like(stellar_spec_resampled)
        s_spec=stellar_spec_resampled
        st_sum=np.sum(s_spec)
        s_spec = stellar_spec_resampled/st_sum
        p_sum=np.sum(planet_spec_resampled)
        planet_spec_resampled = planet_spec_resampled/p_sum
        s_spec_new = s_spec*dSNR**2
        planet_spec_resampled_new = C*dSNR**2*planet_spec_resampled
        #print("{0:1.0e} {1:1.0e}".format(np.sum(s_spec_new),np.sum(planet_spec_resampled_new)))
        #print(np.sum(s_spec)/np.sum(planet_spec_resampled))
        #multiplier=np.sum(s_spec)*C/np.sum(planet_spec_resampled)
        if(C==1):
            print("Attention! il n'y a pas d'etoile ici!")
           # print(multiplier)
        
        self.sum_spec = (1-C)*s_spec_new+planet_spec_resampled_new
#        print("Total flux is {0:1.0e}".format(np.sum(self.sum_spec)))

        print("Inserting {0:3.2f}K {1:3.2f} logg companion at contrast {2:1.0e}"
            .format(teff_c,logg_c,C))
        print("Current resolution is {0:1.0e}".format(waves_low[0]/(waves_low[1]-waves_low[0])))
        self.noisy = np.zeros_like(self.sum_spec)
        #snrs= np.zeros_like(waves_low)
        for i in range(len(waves_low)):
                 self.noisy[i] = np.random.poisson(self.sum_spec[i])


       # print("{0:1.0e}".format(np.std(noise_spec-self.sum_spec)))
        self.stellar_noise = (self.noisy)  - (self.sum_spec)
    
        snr= np.mean(self.sum_spec)/(np.std(self.stellar_noise))
        print("SNR_lambda ={0:1.0e}".format(snr))
        snr_spec = np.sum(self.sum_spec)/np.sqrt((np.sum((self.noisy -self.sum_spec)**2)))
        print("SNR = {0:1.0e}".format(snr_spec))
        ref_star=medfilt(s_spec_new/dSNR**2,kernel_size=1)
        #ref_star = ref_star#/np.sum(ref_star)
        #ref_star=maxMinNorm(ref_star)
        sum_norm =self.noisy/(ref_star)
        #med_noise = medfilt(self.stellar_noise)
        #self.stellar_noise = (noise_spec)  - (self.sum_spec)
        sum_norm =sum_norm*dSNR**2/np.sum(sum_norm)
        #self.stellar_noise = self.stellar_noise*dSNR/np.sum(self.stellar_noise)

        print("PP SNR_lambda ={0:1.0e}".format(np.mean(sum_norm)/np.std(self.stellar_noise)))
        print("PP SNR ={0:1.0e}".format(np.sum(sum_norm)/np.sqrt(np.sum(self.stellar_noise**2))))
        #normalized_noise = self.stellar_noise/ (med_noise*np.nansum(self.stellar_noise))

       #print("Sum spec std is {0:1.0e} noisy std is {1:1.0e}".format(np.std(self.sum_spec),np.std(noise_spec)))
        #print("spec std is {0:1.0e}, noise std is {1:1.0e}".format(np.std(sum_norm),np.nanstd(self.stellar_noise[::])))
        return sum_norm,waves_low,self.stellar_noise,ref_star
    def computeDetectionMatrix(self,wmin,wmax,Cmin,Cmax,Cnum,dSNRmin,dSNRmax,dSNRnum):
        """
        Method to compute the detection matrix

        """
        Cs=np.logspace(Cmin,Cmax,Cnum)
        dSNRs=np.logspace(dSNRmin,dSNRmax,dSNRnum)
        Rnum =30
        Rs = np.logspace(np.log10(8e3),np.log10(1e5),Rnum)

        retMatrix=np.zeros(Rnum*dSNRnum).reshape(dSNRnum,Rnum)
        snr = 0.
        vels = np.arange(-1,1,1)
        CC = CrossCorr(vels)
        for R in range(len(Rs)):
            for dsnr in range(len(dSNRs)):
            #print("")
                snr = 0.
                count=0
                while(snr < 6.0):
                    with suppress_stdout():
                        spec, waves, noise,_ = self.insertCompanion(Rs[R],Cs[count],dSNRs[dsnr],wmin,wmax)
                    m_waves = np.asarray(fits.open(self.companion_file)[1].data['Wavelength'])
                    m_flux = np.asarray(fits.open(self.companion_file)[1].data['Flux'])
                    flux_comp = np.interp(waves,m_waves,m_flux)
                    #flux_comp = removeTelluric(waves,flux_comp,1.81,1.93)
                    #flux_comp = flux_comp*dSNRs[dsnr]**2/np.sum(flux_comp)
                    snr =0.
                    #spec = spec*dSNRs[dsnr]**2
                    true_noise = self.noisy -self.sum_spec
                    sigma= np.std(true_noise)#np.sqrt(np.sum(spec))
                    with suppress_stdout():
                        ccf,_,_ = CC.compareFluxes(waves,
                        spec,
                        m_waves,
                        m_flux,
                        501,
                        1,
                        noise=0,
                        wmin_wmax_tellurics=[1.81,1.93]
                        )
                        acf,_,_ = CC.compareFluxes(waves,
                        flux_comp,
                        m_waves,
                        m_flux,
                        501,
                        1,
                        noise=0,
                        wmin_wmax_tellurics=[1.81,1.93]
                        )
            
                        #n1 = np.nansum(noise)
                        #n2 = medfilt(noise/n1)
                        #n = noise/(n1*n2)
                        #if(np.isnan(sigma)):
                         #   sigma = 10

                    LL,snr = computeSNRLL(ccf,acf,sigma=sigma)
                    zero = find_nearest(vels,0)
                    snr = snr[zero]
                    if(snr>=6.0):
                        count=count
                        C=Cs[count]
                        retMatrix[dsnr,R]=C
                        print("For a contrast of {0:1.0e}, and a dSNR of {1:1.0e} I detect at {2:1.0e} with an SNR of {3:3.2f}".format(C,dSNRs[dsnr],
                        Rs[R], snr))
                        print("snr ={0:3.5f},ccf={1:1.0e},sqrt(acf)= {2:1.0e}, sigma = {3:1.0e}".format(snr,ccf[zero],np.sqrt(acf[zero]),sigma))
                    else:
                        count = count+1
                    #print(count)
                    if count==Cnum:
                        retMatrix[dsnr,R] = np.nan
                        print("No detection at {0:1.0e} and snr {1:1.0e}".format(Rs[R],dSNRs[dsnr]))
                        break
        return retMatrix,dSNRs,Rs
    def constrainTeffLogg(self,template_files,R,C,dSNR,wmin,wmax):
                
        Teff=[]
        logg=[]
        template_files.sort()
        ccfs=[]
        snrs=[]
        i=0
        for fl in template_files:
            Teff.append(fl.split('/')[-1].split('-')[0][-4:])
            logg.append(fl.split('/')[-1].split('-')[1])
        Teff_comp = np.float(self.companion_file.split('/')[-1].split('-')[0][-4:])
        logg_comp = np.float(self.companion_file.split('/')[-1].split('-')[1])
        teff_labs=np.unique(Teff)
        logg_labs=np.unique(logg)
        snrs=np.zeros(teff_labs.size*logg_labs.size).reshape(teff_labs.size,logg_labs.size)
        log_likes=np.zeros_like(snrs)
        vels = np.arange(-1,1,1)
        CC=CrossCorr(vels)
        spec, waves, n,_ = self.insertCompanion(R,C,dSNR,wmin,wmax)
        #n1 = np.nansum(n)
        #n2 = medfilt(noise/n1)
        #noise= n/(n1*n2)
        sigma= np.nanstd(n)
        print("Sigma = {0:1.0e}".format(sigma))
        for fl in template_files:
           # print(fl)
            m_waves = np.asarray(fits.open(self.companion_file)[1].data['Wavelength'])
            m_flux = np.asarray(fits.open(self.companion_file)[1].data['Flux'])
            temp=fits.open(fl)[1].data
            teff=fl.split('/')[-1].split('-')[0][-4:]
            logg=fl.split('/')[-1].split('-')[1]
            temp_wavs=temp['Wavelength']
            temp_flux=temp['Flux']
            loc=[np.ravel(np.where(teff==teff_labs))[0],np.ravel(np.where(logg==logg_labs))[0]]
            LL,snr = computeLLSNRACF(vels,waves,spec,temp_wavs,temp_flux,sigma)
            zero = find_nearest(vels,0)
            snrs[loc[0],loc[1]] = snr[zero]
            log_likes[loc[0],loc[1]] = LL[zero]
        self.log_likes = log_likes
        self.snrs_char =snrs
       # opts=[]
        #teff_labs=np.asarray([np.float(l)*100 for l in teff_labs])
        #logg_labs = np.asarray([np.float(l) for l in logg_labs])
        #teff_locs =[]
        #for i in range(len(teff_labs)):
         #   if(teff_labs[i]%100==0):
          #       teff_locs.append(i)
        #fit_errs=[]
        #for i in range(len(logg_labs)):
            #computing Teff_err
         #   y = -maxMinNorm(-log_likes[teff_locs,i])
          #  opt,fit_err = computeErrBarsUsingLL(y,teff_labs[teff_locs],Teff_comp*100,100)
           # opts.append(opt)

            #fit_errs.append(np.linalg.det(fit_err))
        #print(opts)

        #positives = np.asarray(opts)[np.asarray(opts)[:,1]>0]
        #positives = np.asarray(positives)[np.asarray(positives)[:,-1]>0]
       # best = opts[np.nanargmin(fit_errs)]#positives[np.nanargmin((np.asarray(positives)[:,-1]))]   
        #Teff_est=best[1]
        #if(best[1]>2000):
         #   teff_est =np.nan
        #Teff_err = best[2]
        #opts=[]
        #fit_errs=[]
        #teff_labs=[teff_labs[t] for t in teff_locs]
        #for i in range(len(teff_locs)):
            #computing logg_err
         #   y = -maxMinNorm(-log_likes[teff_locs[i],:])
          #  opt,fit_err = computeErrBarsUsingLL(y,logg_labs,logg_comp,0.8)
           # opts.append(opt)
            #if(np.isnan(opts[0][0])):
             #   break
            #fit_errs.append(np.linalg.det(fit_err))
        #rint(opts,teff_labs[teff_locs])
        #positives = np.asarray(opts)[np.asarray(opts)[:,-1]>0]
        #best = opts[np.nanargmin(fit_errs)]#positives[np.nanargmin((np.asarray(positives)[:,-1]))]   
        #logg_est=best[1]
        #logg_err = best[2]
        #return (Teff_est,Teff_err,logg_est,logg_err,log_likes,snrs)
        return (log_likes,snrs,spec,sigma)


        







