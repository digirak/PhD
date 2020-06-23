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
#from parallelCompareTemplates import CrossCorr
import pandas as pd
from removeTelluric import removeTelluric
import glob
from vip_hci.pca.svd import svd_wrapper
from _utils import find_nearest

font = {'family' : 'serif',
                'weight' : 'bold'}

mpl.rc('font', **font)
def measureSpatialSpec(datcube,loc,fwhm):
    slices=[]
    for wl in range(datcube.shape[0]):
         apertures=CircularAperture([loc[0],loc[1]],r=fwhm[wl]/2)
    #    #print(aperture_p)
         slices.append(np.float64(aperture_photometry(datcube[wl,:,:],apertures)['aperture_sum']))
    return np.asarray(slices)
def applyFilter(flux,window_size,order):
    flux_filt=savgol_filter(flux,window_size,order)
    flux_dat=flux-flux_filt#-vals(waves)
    return flux_dat
class SINFONI:
    def __init__(self,*args,**kwargs):
        self.datpath=kwargs.get('datpath',
                                "/mnt/disk4tb/Users/rnath/Data/HD142527/")
        self.filename=kwargs.get('filename',
                                  'ASDI_cube_derot_med.fits')
        self.cube=fits.open(self.datpath+self.filename)
        self.fwhm=fits.open(self.datpath+kwargs.get('fwhm',"fwhm_vec.fits"))
        self.wavelen=fits.open(self.datpath+kwargs.get('wavelen',"lbda_vec.fits"))
        self.vels=kwargs.get('vels',np.arange(-2000,2000,20))
        self.wmin_max=kwargs.get('wmin_max',[1.47,1.6])
        self.crop_sz=np.int(kwargs.get('sz',71))
        self.cube[0].data=vip_hci.preproc.cube_crop_frames(
        self.cube[0].data,self.crop_sz)
        self.normalized_cube=0
        self.waves_dat=0
        self.fwhm_final=0
        self.sum_spax=0.
        self.wmin_wmax_tellurics=[1.8,2.1]


    def preProcessSINFONI(self,*args,**kwargs):
        idx_wmin = find_nearest(self.wavelen[0].data,self.wmin_max[0])
        idx_wmax = find_nearest(self.wavelen[0].data,self.wmin_max[1])
        print("We are reading %s"%(self.datpath+self.filename))
        cube_whole = self.cube[0].data[idx_wmin:idx_wmax,:,:]
        wavelen = self.wavelen[0].data[idx_wmin:idx_wmax]
        self.waves_dat=wavelen
        fwhm = self.fwhm[0].data[idx_wmin:idx_wmax]
        self.fwhm_final=fwhm
        #sum the spaxels
        cube=np.zeros(cube_whole.shape)
        for xx in range(cube_whole.shape[1]):
            for yy in range(cube_whole.shape[2]):
                cube[:,xx,yy]=np.asarray(removeTelluric(wavelen,np.ravel(cube_whole[:,xx,yy]),
                                              wmin=self.wmin_wmax_tellurics[0],
                                              wmax=self.wmin_wmax_tellurics[1]))
        center=int(self.crop_sz/2)
        self.sum_spax=measureSpatialSpec(cube[:,:,:],[center,center],self.fwhm_final)
        #for wl in range(idx_wmax-idx_wmin):
         #   sum_spax.append(np.nansum(cube[wl,30:40,30:40]))
        print("Wavelength spans from %2.1f to %2.1fmum"%(wavelen.min(),wavelen.max()))
        # divide by this sum
        #for spax in range(len(sum_spax)):
         #   cube[spax,:,:]=cube[spax,:,:]/sum_spax[spax]
        # create a reference spectrum by taking median
        #removed
        #two steps now, divide the whole cube by the ref_spec and also filter
        #for lower order residuals using savgol filter. option to subtract the reference also.
        polyorder=kwargs.get('polyorder',1)
        window_size=kwargs.get('window_size',101)
        self.normalized_cube=np.zeros(cube.shape)
        filt=np.zeros(cube.shape)
        norm_cube=np.zeros(cube.shape)
        for xx in range(cube.shape[1]):
            for yy in range(cube.shape[2]):
                    norm_cube[:,xx,yy]=cube[:,xx,yy]/self.sum_spax
                    #filt[:,xx,yy]=savgol_filter(norm_cube[:,xx,yy]
                    #    ,window_size
                    #    ,polyorder=polyorder)
                    self.normalized_cube[:,xx,yy]=applyFilter(norm_cube[:,xx,yy],
                    window_size,polyorder)
        #self.normalized_cube[np.where(filt!=0)]=norm_cube[np.where(filt!=0)]-filt[np.where(filt!=0)]
        recon_3dcube=np.zeros_like(self.normalized_cube)
        ncomp=kwargs.get("n_comps",2)
        if(ncomp==0):
            recon_3dcube=self.normalized_cube
            print("no PCA ")
        else:
            PCA_matrix=np.reshape(np.zeros(len(wavelen)*cube.shape[1]*cube.shape[2]),
                    (len(wavelen),cube.shape[1]*cube.shape[2]))

            for wl in range(len(wavelen)):
                for xx in range(cube.shape[1]):
                    for yy in range(cube.shape[2]):
                        PCA_matrix[wl,xx*cube.shape[1]+yy]=self.normalized_cube[wl,xx,yy]
            V=svd_wrapper(PCA_matrix,'lapack',ncomp,verbose=True)
            transformed=np.dot(V,PCA_matrix.T)
            reconstructed=np.dot(transformed.T,V)
            residuals=PCA_matrix-reconstructed

            #re-wrap into a cube
            for wl in range(idx_wmax-idx_wmin):
                for xx in range(cube.shape[1]):
                    for yy in range(cube.shape[2]):
                        recon_3dcube[wl,xx,yy]=residuals[wl,cube.shape[1]*xx+yy]
        #waves=np.load("/mnt/diskss/home/rnath/Numpy_PDS70/wavelens_new28.npy")
        #locs=np.where(self.wavelen[0].data>=0)

        return recon_3dcube
    def preProcessAltPCASINFONI(self,*args,**kwargs):
        idx_wmin = find_nearest(self.wavelen[0].data,self.wmin_max[0])
        idx_wmax = find_nearest(self.wavelen[0].data,self.wmin_max[1])
        print("We are reading %s"%(self.datpath+self.filename))
        cube_whole = self.cube[0].data[idx_wmin:idx_wmax,:,:]
        wavelen = self.wavelen[0].data[idx_wmin:idx_wmax]
        self.waves_dat=wavelen
        fwhm = self.fwhm[0].data[idx_wmin:idx_wmax]
        self.fwhm_final=fwhm
        #sum the spaxels
        cube=np.zeros(cube_whole.shape)
        for xx in range(cube_whole.shape[1]):
            for yy in range(cube_whole.shape[2]):
                cube[:,xx,yy]=np.asarray(removeTelluric(wavelen,np.ravel(cube_whole[:,xx,yy]),
                                              wmin=self.wmin_wmax_tellurics[0],
                                              wmax=self.wmin_wmax_tellurics[1]))
        center=int(self.crop_sz/2)
        self.sum_spax=measureSpatialSpec(cube[:,:,:],[center,center],self.fwhm_final)
        #for wl in range(idx_wmax-idx_wmin):
         #   sum_spax.append(np.nansum(cube[wl,30:40,30:40]))
        print("Wavelength spans from %2.1f to %2.1fmum"%(wavelen.min(),wavelen.max()))
        # divide by this sum
        #for spax in range(len(sum_spax)):
         #   cube[spax,:,:]=cube[spax,:,:]/sum_spax[spax]
        # create a reference spectrum by taking median
        #removed
        #two steps now, divide the whole cube by the ref_spec and also filter
        #for lower order residuals using savgol filter. option to subtract the reference also.
        polyorder=kwargs.get('polyorder',1)
        window_size=kwargs.get('window_size',101)
        self.normalized_cube=np.zeros(cube.shape)
        filt=np.zeros(cube.shape)
        norm_cube=np.zeros(cube.shape)
        for xx in range(cube.shape[1]):
            for yy in range(cube.shape[2]):
                    norm_cube[:,xx,yy]=cube[:,xx,yy]/self.sum_spax
                    #filt[:,xx,yy]=savgol_filter(norm_cube[:,xx,yy]
                    #    ,window_size
                    #    ,polyorder=polyorder)
                    self.normalized_cube[:,xx,yy]=applyFilter(norm_cube[:,xx,yy],
                    window_size,polyorder)
        #self.normalized_cube[np.where(filt!=0)]=norm_cube[np.where(filt!=0)]-filt[np.where(filt!=0)]
        recon_3dcube=np.zeros_like(self.normalized_cube)
        ncomp=kwargs.get("n_comps",2)
        if(ncomp==0):
            recon_3dcube=self.normalized_cube
            print("no PCA ")
        else:
            PCA_matrix=np.reshape(np.zeros(len(wavelen)*cube.shape[1]*cube.shape[2]),
                    (len(wavelen),cube.shape[1]*cube.shape[2]))

            for wl in range(len(wavelen)):
                for xx in range(cube.shape[1]):
                    for yy in range(cube.shape[2]):
                        PCA_matrix[wl,xx*cube.shape[1]+yy]=self.normalized_cube[wl,xx,yy]
            V=svd_wrapper(PCA_matrix.T,'lapack',ncomp,verbose=True)
            transformed=np.dot(V,PCA_matrix)
            reconstructed=np.dot(transformed.T,V)
            residuals=PCA_matrix-reconstructed.T

            #re-wrap into a cube
            for wl in range(idx_wmax-idx_wmin):
                for xx in range(cube.shape[1]):
                    for yy in range(cube.shape[2]):
                        recon_3dcube[wl,xx,yy]=residuals[wl,cube.shape[1]*xx+yy]
        #waves=np.load("/mnt/diskss/home/rnath/Numpy_PDS70/wavelens_new28.npy")
        #locs=np.where(self.wavelen[0].data>=0)

        return recon_3dcube


    def valentinPreProcessSINFONI(self,*args,**kwargs):
        idx_wmin = find_nearest(self.wavelen[0].data,self.wmin_max[0])
        idx_wmax = find_nearest(self.wavelen[0].data,self.wmin_max[1])
        cube = self.cube[0].data[idx_wmin:idx_wmax,:,:]
        wavelen = self.wavelen[0].data[idx_wmin:idx_wmax]
        self.waves_dat=wavelen
        fwhm = self.fwhm[0].data[idx_wmin:idx_wmax]
        cube_nt=np.zeros(cube.shape)
        for xx in range(cube.shape[1]):
            for yy in range(cube.shape[2]):
                cube_nt[:,xx,yy]=removeTelluric(wavelen,cube[:,xx,yy],wmin=self.wmin_wmax_tellurics[0],wmax=self.wmin_wmax_tellurics[1])
                #sum the spaxels
        med_frame= np.median(cube_nt, axis=0)
        perc=0.95

        thr = np.percentile(med_frame,perc)
        idx_high = np.where(med_frame>thr)

        print(len(idx_high[0]))


        # take the spectrum of the star at the location of the brightest pixels
        spec_high= []
        for wl in range(len(wavelen)):
            spec_high.append(cube_nt[wl][idx_high])
        spec_high = np.array(spec_high)
        norm_specs_high = np.ones_like(spec_high)
        sum_spax_high= np.sum(spec_high,axis=0)
        # normalized spectra for each of the top percentile spaxels:
        for wl in range(len(wavelen)):
            norm_specs_high[wl] = spec_high[wl]/sum_spax_high

        # averaging spatially among the perc% brightest pixels:
        ref_spec=[]
        for wl in range(len(wavelen)):
            ref_spec.append(np.median(norm_specs_high[wl]))
        sum_spax_high= np.sum(spec_high,axis=0)
        norm_cube = cube_nt.copy()
        spat_sum = np.sum(cube_nt,axis=0)

        for zz in range(cube_nt.shape[0]):
            norm_cube[zz]=cube_nt[zz]/(spat_sum*ref_spec[zz])
        res_cube = np.zeros_like(norm_cube)
        norm_res_cube = np.zeros_like(norm_cube)
        filt = np.zeros_like(norm_cube)
        polyorder=kwargs.get('polyorder',1)
        window_size=kwargs.get('window_size',101)
        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                filt[:,i,j]=savgol_filter(norm_cube[:,i,j],window_size,polyorder=polyorder)

        norm_res_cube = norm_cube-filt
        for wl in range(cube.shape[0]):
            good= np.where(filt[wl]!=0)
            res_cube[wl][good] = cube_nt[wl][good] - (filt[wl][good]*spat_sum[good]*ref_spec[wl])
        new_matrix=np.zeros([wavelen.shape[0],cube.shape[1]*cube.shape[2]])
        counter=0
        ncomp=int(kwargs.get("n_comps",2))
        print("Subtract %d comps"%ncomp)
        if(ncomp==0):
            #recon_3dcube=res_cube
            print("no PCA ")
            return res_cube

        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                if res_cube[0,j,k] != 0:
                    for i in range(wavelen.shape[0]):
                        new_matrix[i,j*cube.shape[1]+k]=res_cube[i,j,k]
                    counter+=1
        new_matrix=np.asanyarray(pd.DataFrame(new_matrix).fillna(0))
        new_matrix = new_matrix[:,:]
        V = svd_wrapper(new_matrix,ncomp=ncomp, mode='randsvd',verbose=True)
        transformed = np.dot(V, new_matrix.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = new_matrix - reconstructed
        resi_3dcube = np.zeros_like(res_cube)

        new_counter = 0

        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                if res_cube[0,j,k] != 0:
                    for i in range(wavelen.shape[0]):
                        resi_3dcube[i,j,k] = residuals[i,new_counter]
                    new_counter+=1
        return resi_3dcube
