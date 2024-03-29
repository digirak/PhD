__author__='Rakesh Nath'
__all__=['measureSpatialSpec',
          'applyFilter']
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from photutils import CircularAperture,aperture_photometry
import vip_hci
from scipy.signal import savgol_filter
try:
    from vip_hci import pca# for the latest vip
except ImportError:
    from vip_hci.psfsub import pca_local
try:
    from sklearn.decomposition import pca
except ImportError:
    from sklearn.decomposition import PCA
import matplotlib as mpl
#from parallelCompareTemplates import CrossCorr
import pandas as pd
import glob
try:
    from vip_hci.pca.svd import svd_wrapper
except ImportError:
    from vip_hci.psfsub.svd import svd_wrapper
from vip_hci.var.shapes import frame_center
from ._utils import find_nearest
from .removeTelluric import removeTelluric

font = {'family' : 'serif',
                'weight' : 'bold'}

mpl.rc('font', **font)
def measureSpatialSpec(datcube,loc,fwhm):
    """
    This is a function to measure the the spectrum spatially

    Parameters
    ----------
    datcube : 3d asarray
        The 3d data cube which has the format wavelength,xx,yy
    loc : list
        location at which the spectrum needs to be measured in [yy,xx]
    fwhm : list or array
        List of full-width half max

    Returns
    -------
    array :
        Measured spectrum

    """
    slices=[]
    for wl in range(datcube.shape[0]):
         apertures=CircularAperture([loc[0],loc[1]],r=fwhm[wl]/2)
    #    #print(aperture_p)
         slices.append(np.float64(aperture_photometry(datcube[wl,:,:],apertures)['aperture_sum']))
    return np.asarray(slices)
def applyFilter(flux,window_size,order):
    """
    Function  to apply the Savitzky-Golay filter and subtract the filtered
    spectrum from the target.

    Parameters
    ----------
    flux : array
        Spectrum to which this needs to be applied
    window_size: int
        The size of the savgol filter in bins
    order : int
        The polynomial order of the filter


    Returns
    -------
    array :
        The spectrum after subtracting the savgol filter from flux
    """
    flux_filt=savgol_filter(flux,window_size,order)
    flux_dat=flux-flux_filt
    return flux_dat
class SINFONI:
    """
    This is a class that defines data process for a SINFONI like instrument

    Attributes
    -----------
        datpath : str
            location where your data resides.

        filename : str
            filename of the data file
        fwhm : str
            filename of the FWHM file
        wavelen : str
            filename of the wavelength solution

        vels : array
            velocity array should have at least upto 1100 km/s

        wmin_max : list
                The min max frequencies that need to be set up for this
        sz : int
                The size to which the cube needs to be cropped down to. Inherits VIP crop_sz
    Methods
    -------
       preProcessSINFONI(polyorder,window_size,n_comps)
            Does the pre processing with basic processing. Subtracting stellar spectra
            and returning a cube after PC is subtracted if n_comps>0
       valentinPreProcess(perc,polyorder,window_size,n_comps)
            Does pre process as per Haffert et al with more accurate stellar modeling. Returns
            cube as before
    """
    def __init__(self,*args,**kwargs):
        self.datpath=kwargs.get('datpath',
                                "/mnt/disk4tb/Users/rnath/Data/HD142527/")
        self.filename=kwargs.get('filename')
        if self.filename == None:
            self.cube=np.asarray(kwargs.get('cube'))
        else:
            self.cube= fits.open(self.datpath+self.filename)[0].data
        self.fwhm=fits.open(self.datpath+kwargs.get('fwhm',"fwhm_vec.fits"))[0].data
        self.wavelen=fits.open(self.datpath+kwargs.get('wavelen',"lbda_vec.fits"))[0].data
        self.vels=kwargs.get('vels',np.arange(-2000,2000,20))
        self.wmin_max=kwargs.get('wmin_max',[1.47,1.6])
        self.crop_sz=np.int(kwargs.get('sz',71))
        self.cube=vip_hci.preproc.cube_crop_frames(
        self.cube,self.crop_sz)
        self.normalized_cube=0
        self.waves_dat=0
        self.fwhm_final=0
        self.sum_spax=0.
        self.wmin_wmax_tellurics=kwargs.get("wmin_wmax_tellurics",[1.75,2.1])
        self.V=0
        self.spec=0

    def preProcessSINFONI(self,*args,**kwargs):
        """
        Performs some basic stellar subtraction on the cube. if n_comps>0
        performs the PCA

        Parameters
        ----------
        polyorder (int optional, default = 1):
                Order of the Savitzky-Golay filter. See applyFilter.
        window_size (int option, default =101):
                Window size has to be odd
        n_comps (int, default=2):
                number of components to be subtractd.

        Returns
        -------
        float :
            Returns a 3D matrix formatted (wl,RA,DEC)
        """


        idx_wmin = find_nearest(self.wavelen,self.wmin_max[0])
        idx_wmax = find_nearest(self.wavelen,self.wmin_max[1])
        if(self.filename==None):
            print("Cube is passed as parameter")
        else:
            print("We are reading %s"%(self.datpath+self.filename))
        cube_whole = self.cube[idx_wmin:idx_wmax,:,:]
        wavelen = self.wavelen[idx_wmin:idx_wmax]
        self.waves_dat=wavelen
        fwhm = self.fwhm[idx_wmin:idx_wmax]
        self.fwhm_final=fwhm
        #sum the spaxels
        cube_notell=np.zeros_like(cube_whole)
        for xx in range(cube_notell.shape[1]):
            for yy in range(cube_notell.shape[2]):
                self.spec=(cube_whole[:,xx,yy])
                self.spec=removeTelluric(wavelen,self.spec,self.wmin_wmax_tellurics[0],self.wmin_wmax_tellurics[1])
                cube_notell[:,xx,yy]=self.spec
                #cube_notell[:,xx,yy]=removeTelluric(wavelen,cube_whole[:,xx,yy],self.wmin_wmax_tellurics[0],self.wmin_wmax_tellurics[1])
        #center=int(self.crop_sz/2)
        center=frame_center(cube_notell)
        self.sum_spax=measureSpatialSpec(cube_notell,center,self.fwhm_final)
        print("Wavelength spans from %2.1f to %2.1fmum"%(wavelen.min(),wavelen.max()))

        #two steps now, divide the whole cube by the ref_spec and also filter
        #for lower order residuals using savgol filter. option to subtract the reference also.
        polyorder=kwargs.get('polyorder',1)
        window_size=kwargs.get('window_size',101)
        self.normalized_cube=np.zeros(cube_notell.shape)
        filt=np.zeros(cube_notell.shape)
        norm_cube=np.zeros(cube_notell.shape)
        for xx in range(cube_notell.shape[1]):
            for yy in range(cube_notell.shape[2]):
                    norm_cube[:,xx,yy]=cube_notell[:,xx,yy]/self.sum_spax
                    self.normalized_cube[:,xx,yy]=applyFilter(norm_cube[:,xx,yy],
                    window_size,polyorder)
                    self.normalized_cube[:,xx,yy]=removeTelluric(wavelen,self.normalized_cube[:,xx,yy],self.wmin_wmax_tellurics[0],self.wmin_wmax_tellurics[1])
        #self.normalized_cube[np.where(filt!=0)]=norm_cube[np.where(filt!=0)]-filt[np.where(filt!=0)]
        recon_3dcube=np.zeros_like(self.normalized_cube)
        ncomp=kwargs.get("n_comps",2)
        if(ncomp==0):
            recon_3dcube=self.normalized_cube
            print("no PCA ")
        else:
            PCA_matrix=np.reshape(np.zeros(len(wavelen)*cube_notell.shape[1]*cube_notell.shape[2]),
                    (len(wavelen),cube_notell.shape[1]*cube_notell.shape[2]))

            for wl in range(len(wavelen)):
                for xx in range(cube_notell.shape[1]):
                    for yy in range(cube_notell.shape[2]):
                        PCA_matrix[wl,xx*cube_notell.shape[1]+yy]=self.normalized_cube[wl,xx,yy]

            self.V=svd_wrapper(PCA_matrix.T,'lapack',ncomp,verbose=True)
            transformed=np.dot(self.V,PCA_matrix)
            reconstructed=np.dot(transformed.T,self.V)
            residuals=PCA_matrix-reconstructed.T

            #re-wrap into a cube
            for wl in range(idx_wmax-idx_wmin):
                for xx in range(cube_notell.shape[1]):
                    for yy in range(cube_notell.shape[2]):
                        recon_3dcube[wl,xx,yy]=residuals[wl,cube_notell.shape[1]*xx+yy]
        #waves=np.load("/mnt/diskss/home/rnath/Numpy_PDS70/wavelens_new28.npy")
        #locs=np.where(self.wavelen[0].data>=0)

        return recon_3dcube
    def valentinPreProcessSINFONI(self,*args,**kwargs):
        """
        Performs stellar subtraction on the cube based on Haeffert et al. if n_comps>0
        performs the PCA

        Parameters
        ----------
        perc (float optional, default = 95):
                Percentage of pixels to consider around center.

        polyorder (int optional, default = 1):
                Order of the Savitzky-Golay filter. See applyFilter.
        window_size (int option, default =101):
                Window size has to be odd
        n_comps (int, default=2):
                number of components to be subtractd.

        Returns
        -------
        float :
            Returns a 3D matrix formatted (wl,RA,DEC)
        """


        idx_wmin = find_nearest(self.wavelen,self.wmin_max[0])
        idx_wmax = find_nearest(self.wavelen,self.wmin_max[1])
        cube = self.cube[idx_wmin:idx_wmax,:,:]
        wavelen = self.wavelen[idx_wmin:idx_wmax]
        self.waves_dat=wavelen
        fwhm = self.fwhm[idx_wmin:idx_wmax]
        cube_nt=np.zeros(cube.shape)
        for xx in range(cube.shape[1]):
            for yy in range(cube.shape[2]):
                cube_nt[:,xx,yy]=removeTelluric(wavelen,cube[:,xx,yy],wmin=self.wmin_wmax_tellurics[0],wmax=self.wmin_wmax_tellurics[1])
                #sum the spaxels
        med_frame= np.median(cube_nt, axis=0)
        perc=0.95

        thr = np.percentile(med_frame,perc)
        idx_high = np.where(med_frame>thr)



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
        self.normalized_cube=norm_res_cube

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
        self.V=svd_wrapper(new_matrix.T,'lapack',ncomp,verbose=True)
        transformed=np.dot(self.V,new_matrix)
        reconstructed=np.dot(transformed.T,self.V)
        residuals=new_matrix-reconstructed.T

        new_counter = 0
        resi_3dcube=np.zeros_like(res_cube)

        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                if res_cube[0,j,k] != 0:
                    for i in range(wavelen.shape[0]):
                        resi_3dcube[i,j,k] = residuals[i,new_counter]
                    new_counter+=1
        return resi_3dcube
