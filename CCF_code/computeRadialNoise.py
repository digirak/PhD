#from callPCACrossCorr import callPCACrossCorr
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from photutils import CircularAperture,aperture_photometry
import vip_hci
from parallelCompareTemplates import CrossCorr

def rec2polar(x,y):
    r= np.sqrt((x-35)**2+(y-35)**2)
    #print(x-35,y-35)
    thetap=np.arctan((y-35)/(x-35))
    if((x-35>=0) &((y-35)>=0)):
        theta=thetap
    elif (((x-35)<=0) &((y-35)>=0)):
        #QII
        theta=np.pi-thetap
    elif(((x-35)<=0) & ((y-35)<=0)):
        #QIII
        theta=np.pi+thetap
    elif(((x-35)>=0) & ((y-35)<=0)):
        #QIV
        theta=2*np.pi-thetap

    return r,(theta)
def polar2rec(r,theta):
    theta=(theta)
    r=int(r)
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    #print(x-35,y-35)
    return (np.int64(x+35),np.int64(y+35))
def measureSpatialSpec(datcube,loc,aperture_size):
    slices=[]

    for wl in range(len(aperture_size)):
        apertures=CircularAperture([loc[0],loc[1]],r=aperture_size[wl]/2)
        slices.append(np.float64(aperture_photometry(datcube[wl,:,:],
                                    apertures)['aperture_sum']))
    return np.asarray(slices),apertures
def computeRadialNoise(cube,
                       loc,
                       vels,
                       waves_dat,
                       temp_wavs,
                       temp_flux,
                       window_size,
                       order,
                       aperture_size=[5],
                       skip_pixels=8):
    apertures=CircularAperture(loc,r=skip_pixels)
    mask=np.zeros(cube[0,:,:].shape)
    angs=np.arange(0,2*np.pi,2*np.pi/360.0)
    temp_w=temp_wavs
    temp_f=temp_flux
    r,theta=rec2polar(loc[0],loc[1])
    for k in range(len(angs)):
        i=angs[k]
        x,y=polar2rec(r,theta+i)
        if((((x>apertures.bbox.ixmin) & (x<apertures.bbox.ixmax ) &
             (y>apertures.bbox.iymin) & (y<apertures.bbox.iymax)))):
            #print("hit")
            mask[x,y]=-1
        else:
            mask[x,y]=1
    locs_noise=np.where(mask==1)
    sps=[]
    #posns=np.random.randint(0,np.max(locs_noise),10)
    gif_path = "test.gif"
    frames_path = "pixels/im_"
    masks=[]
    #sps_final=[]
    for epoch in range(15):
        mask_clone=np.zeros(mask.shape)
        posns=np.random.randint(0,len(locs_noise[0]),1)
        #print(locs_noise)
        #posnx=np.random.randint(min(locs_noise[0]),max(locs_noise[0]),1)
        #posny=np.random.randint(min(locs_noise[1]),max(locs_noise[1]),1)
        mask_clone[locs_noise[0],locs_noise[1]]=1
        mask_clone[locs_noise[0][posns],locs_noise[1][posns]]=5


        #plt.pcolormesh(mask_clone)
        for i in range(len(posns)):
            if len(aperture_size)>1:

                sp,apertures=measureSpatialSpec(cube,[locs_noise[0][posns[i]],
                locs_noise[1][posns[i]]],aperture_size)
                apmask=apertures.to_mask()
                mask_clone=mask_clone+np.transpose(
                                            apmask.to_image(shape=mask_clone.shape))
            else:
                sp=cube[:,locs_noise[0][posns[i]],locs_noise[1][posns[i]]]
            t=temp_f
            w=temp_w
            CC=CrossCorr(vels)
            mean_n,n,snn=CC.compareFluxes(waves_dat,
                                          sp,temp_wavs,
                                          temp_flux,
                                          window_size=window_size,
                                          order=order)
               # n.append(n1
            sps.append(np.std(mean_n))
            masks.append(mask_clone)
        #sp_inter=np.mean(sps,axis=0)
        #sps_final.append(sp_inter)
    sp_final=np.mean(sps)
    #with imageio.get_writer(gif_path, mode='I',fps=3) as writer:
     ##   for i in range(30):
       #     print(frames_path.format(i=i))
        #    writer.append_data(imageio.imread(frames_path+"{i}.jpg".format(i=i)))

    #print(np.std(sp_final))
    return np.mean(sps),n,masks#,sps

