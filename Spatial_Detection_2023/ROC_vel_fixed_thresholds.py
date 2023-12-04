from vip_hci.fits import open_fits,write_fits
import joblib
from vip_hci.metrics.stim import compute_stim_map as stim_map
from hciplot import plot_frames
import numpy as np
from vip_hci.preproc.cosmetics import cube_crop_frames,frame_center,frame_crop
from vip_hci.var.shapes import mask_circle
from tensorflow import keras
from matplotlib import pyplot as plt
import glob
from vip_hci.preproc import derotation
from vip_hci.metrics.roc import compute_binary_map
#rads=[18, 19 ,20, 21,22]
rads=range(19,23)
#rads=[ 17, 18,19, 20,21]
#thetas=[90.00,120.00,180.00,330.00,360.00]
#thetas=[90,120,180.00,330.00,360.00]
thetas=[60,120,180.00,360]
#thetas=[60,120.0]#this is for close in commpanions
conts = [5e-04,1e-04]
fpr = np.zeros(10)
tpr = np.zeros(10)
fpr_stim = np.zeros(10)
tpr_stim = np.zeros(10)
fpr_sodinn = np.zeros(10)
tpr_sodinn = np.zeros(10)
fpr_rf = np.zeros(10)
tpr_rf = np.zeros(10)
thresh_stim = np.zeros(10)
thresh_cnn = np.zeros(10)
thresh_sodinn = np.zeros(10)
thresh_rf = np.zeros(10)
mask_rad = 3.5
#model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_notnormalized_hc.h5")
#model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_normalized_hc_lowercont.h5")
#model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_normalized_hc_lowercont.h5")
for rad in rads:
    for theta in thetas:
        for cont in conts:

            #test_framefname="/mnt/disk12tb/Users/rakesh/SpatialDetection/Testing_vel/Training_vel/ccf_cont_{0:1.0e}_rad_{1:3.1f}_theta_{2:3.2f}_frame_".format(cont,rad,theta)
            stim_sig = open_fits("fitsfiles/stim_detmap_{0:1.0e}_{1:3.1f}_{2:3.2f}.fits".format(cont,rad,theta))
            y_pos = frame_center(stim_sig)[0]+rad*np.sin(np.deg2rad(theta))
            x_pos = frame_center(stim_sig)[0] +rad*np.cos(np.deg2rad(theta))
            #plot_frames((stim_sig),circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',save="ROCplots_velstim/detmaps/stim_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta),log=False)
           #thresholds = np.linspace(np.max(stim_sig)/10,(np.max(stim_sig)-1e-04),10)
            thresholds = np.linspace(1.2,5.9,10)
            thresh_stim = thresh_stim+thresholds
            try:
                tpf,fpf,bmp = compute_binary_map(stim_sig, thresholds=thresholds,injections=(y_pos,x_pos)
                                                             ,fwhm=4.8,overlap_threshold=0.5,npix=1,max_blob_fact=6)#0.7 for all 0.5 for close in
            except AttributeError:
                thresholds = thresholds[0:-1]
                try:
                    res_stim = compute_binary_map(stim_sig,thresholds=thresholds,
                                                                   fwhm=4.8,injections=(y_pos,x_pos),npix=1,
                                                                   max_blob_fact=4)
                    tpf = np.zeros(10)
                    fpf = np.zeros(10)
                    for i in range(len(res_stim[0])):
                        tpf[i] = res_stim[0][i]
                        fpf[i] = res_stim[1][i]

                except AttributeError:
                    thresholds = thresholds[0:-1]
                    res_stim = compute_binary_map(stim_sig,thresholds=thresholds,
                                                                                                          fwhm=4.8,injections=(y_pos,x_pos),npix=1,
                                                                                                                                               max_blob_fact=4)
                    tpf = np.zeros(10)
                    fpf = np.zeros(10)
                    for i in range(len(res_stim[0])):
                        tpf[i] = res_stim[0][i]
                        fpf[i] = res_stim[1][i]
            print("STIM TPF={0},FPF={1}".format(tpf,fpf))
            for i in range(len(thresholds)):
                if(tpf[i]>1):
                    temp = tpf[i]-1
                    fpf[i] =fpf[i]+temp
                    tpf[i] =1
            fpr_stim = fpr_stim+fpf
            tpr_stim = tpr_stim+tpf
            
            #print(model.to_json())
            #detmap = np.zeros(61*61).reshape(61,61)
        
            detmap = open_fits("fitsfiles_vel/cnn_detmap_{0:1.0e}_{1:3.1f}_{2:3.2f}.fits".format(cont,rad,theta))
            detmap = mask_circle(detmap,mask_rad)
            y_pos = frame_center(detmap)[0]+rad*np.sin(np.deg2rad(theta))
            x_pos = frame_center(detmap)[0] +rad*np.cos(np.deg2rad(theta))
            thresh = np.linspace(0.09,0.999,10)
            thresh_cnn = thresh_cnn +thresh
            try:
                #thresh=np.linspace(np.max(detmap)/10,np.max(detmap)-1e-04,10)#change 1e-07 to 1e-0
#                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
 #                                                  fwhm=4.8,max_blob_fact=10,overlap_threshold=0.6,npix=2)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                                                   fwhm=4.8,max_blob_fact=10,overlap_threshold=0.5,npix=2)#for the close in ones
            except AttributeError:
  #              thresh=np.linspace(np.max(detmap)/10,np.max(detmap)-1e-02,10)#change 1e-07 to 1e-04
#                thresh=np.linspace(0.1,np.max(detmap)-1e-04,10)#change 1e-07 to 1e-04
                thresh =thresh[0:-1]#np.linspace(0.09,0.95,10)
                #res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                 #                                  fwhm=4.8,max_blob_fact=10,overlap_threshold=0.6,npix=2)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                                                   fwhm=4.8,max_blob_fact=10,overlap_threshold=0.5,npix=2)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
                if(len(res[0])==9):
                    res[0].append(0)
                    res[1].append(0)

            fpr = fpr+res[1]
            tpr = tpr+res[0]

            print("C3PO TPF={0},FPF={1}".format(res[0],res[1]))
            #sodinn
            detmap = open_fits("fitsfiles_vel/sodinn_detmap_{0:1.0e}_{1:3.1f}_{2:3.2f}.fits".format(cont,rad,theta))
            detmap = mask_circle(detmap,mask_rad)
            thresh = np.linspace(0.09,0.999,10)
            thresh_sodinn= thresh_sodinn +thresh
            try:
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                                               fwhm=4.8,max_blob_fact=10,overlap_threshold=0.5,npix=1)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
            except AttributeError:
                thresh = thresh[0:-1]# np.linspace(0.09,0.95,10)
#                thresh=np.linspace(0.1,np.max(detmap)-1e-03,10)#change 1e-07 to 1e-04
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                        fwhm=4.8,max_blob_fact=3,overlap_threshold=0.6,npix=1)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
                if(len(res[0])==9):
                    res[0].append(0)
                    res[1].append(0)
            #for i in range(len(thresh)):
             #   if(res[0][i]>1):
                    #temp = res[0][i]-1
                    #fpf[i] =fpf[i]+temp
              #      res[0][i] =1
            print("SODINN TPF={0},FPF={1}".format(res[0],res[1]))
            tpr_sodinn = tpr_sodinn+res[0]
            fpr_sodinn = fpr_sodinn+res[1]
            #print(tpr_stim,tpr,tpr_sodinn)
print(fpr_stim,fpr,fpr_sodinn)
print(tpr_stim,tpr,tpr_sodinn)
total_insertions=len(thetas)*len(rads)*len(conts)
write_fits("stats_roc_runs_vel_fixedthresh/fpr_stim_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),fpr_stim/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//fpr_cnn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),fpr/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//fpr_sodinn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),fpr_sodinn/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//tpr_stim_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),tpr_stim/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//tpr_cnn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),tpr/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//tpr_sodinn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),tpr_sodinn/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh/thresh_sodinn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),thresh_sodinn/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//thresh_cnn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),thresh_cnn/total_insertions)
write_fits("stats_roc_runs_vel_fixedthresh//thresh_stim_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),thresh_stim/total_insertions)
#plt.savefig("Trial_old.png")
#plt.savefig("vel_fromfitsroc_conts_{0:1.0e}_to_{1:1.0e}_sep_{2:d}_to_{3:d}_mask_{4:2.1f}_correctrange.png".format(np.min(conts),np.max(conts),rads[0],rads[-1],mask_rad))

