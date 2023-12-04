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
rads=range(19,24)
#[19,20, 21,22, 23, 24]
#rads=[ 17, 18,19, 20,21]
#thetas=[90.00,120.00,180.00,330.00,360.00]
#thetas=[90,120,180.00,330.00,360.00]
thetas=[60,120,180.00,360.00]
conts =[5e-04]# [5e-04,3e-04, 1e-04]
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
RF_model = joblib.load("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/rf.joblib")
mask_rad = 3.5
#model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_notnormalized_hc.h5")
#model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_normalized_hc_lowercont.h5")
#model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_normalized_hc_lowercont.h5")
model = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/cnn_notnormalized_largevel_LC_withvel.h5")
#sodinn = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/sodinn_notnormalized_hc_lowercont_changedH0.h5")
#sodinn = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/sodinn_notnormalized_olddata.h5")
sodinn = keras.models.load_model("/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/sodinn_notnormalized_LC_largevel.h5")
for rad in rads:
    for theta in thetas:
        for cont in conts:

            test_framefname="/mnt/disk12tb/Users/rakesh/SpatialDetection/Testing_vel/Training_vel/ccf_cont_{0:1.0e}_rad_{1:3.1f}_theta_{2:3.2f}_frame_".format(cont,rad,theta)
            fitsfiles=[]
            angs=[]
            y_pos = 30+rad*np.sin(np.deg2rad(theta))
            x_pos = 30 +rad*np.cos(np.deg2rad(theta))
            rot_angles = open_fits("/mnt/disk12tb/Users/rakesh/HD179218/derot_angles.fits")
            print(test_framefname)
            for i in range(83):
                fitsfiles.append(open_fits(test_framefname+"{0:02d}.fits".format(i),verbose=False))
                angs.append(rot_angles[i])
            adi_seq = np.asarray(fitsfiles)
            angs = np.asarray(angs)
            seqs=np.zeros(83*61*61*20).reshape(83,20,61,61)#using all three
            for i in range(20):
                seqs[:,i,:,:] = derotation.cube_derotate(adi_seq[:,:,:,i],angle_list=-angs)
            stim_sig = stim_map(seqs[:,10,:,:])
            stim_sig= mask_circle(stim_sig,radius=mask_rad)
            #plot_frames((stim_sig),circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',save="ROCplots_velstim/detmaps/stim_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta),log=False)
           #thresholds = np.linspace(np.max(stim_sig)/10,(np.max(stim_sig)-1e-04),10)
            thresholds = np.linspace(1.2,(np.max(stim_sig)-1e-04),10)
            tpf,fpf,bmp = compute_binary_map(stim_sig, thresholds=thresholds,injections=(y_pos,x_pos)
                                                         ,fwhm=4.8,overlap_threshold=0.7,npix=1,max_blob_fact=3)
            thresh_stim = thresh_stim+thresholds
            print("STIM TPF={0},FPF={1}".format(tpf,fpf))
            for i in range(len(thresholds)):
                if(tpf[i]>1):
                    temp = tpf[i]-1
                    fpf[i] =fpf[i]+temp
                    tpf[i] =1
            fpr_stim = fpr_stim+fpf
            tpr_stim = tpr_stim+tpf
            labs = ["TPs={0},FPs={1},thres={2}".format(i,j,k) for i,j,k in zip(tpf,fpf,thresholds)]
            labs = tuple(labs)
            plot_frames(tuple(bmp),rows=5,circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',
                    label = labs,
                    save="ROCplots_velstim/binmaps/binmap_regrange_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta))
            
            #print(model.to_json())
            #detmap = np.zeros(61*61).reshape(61,61)
            featmap_cnn = np.zeros(61*61*121*20).reshape(61,61,20,11,11)
            featmap_sodinn = np.zeros(61*61*121*83*20).reshape(61,61,83,20,11,11)
            for x in range(5,56):
                for y in range(5,56):
                    cropped_adi_seq = cube_crop_frames(seqs,size=11,xy=(y,x),verbose=False)
                    test_pixel = np.mean(cropped_adi_seq,axis=0)[:,:,:]
                    featmap_sodinn[x,y] = cropped_adi_seq   
                    featmap_cnn[x,y,:,:,:] = test_pixel

             #       test_pixel = (test_pixel-np.min(test_pixel))/(np.max(test_pixel)-np.min(test_pixel))
            #        featmap_RF[x,y] = test_pixel
            feats = np.reshape(featmap_cnn,((61*61),20,11,11))
            dets = model.predict(feats)
            dets = np.ravel(dets)
            detmap = dets.reshape(61,61)
            detmap = mask_circle(detmap,radius=mask_rad)
            y_pos = frame_center(detmap)[0]+rad*np.sin(np.deg2rad(theta))
            x_pos = frame_center(detmap)[0] +rad*np.cos(np.deg2rad(theta))
            plot_frames((detmap),circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',save="ROCplots_velcnn/detmaps/cnn3_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta),log=False)
#            thresh=np.linspace(np.max(detmap)/10,np.max(detmap)-1e-04,10)#change 1e-07 to 1e-04
        
            try:
                #thresh=np.linspace(np.max(detmap)/10,np.max(detmap)-1e-04,10)#change 1e-07 to 1e-0
                thresh = np.linspace(0.09,0.999,10)
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                                                   fwhm=4.8,max_blob_fact=10,overlap_threshold=0.6,npix=2)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
            except AttributeError:
  #              thresh=np.linspace(np.max(detmap)/10,np.max(detmap)-1e-02,10)#change 1e-07 to 1e-04
#                thresh=np.linspace(0.1,np.max(detmap)-1e-04,10)#change 1e-07 to 1e-04
                thresh = np.linspace(0.09,0.988,10)
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                                                   fwhm=4.8,max_blob_fact=10,overlap_threshold=0.6,npix=2)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3

            thresh_cnn = thresh_cnn +thresh
            print("CNN TPF={0},FPF={1}".format(res[0],res[1]))
            labs = ["TPs={0},FPs={1},thresh={2}".format(i,j,k) for i,j,k in zip(res[0],res[1],thresh)]
            labs = tuple(labs)
            plot_frames(tuple(res[2]),rows=5,circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',
                    label = labs,
                    save="ROCplots_velcnn/binmaps/binmap3_regrange_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta))
            fpr = fpr+res[1]
            tpr = tpr+res[0]
            #sodinn
            feats = np.reshape(featmap_sodinn,((61*61),83,20,11,11))
            dets = sodinn.predict(feats)
            dets = np.ravel(dets)
            detmap = dets.reshape(61,61)
            detmap = mask_circle(detmap,radius=mask_rad)
#            y_pos = frame_center(detmap)[0]+rad*np.sin(np.deg2rad(theta))
 #           x_pos = frame_center(detmap)[1]+rad*np.cos(np.deg2rad(theta))
            plot_frames((detmap),circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',save="ROCplots_velsodinn/detmaps/sodinn_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta),log=False)
            #thresh=np.linspace(np.max(detmap)/10,np.max(detmap)-1e-03,10)#change 1e-07 to 1e-04
            try:
                thresh = np.linspace(0.09,0.999,10)
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                                               fwhm=4.8,max_blob_fact=8,overlap_threshold=0.6,npix=1)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
            except AttributeError:
                thresh = np.linspace(0.09,0.95,10)
#                thresh=np.linspace(0.1,np.max(detmap)-1e-03,10)#change 1e-07 to 1e-04
                res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos,x_pos),
                        fwhm=4.8,max_blob_fact=2,overlap_threshold=0.6,npix=1)#undercounted TPF so changing overlap thresh, changed maxblob form 10 to 3
            #for i in range(len(thresh)):
             #   if(res[0][i]>1):
                    #temp = res[0][i]-1
                    #fpf[i] =fpf[i]+temp
              #      res[0][i] =1
            print("SODINN TPF={0},FPF={1}".format(res[0],res[1]))
            labs = ["TPs={0},FPs={1},thresh={2}".format(i,j,k) for i,j,k in zip(res[0],res[1],thresh)]
            labs = tuple(labs)
            plot_frames(tuple(res[2]),rows=5,circle=(y_pos,x_pos),circle_radius=2.5,circle_color='r',
                    label = labs,
                    save="ROCplots_velsodinn/binmaps/binmap3_regange_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta))
            tpr_sodinn = tpr_sodinn+res[0]
            fpr_sodinn = fpr_sodinn+res[1]
            thresh_sodinn= thresh_sodinn +thresh
            #print(tpr_stim,tpr,tpr_sodinn)
plt.style.use('seaborn')
ax = plt.gca()
total_insertions =len(rads)*len(thetas)*len(conts)
print("Total insertions:{0}".format(total_insertions))
plt.plot(fpr/total_insertions,tpr/total_insertions,label='CNN',ls='--',marker='d')
#plt.plot(fpr_stim[::]/total_insertions,tpr_stim[::]/(len(rads)*len(thetas)*len(conts)),label='STIM',ls='--',marker='d')
plt.plot(fpr_sodinn[::]/total_insertions,tpr_sodinn[::]/(len(rads)*len(thetas)*len(conts)),label='SODINN',ls='--',marker='d')
ax.set_xscale('log')
plt.xlabel("mean full frame FPs")
plt.ylabel("TPR")
plt.title("3-4 FWHM from the star contrast {0:1.0e} - {1:1.0e}".format(np.min(conts),np.max(conts)))
plt.legend()
print(fpr_stim,fpr,fpr_sodinn,fpr_rf)
print(tpr_stim,tpr,tpr_sodinn,tpr_rf)
write_fits("stats_roc_runs_vel/fpr_stim_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),fpr_stim/total_insertions)
write_fits("stats_roc_runs_vel/fpr_cnn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),fpr/total_insertions)
write_fits("stats_roc_runs_vel/fpr_sodinn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),fpr_sodinn/total_insertions)
write_fits("stats_roc_runs_vel/tpr_stim_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),tpr_stim/total_insertions)
write_fits("stats_roc_runs_vel/tpr_cnn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),tpr/total_insertions)
write_fits("stats_roc_runs_vel/tpr_sodinn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),tpr_sodinn/total_insertions)
write_fits("stats_roc_runs_vel/thresh_sodinn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),thresh_sodinn/total_insertions)
write_fits("stats_roc_runs_vel/thresh_cnn_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),thresh_cnn/total_insertions)
write_fits("stats_roc_runs_vel/thresh_stim_mask_{0:1.1f}_cont_from{1:1.0e}_to_{2:1.0e}_rad_from_{3:1.2f}_to_{4:1.2f}.fits".format(mask_rad,np.min(conts),np.max(conts),np.min(rads),np.max(rads)),thresh_stim/total_insertions)
#plt.savefig("Trial_old.png")
plt.savefig("vel_justMLroc_conts_{0:1.0e}_to_{1:1.0e}_sep_{2:d}_to_{3:d}_allthree_mask3.5_correctrange.png".format(np.min(conts),np.max(conts),rads[0],rads[-1]))

