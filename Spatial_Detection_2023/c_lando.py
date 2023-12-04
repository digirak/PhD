from tensorflow import keras
import tensorflow as tf
from vip_hci.fits import open_fits
from vip_hci.preproc.cosmetics import cube_crop_frames,frame_center
from vip_hci.preproc import derotation
from vip_hci.metrics.roc import compute_binary_map
from vip_hci.var.shapes import mask_circle
from hciplot import plot_frames
from sklearn.model_selection import train_test_split
from generateData import generateH0,generateH1,genTrainCV,tfConvert,convertToRGBscale,loadPatchData,loadAugPatchData
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle

def visionLSTM(input_shape):
    initializer= 'glorot_uniform'
    data_format='channels_first'
    inp = keras.layers.Input(shape=input_shape)
    x = keras.layers.ConvLSTM2D(
            filters=40,
                kernel_size=(3, 3),
                padding="same",
                return_sequences=True,
                activation="tanh",
                kernel_initializer=initializer,
                data_format=data_format)(inp)
    #x = keras.layers.BatchNormalization()(x)
    x= keras.layers.MaxPooling3D(pool_size=(2,2,2),strides=(1,1,1),data_format=data_format)(x)
    x = keras.layers.ConvLSTM2D(
                filters=80,
            kernel_size=(2, 2),
            padding="same",
            return_sequences=True,
            activation="tanh",
            kernel_initializer=initializer,
            data_format=data_format)(x)
    #x = keras.layers.BatchNormalization()(x)
#    x= keras.layers.MaxPooling3D(pool_size=(1,2,2),strides=(1,1,1))(x)
 #   x = keras.layers.ConvLSTM2D(
  #              filters=32,
   #             kernel_size=(1, 1),
    #            padding="same",
     #           return_sequences=False,
      #          activation="relu")(x)
    #x = keras.layers.Conv2D(filters=0, kernel_size=(1, 1), activation="sigmoid", padding="valid")(x)
    x= keras.layers.MaxPooling3D(pool_size=(2,2,2),data_format=data_format)(x)
    #x=keras.layers.SpatialDropout3D(0.25)(x)
    x= keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu",use_bias=True)(x)
    x = keras.layers.Dropout(0.25)(x)
    output = keras.layers.Dense(1, activation="sigmoid",use_bias=True)(x)
    lstm_model = keras.Model(inp,output)
    return lstm_model
#X,y,fs = loadPatchData(h0path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Data/H0patches_fov11/",
 #                      h1path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Data/H1patches_fov11/",
  #                     Cs=[5e-02,3e-02,1e-02,7e-03,5e-03,3e-03,1e-03,7e-04,5e-04,3e-04,1e-04],
   #                      shifts=[-5,-4,-3,-2,-1,0,1,2,3,4])
#X,y,fs = loadPatchData(h0path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Data/H0patches_fov11/",
 #                      h1path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Data/H1patches_fov11_HC/",
  #                    Cs=[5e-02,3e-02,1e-02,7e-03,5e-03,3e-03,1e-03,7e-04,5e-04,3e-04,1e-04,7e-05,5e-05,3e-05,1e-05,
   #                        7e-06,5e-06,3e-06,1e-06],
    #                 shifts=[-3,-2,-1,0,1,2,3])

#from sklearn.utils import shuffle
#X_shuf,y_shuf,fs_shuf = shuffle(X,y,fs,random_state=42)
#if(len(X.shape)==5):
 #   X_rest = np.mean(np.mean(X_shuf,axis=2),axis=1).reshape(len(X),1,X.shape[-2],X.shape[-1])#X[:,0,1,:,:].reshape(len(X),1,9,9)
#else:
  #  X_rest = np.mean(X_shuf,axis=1).reshape(len(X),1,X.shape[-2],X.shape[-1])
#X_rest = X_rest.reshape(len(X),X.shape[-2]*X.shape[-1])

import random
#locs =random.sample(range(0,len(X_shuf)),10000)
#mask = np.full(X_shuf.shape[0],dtype=bool,fill_value=True)
#mask[locs]=False

#X_test = X_shuf[locs]
#y_test = y_shuf[locs]
#files_test = [fs_shuf[loc] for loc in locs]
#X = X_shuf[mask]
#y = y_shuf[mask]
Cs=[5e-02,3e-02,1e-02,7e-03,5e-03,3e-03,1e-03]#,7e-04,5e-04,3e-04,1e-04]#,7e-05,5e-05,3e-05,1e-05]
 #           7e-06,5e-06,3e-06,1e-06]
#Cs=[5e-02,3e-02,1e-02,7e-03,5e-03,3e-03,1e-03,7e-04,5e-04,3e-04,1e-04]
#rad_distances = [4,8,15,25]
rad_distances = range(4,27)

X_raw,y_raw,fs = loadPatchData(h0path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Training_vel/Training/H0patches/",
                           h1path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Training_vel/Training/H1patches/",
                           Cs=Cs,
                              shifts=[-1,0],
                           rad_distances = rad_distances)
#X_raw,y_raw,fs = loadAugPatchData(h0path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Data/H0patches_fov11/",
 #                          h1path="/mnt/disk12tb/Users/rakesh/SpatialDetection/Data/H1patchesaugmented_HCfov11/",
  #                        Cs=Cs,
   #                       shifts=[-5,-4,-3,-2,-1,1,2,3,4,5],
    #                       rad_distances = rad_distances)

file_names = [f.split('/')[-1] for f in fs]
zero_start = int(len(fs)/2)
test_indices =[]
#angles =[60.00,120.00,150.00,180.00,300.00,330.00]
#angles =[60.00]#,150.00,180.00,270.00,300.00,330.00]
#angles =[90.00,120.00,180.00,330.00,360.00]
angles =[60.00,120.00,180.00,360.00]
for i in range(len(file_names)):
    for ang in angles:
        search_string ="_theta_{0:3.2f}.fits".format(ang)
        if search_string in file_names[i]:
            test_indices.append(i)
locs = test_indices
mask = np.full(X_raw.shape[0],dtype=bool,fill_value=True)
mask[locs] = False
zero_locs = np.random.randint(zero_start,len(fs),int(len(locs)))
mask[zero_locs]= False
X = X_raw[mask][:,:,:,:,:]
y =y_raw[mask]

X_test = X_raw[~mask][:,:,:,:,:]
y_test = y_raw[~mask]
fs_test = np.asarray(fs)[~mask]
fs_test = list(fs_test)
n_seq = 83
if(len(X.shape)==5):
   # X = np.mean(X[:,0:n_seq,:,:,:],axis=2).reshape(len(X),n_seq,1,X.shape[-2],X.shape[-1])
    X = X[:,0:n_seq,:,:,:].reshape(len(X),n_seq,20,X.shape[-2],X.shape[-1])
else:
    X = X[:,0:n_seq,:,:].reshape(len(X),n_seq,1,X.shape[-2],X.shape[-1])
print(X.shape)
from sklearn.utils import shuffle
X_test,y_test,fs_test = shuffle(X_test,y_test,fs_test,random_state=51)
file_strings=[]
for f in fs_test:
    if(f!='zeros'):
        st = f.split('/')[-1].split('_')
        if(st[0]=='H1'):
            file_strings.append([st[4],st[6]])
        else:
#            file_strings.append([st[7],st[9]])
            file_strings.append([st[5],st[7]])
    else:
        file_strings.append(['0','0'])
scaler = MinMaxScaler()
if(len(X_test.shape)==5):
    #X_test = np.mean(X_test[:,0:n_seq,:,:,:],axis=2).reshape(len(X_test),n_seq,1,X.shape[-2],X.shape[-1])
    X_test = (X_test[:,0:n_seq,:,:,:]).reshape(len(X_test),n_seq,20,X.shape[-2],X.shape[-1])
else:
    X_test = X_test[:,0:n_seq,:,:].reshape(len(X_test),n_seq,1,X.shape[-2],X.shape[-1])
#for i in range(X.shape[0]):
 #   im_seq = X[i]#scaler.fit_transform(X[:,i,:,:,j])
  #  im_seq = (im_seq - np.min(im_seq))/(np.max(im_seq)-np.min(im_seq))
   # X[i]= im_seq

#for i in range(X_test.shape[0]):
 #   im_seq = X_test[i]#scaler.fit_transform(X[:,i,:,:,j])
  #  im_seq = (im_seq - np.min(im_seq))/(np.max(im_seq)-np.min(im_seq))
   # X_test[i]= im_seq
X_train,X_valid, y_train,y_valid = train_test_split(X,y,test_size=0.1,random_state=42,shuffle =True)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model =visionLSTM(X_train.shape[1::])
    model.summary()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    opt = tf.keras.optimizers.Adam(0.0005)#10^-2
    #opt = tf.keras.optimizers.Adagrad(0.001)#10^-2
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
            "/mnt/disk12tb/Users/rakesh/SpatialDetection/Models/sodinn_notnormalized_LC_largevel.h5", save_best_only=True
            )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2,restore_best_weights = True,min_delta = 1e-07)
    model.compile(loss =loss,optimizer=opt,
            metrics=[tf.metrics.TruePositives(),tf.metrics.TrueNegatives()])
    batch_size = 32
    train = tfConvert(X_train,y_train,batch_size)
    valid = tfConvert(X_valid,y_valid,batch_size)
    history=model.fit(train,epochs=15,batch_size=batch_size,validation_data=valid,callbacks=[callback,checkpoint_cb])
    f_pkl = open("hist_LSTM_hc_newdata.pkl","wb")
    pickle.dump(history.history,f_pkl)
    f_pkl.close()

    cv_scores = model.evaluate(X_test,y_test)


    preds = model.predict(X_test)
    print("TP Test is {0:1.0f}/{1:1.0f}".format(cv_scores[1],np.sum(y_test)))
    print("TN Test is {0:1.0f}/{1:1.0f}".format(cv_scores[2],len(y_test)-np.sum(y_test)))
cont =7e-04
rad=22
theta=60
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

featmap = np.zeros(61*61*121*83*20).reshape(61,61,83,20,11,11)#reshape(samplex,sampley,time,vel,featurex,featurey)
for x in range(5,56):
    for y in range(5,56):
        cropped_adi_seq = cube_crop_frames(seqs,size=11,xy=(y,x),verbose=False)
        test_pixel =cropped_adi_seq[:,:,:,:]
        #np.ravel(np.mean(np.median(cropped_adi_seq,axis=1),axis=0))
#        test_pixel = (test_pixel-np.min(test_pixel))/(np.max(test_pixel)-np.min(test_pixel))
        featmap[x,y,:,:,:] = test_pixel
feats = np.reshape(featmap,((61*61),83,20,11,11))
dets = model.predict(feats)
dets = np.ravel(dets)
detmap = dets.reshape(61,61)[5:56,5:56]
detmap = mask_circle(detmap,radius=3.5)
plot_frames((detmap),circle=(y_pos-5,x_pos-5),circle_radius=2.5,circle_color='r',save="largevel_trialsodinn_{0:1.0e}_{1:3.1f}_{2:3.2f}.png".format(cont,rad,theta),log=False)
thresh=np.linspace(0.1,np.max(detmap)-1e-04,10)
res =compute_binary_map(detmap,thresholds=thresh,injections=(y_pos-5,x_pos-5),
                                   fwhm=4.8,max_blob_fact=3,overlap_threshold=0.7,npix=2)
print(res[0],res[1])
plot_frames(tuple(res[2]),rows=5,circle=(y_pos-5,x_pos-5),circle_radius=2.5,circle_color='r',save="sodinn_largevelbinmap_{0:1.0e}_{1:3.1f}_{2:3.2f}_withvel.png".format(cont,rad,theta))

import pandas as pd

df = pd.DataFrame(file_strings,columns=['Contrast','R'])
df['Contrast']=df['Contrast'].astype('float64')
df['Predicted'] = preds
df['Truth'] = y_test
search_string ="H1_seq_training_cont_"
indices=[]
rs_detect=[]

for i in range(len(fs_test)):
    if search_string in fs_test[i]:
        indices.append(i)
        rs_detect.append(file_strings[i][1])
df['Nonaugmented'] = False
df['Nonaugmented'].iloc[indices] = True
df.to_excel("lstm_septest_{0:d}_hc_vel_newdata.xlsx".format(n_seq))
preds_int = [1 if item>0.5 else 0 for item in preds]
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test.astype(np.int),preds_int)


import seaborn as sns
from matplotlib import pyplot as plt

ax = sns.heatmap(cf_matrix, annot=True,fmt ='d',cbar=False,
                        cmap ='Blues')
#ax.set_title('Confusion Matrix with RF R={0:1.0e}, C={1:1.0e}'.format(R,C));
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['No planet','Planet'])
ax.yaxis.set_ticklabels(['No planet','Planet'])
plt.savefig("/mnt/diskss/home/rnath/Monday_meetings/Plots_29.08.2022/cm_lstm_test_fov11_newdata.png")
plt.close()
conts=np.sort(df['Contrast'].unique())[::-1]
tpf=[]
from matplotlib import pyplot as plt
plt.style.use('seaborn')
for cont in conts:
    if(cont==0):
        break

    df_filt = df.loc[((df['Contrast']==cont) &(df['Nonaugmented']==True)),['Predicted','Truth']]
    print("C={0:1.0e}, predicted positives = {1:1.0f}/{2:1.0f}".format(cont,np.sum(df_filt['Predicted']),np.sum(df_filt['Truth'])))
    tpf.append(np.sum(df_filt['Predicted'])/np.sum(df_filt['Truth']))
ax = plt.gca()
for i in range(len(conts)):
    if(conts[i]==0):
        break
    ax.scatter(conts[i],tpf[i],color='k')
    ax.set_xscale('log')
    ax.set_xlabel('Contrast')
    ax.set_ylabel("TPF")
    plt.savefig("/mnt/diskss/home/rnath/Monday_meetings/Plots_29.08.2022/lstm_auc_tpf_nseq_{0:d}_hc_vel_newdata.png".format(n_seq))
                    


