import numpy as np
from sklearn import neural_network
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler,Callback
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
        concatenate
import keras
class AllAIMethods():
    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        if(len(self.X_train.shape)!=len(X_test.shape)):
            print("Check the Data!")
    def autoencoder(self):
        inp = Input(shape=(self.X_train.shape[1],))
        encoded = Dense(self.X_train.shape[1], activation='relu')(inp)
        encoded = Dense(1024, activation='relu')(encoded)
        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dense(256, activation='relu')(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(self.X_train.shape[1], activation='relu')(decoded)
        output  = Dense(self.y_train.shape[-1], activation='relu')(decoded)
        model = models.Model(inputs=inp, outputs=output)
    #    opt = optimizers.SGD(0.001)
    #model = keras.utils.multi_gpu_model(model, gpus=2)
   # model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model

    def conv1d(self):
        nclass = self.y_train.shape[-1]
        if(len(self.X_train.shape)<3):
            x_conv=np.reshape(self.X_train,(1,self.X_train.shape[0],self.X_train.shape[1]))
        inp = Input(shape=(x_conv.shape[1],x_conv.shape[2]))
        #inp = Input(shape=(x_.shape[1],x_conv.shape[2]))
        img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(inp)
        img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=4)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(2048, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(2048, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(2048, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Convolution1D(2048, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(2048, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(2048, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=4)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(2048, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(2048, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(2048, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(1024, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(1024, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(1024, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = Convolution1D(1024, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
        img_1 = GlobalMaxPool1D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)

        dense_1 = Dense(128, activation=activations.relu)(img_1)
        dense_1 = Dense(512, activation=activations.relu)(dense_1)
        dense_1 = Dense(512, activation=activations.relu)(img_1)
        dense_1 = Dense(1024, activation=activations.relu)(dense_1)
        dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)
        #opt = optimizers.SGD(0.001)
        #model = keras.utils.multi_gpu_model(model, gpus=2)
        #model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
        model.summary()
        return model
    def mlp_regression(self):
        inp=Input(shape=(self.X_train.shape[1],))
        if len(self.y_train.shape)>1:
            op=self.y_train.shape[-1]
        else:
            op=1
        mlp=Dense(6000,activation=activations.relu)(inp)
        mlp=Dense(4000,activation=activations.relu)(inp)
        mlp=Dense(3000,activation=activations.relu)(inp)
        mlp=Dense(2300,activation=activations.relu)(mlp)
        mlp=Dense(1800,activation=activations.relu)(mlp)
        mlp=Dense(1500,activation=activations.relu)(mlp)
        mlp=Dense(1000,activation=activations.relu)(mlp)
        #mlp_l6=Dense(750,activation=activations.relu)(mlp_l5)
        mlp=Dense(750,activation=activations.relu)(mlp)
        mlp=Dense(500,activation=activations.relu)(mlp)
        mlp=Dense(200,activation=activations.relu)(mlp)
        out=Dense(op,activation=activations.relu)(mlp)
        model=models.Model(inputs=inp,outputs=out)
        #opt=optimizers.adam(0.01)
       # model = keras.utils.multi_gpu_model(model, gpus=2,cpu_relocation=True)
        #model.compile(optimizer=opt, loss=losses.mean_squared_error, metrics=['mae'])
        model.summary()
        return model



