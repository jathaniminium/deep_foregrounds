import os
import sys
import glob
import pickle as pk
import numpy as np
import pandas as pd
import random as rn

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.models import load_model

from keras import optimizers
from keras import callbacks

from my_classes import DataGenerator
from models import *
K.set_image_dim_ordering('th')

########################################################################
##Code from example by Zolton Fedor to have tensorflow dynamically grow GPU memory
##as needed, as opposed to greedily taking all available GPU memory.
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
##config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
##config.log_device_placement = True  # to log device placement (on which device the operation ran)
## (nothing gets printed in Jupyter, only if you run it standalone)
#
##OR, specify a maximum fraction of GPU memory to allocate.
#config.gpu_options.per_process_gpu_memory_fraction=0.9
#
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras
#######################################################################


import matplotlib.pyplot as plt
plt.ion()

npix = 128
seed = 1111
sense_P = 70.0
np.random.seed(seed)
rn.seed(seed)


n_nu = 9
feature_scale = False
lr = 0.25

val_size = 0.1
batch_size = 16
epochs = 50

input_shape = (n_nu, npix, npix)

#how many Stokes channels?
n_stokes = 1

#Collect training data.
signal_list = np.sort(glob.glob('/home/jason/codes/data/deep_foregrounds/total_signal_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(int(n_nu))+'/tot*npy'))
cmb_list = np.sort(glob.glob('/home/jason/codes/data/deep_foregrounds/cmb_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(int(n_nu))+'/cmb*.npy'))


##########################################
def load_signal(file_path, feature_scale=False):
	this_signal = np.load(file_path)
	if just_T:
		this_signal = this_signal[:,0,:,:]
		
	#Reshape feature array.
	this_signal = this_signal.reshape((n_stokes*n_nu,npix,npix))
	
	#Mean-subtract and normalize each channel.
	if feature_scale:
		for j in range(this_signal.shape[0]):
			this_signal[j,:,:] -= this_signal[j,:,:].mean()
			this_signal[j,:,:] /= this_signal[j,:,:].std()
			
	return this_signal
	
def load_cmb(file_path):
	this_cmb = np.load(file_path)
	
	if just_T:
		this_cmb = this_cmb[0,:,:]
	
	#Reshape array.
	this_cmb = this_cmb.reshape((n_stokes,npix,npix))
	
	return this_cmb
	
	
def prepare_data(signal_list, cmb_list):
	X = np.empty(shape=(len(signal_list),n_stokes*n_nu,npix,npix), dtype=np.float32)
	y = np.empty(shape=(len(signal_list),n_stokes,npix,npix), dtype=np.float32)

        #for i in range(len(signal_list)):
 	#	if i%1000 == 0 or i == len(signal_list):
	for i in range(850):
		if i%1000 == 0 or i == 849:
			print(str(i) + '/' + str(len(signal_list)))
		X[i] = load_signal(signal_list[i], feature_scale=feature_scale)
		y[i] = load_cmb(cmb_list[i])
		
	return X, y

def prepare_batch(X_samples, y_samples, batch_size):
	batch_size = len(X_samples) / batch_size
	X_batches = np.split(X_samples, batch_size)
	y_batches = np.split(y_samples, batch_size)
	while True:
		for b in range(len(X_batches)):
			x = np.array(list(map(load_signal, X_batches[b])))
			y = np.array(list(map(load_cmb, y_batches[b])))
		yield x, y
    
    
callbacks_list = [#callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                  #                            factor = 0.5,
                  #                            patience = 3,
                  #                            verbose = 1),
    
                  #callbacks.EarlyStopping(monitor='val_loss', 
                  #                        min_delta=1e-6, 
                  #                        patience=5),
                  callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.4f}.hdf5', 						    monitor='val_loss',
					    verbose=0, save_best_only=False,
					    save_weights_only=False, 
                  			    mode='auto', period=1)
                 ]
	

	
	
def run_model(Xlist_train, ylist_train, Xlist_val, ylist_val):
	#hist = model.fit(X_train, y_train, 
	#                 epochs=epochs,
	#                 batch_size=batch_size,
	#                 validation_data = (X_val, y_val),
	#                 callbacks = callbacks_list,
	#                 shuffle = True)
                  

	params = {'dim': (npix,npix),
		  'batch_size': batch_size,
		  'n_nu': n_nu,
		  'n_stokes': n_stokes,
		  'feature_scale': feature_scale}
    
	train_datagen = DataGenerator(Xlist_train, ylist_train, **params)
	val_datagen = DataGenerator(Xlist_val, ylist_val, **params)
  								  
	hist = model.fit_generator(generator=train_datagen,
 	                           validation_data=val_datagen,
 	                           #validation_steps=int(np.floor(len(X_val)/batch_size)),
 	                           epochs=epochs,
    	                           #steps_per_epoch=int(np.floor(len(X_train)/batch_size)),
    	                           callbacks=callbacks_list,
    	                           use_multiprocessing=True,
    	                           workers=10
    	                           )    
        
	model_name = 'final_model.h5'
	model.save(model_name)

	return hist

def plot_history(hist):
	plt.figure(1)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.semilogy(hist.history['loss'], label='Training')
	plt.semilogy(hist.history['val_loss'], label='Validation')
	plt.title('Loss')
	plt.legend(['Training', 'Validation'])

	plt.figure(1)
	plt.legend(loc='best')

	#Compare prediction to validation set.
	#train_datagen = prepare_batch(X_train, y_train, batch_size=batch_size)
	#val_datagen = prepare_batch(X_val, y_val, batch_size=batch_size)
	prediction_val = model.predict(X_val[0:5])
	prediction_train = model.predict(X_train[0:5])

	v0 = y_val[0]
	p0 = prediction_val[0].reshape(n_stokes,npix,npix)

	plt.figure(20)
	plt.clf()
	plt.imshow(v0[0,:,:])
	plt.title('Expectation')
	plt.colorbar()
	
	plt.figure(30)
	plt.clf()
	plt.imshow(p0[0,:,:])
	plt.title('Prediction')
	plt.colorbar()	
	
	
	
	
	
	
	
adam = optimizers.adam(lr=lr)

model = deepCMBNet(input_shape=input_shape)
model.compile(optimizer=adam, loss='mean_squared_error')

model.summary()

print('Preparing data...')
#X, y = prepare_data(signal_list, cmb_list)

#X_train, X_val, y_train, y_val = train_test_split(X, y,
#	                                              test_size=val_size,
#	                                              random_state=seed,
#	                                              shuffle=True)
	              
print('Begin training...')
Xlist_train, Xlist_val, ylist_train, ylist_val = train_test_split(signal_list, cmb_list,
	                                  	                  test_size=val_size,
	                                               		  random_state=seed,
	                                              		  shuffle=True)                                
hist = run_model(Xlist_train, ylist_train, Xlist_val, ylist_val)


