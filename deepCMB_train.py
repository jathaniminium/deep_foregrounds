import os
import sys
import glob
import pickle as pk
import numpy as np
import pandas as pd
import random as rn
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input
from keras import optimizers
from keras import callbacks
K.set_image_dim_ordering('th')

import matplotlib.pyplot as plt
plt.ion()

npix = 64
seed = 1111
sense_P = 70.0
np.random.seed(seed)
rn.seed(seed)

just_T = True
n_nu = 5
feature_scale = False
lr = 0.25

val_size = 0.1
batch_size = 32
epochs = 1000

input_shape = (n_nu, npix, npix)

#how many Stokes channels?
if just_T:
    n_stokes = 1
else:
    n_stokes = 3

#Collect training data.
signal_list = np.sort(glob.glob('/home/jason/codes/data/deep_foregrounds/total_signal_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(int(n_nu))+'/tot*npy'))
cmb_list = np.sort(glob.glob('/home/jason/codes/data/deep_foregrounds/cmb_SV_'+str(int(npix))+'x'+str(int(npix))+'_senseP'+str(sense_P)+'_seed'+str(int(seed))+'_nu'+str(int(n_nu))+'/cmb*.npy'))

##############
#Code shamelessly ripped from https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/	
import threading

class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return next(self.it)


def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g
####################

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
	
	for i in range(len(signal_list)):
		if i%1000 == 0 or i == len(signal_list):
			print(str(i) + '/' + str(len(signal_list)))
		X[i] = load_signal(signal_list[i], feature_scale=feature_scale)
		y[i] = load_cmb(cmb_list[i])
		
	return X, y

@threadsafe_generator
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
                  callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.4f}.hdf5', 												monitor='val_loss', 
                  							verbose=0, save_best_only=False, 												save_weights_only=False, 
                  							mode='auto', period=1)
                 ]
	
def conv_block(input_data, filters, 
               kernel_size=(5,5),
               activation='selu',
               dropout_rate=0.3,
               batchnorm=True):
	'''
	Define a generic convoluton block to use with the deepCMBNet model.
	'''
	x = Dropout(dropout_rate)(input_data)
	x = Conv2D(filters, kernel_size, 
	            activation=activation, 
	            strides=(1,1), 
	            padding='same')(x)
	if batchnorm:
		x = BatchNormalization()(x)
	
	return x
	
def add_residuals(x1,x2):
	'''
	Combine inputs to make one input.
	'''
	try:
		assert x1.shape[1:] == x2.shape[1:]
	except AssertionError:
		print('Shape of inputs being added must be the same!')
		
	x = keras.layers.add([x1,x2])
	
	return x
	
	
	
def deepCMBNet(input_shape=(5,64,64)):
	'''
	Implementation of model in arXiv:1810.01483
	'''
	
	x0 = Input(shape=input_shape)
	
	#conv1 block
	x1 = conv_block(x0, filters=64)
	
	#conv2 block
	x2 = conv_block(x1, filters=64)
	
	#conv3 block + residual connection.
	print('x1 shape: ', x1.shape[1:])
	print('x2 shape: ', x2.shape[1:])
	print('Adding x1 and x2...')
	x3 = add_residuals(x1,x2)
	x3 = conv_block(x3, filters=64)
	
	#conv4 block + MaxPool
	x4 = MaxPooling2D((2,2))(x3)
	x4 = conv_block(x4, filters=128)
	
	#conv5 block + residual connection.
	x3b = MaxPooling2D((2,2))(x3)
	x3b = conv_block(x3b, filters=128)
	
	print('x3b shape: ', x3b.shape[1:])
	print('x4 shape: ', x4.shape[1:])
	print('Adding x3b and x4...')
	x5 = add_residuals(x3b,x4)
	x5 = conv_block(x5, filters=128)
	
	#conv6 block
	x6 = conv_block(x5, filters=128)
	
	#conv7 block + MaxPool
	print('x5 shape: ', x5.shape[1:])
	print('x6 shape: ', x6.shape[1:])
	print('Adding x5 and x6...')
	x7 = add_residuals(x5,x6)
	x7 = MaxPooling2D((2,2))(x7)
	x7 = conv_block(x6, filters=256)
	
	#conv8 block + UpSampling
	x8 = UpSampling2D((1,1))(x7)
	x8 = conv_block(x8, filters=128)
	
	#conv9 block + SKIP connection
	print('x6 shape: ', x6.shape[1:])
	print('x8 shape: ', x8.shape[1:])
	print('Adding x6 and x8...')
	x9 = add_residuals(x6,x8)
	x9 = conv_block(x9, filters=128)
	
	#conv10 block
	x10 = conv_block(x9, filters=128)
	
	#conv11 block + UpSampling + residual connection.
	print('x9 shape: ', x9.shape[1:])
	print('x10 shape: ', x10.shape[1:])
	print('Adding x9 and x10...')
	x11 = add_residuals(x9,x10)
	x11 = UpSampling2D((2,2))(x11)
	x11 = conv_block(x11, filters=64)
	
	#conv12 block + SKIP
	print('x3 shape: ', x3.shape[1:])
	print('x11 shape: ', x11.shape[1:])
	print('Adding x3 and x11...')
	x12 = add_residuals(x3,x11)
	x12 = conv_block(x12, filters=64)
	
	#conv13 block
	x13 = conv_block(x12, filters=64)
	
	#conv14 block + residual connection
	x14 = add_residuals(x12,x13)
	x14 = conv_block(x14, filters=64)
	
	#conv15 block
	x15 = conv_block(x14, filters=64)
	
	#conv16 + residual
	x16 = add_residuals(x14,x15)
	out = conv_block(x16, filters=1, 
	                 activation='linear',
	                 batchnorm=False)
	
	#Define model
	model = Model(inputs=x0, outputs=out)
	
	return model
	
	
def run_model(X_train, y_train, X_val, y_val):
	hist = model.fit(X_train, y_train, 
	                 epochs=epochs,
	                 batch_size=batch_size,
	                 validation_data = (X_val, y_val),
	                 callbacks = callbacks_list,
	                 shuffle = True)
                  
    
	#train_datagen = prepare_batch(signal_train, cmb_train, batch_size=batch_size)
	#val_datagen = prepare_batch(signal_val, cmb_val, batch_size=batch_size)
  								  
	#hist = model.fit_generator(generator=train_datagen,
 	#                           validation_data=val_datagen,
 	#                           validation_steps=len(signal_val)/batch_size,
 	#                           epochs=epochs,
    #                           steps_per_epoch=len(signal_train)/batch_size,
    #                           callbacks=callbacks_list,
    #                           use_multiprocessing=True,
    #                           workers=8
    #                           )    
        
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
X, y = prepare_data(signal_list, cmb_list)

X_train, X_val, y_train, y_val = train_test_split(X, y,
	                                              test_size=val_size,
	                                              random_state=seed,
	                                              shuffle=True)
	              
print('Begin training...')	                                              
hist = run_model(X_train, y_train, X_val, y_val)
