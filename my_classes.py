#Data generator class based on example code from
#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, datapaths, targetpaths, batch_size=32, dim=(128,128), 
                     n_nu=9, n_stokes=1,
		     shuffle=True, feature_scale=False):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.datapaths = datapaths
		self.targetpaths = targetpaths
		self.n_nu = n_nu
		self.n_stokes = n_stokes
		self.shuffle = shuffle
		self.feature_scale = feature_scale
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.datapaths) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of data and target paths.
		datapaths_temp = [self.datapaths[k] for k in indexes]
		targetpaths_temp = [self.targetpaths[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(datapaths_temp, targetpaths_temp)
	
		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.datapaths))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __load_signal(self, datapath):
		'''Load the signal maps as our data'''
		this_signal = np.load(datapath)
	
		#Assume T map only, and reshape.	
		this_signal = this_signal[:,0,:,:].reshape((self.n_stokes*self.n_nu, *self.dim))
	
		#Mean-subtract and normalize each channel.
		if self.feature_scale:
			for j in range(this_signal.shape[0]):
				this_signal[j,:,:] -= this_signal[j,:,:].mean()
				this_signal[j,:,:] /= this_signal[j,:,:].std()
			
		return this_signal
	
	def __load_cmb(self, targetpath):
		'''Load CMB-only maps as our targets'''
		this_cmb = np.load(targetpath)
	
		#Assume we only want T output, and reshape.	
		this_cmb = this_cmb[0,:,:].reshape((self.n_stokes,*self.dim))
		
		return this_cmb

	def __data_generation(self, datapaths_temp, targetpaths_temp):
		'Generates data containing batch_size samples'
		# X : (n_samples, n_nu, *dim)
		# Initialization
		X = np.empty((self.batch_size, self.n_nu, *self.dim))
		y = np.empty((self.batch_size, 1, *self.dim), dtype=int)

		# Generate data
		for i, datapath in enumerate(datapaths_temp):
			# Store sample
			X[i,] = self.__load_signal(datapath)

		# Generate target
		for i, targetpath in enumerate(targetpaths_temp):
			# Store sample
			y[i,] = self.__load_cmb(targetpath)
	
		return X, y



