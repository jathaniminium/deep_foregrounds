import keras
from keras.layers import Dense, BatchNormalization, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model

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
	            activation='linear', 
	            strides=(1,1), 
	            padding='same')(x)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation(activation)(x)
	
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
	
	
	
def deepCMBNet(input_shape=(5,128,128)):
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
