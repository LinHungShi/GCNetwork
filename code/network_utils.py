from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from conv3dTranspose import *
import numpy as np
from gcnetwork import *
from config import *
print "Reading all necessary libraries"

def _train_(model, train_data, lr = 0.001, epochs = 10, batch_size = 1, callbacks = None, val_data = None):
        print "Start training model"
	tr_limages, tr_rimages, tr_gtimages = train_data
        if not val_data == None:
		val_limages, val_rimages, val_gtimages = val_data
        	model.fit([tr_limages, tr_rimages], tr_gtimages, batch_size = batch_size, validation_data = ([val_limages, val_rimages], val_gtimages), epochs = epochs, callbacks = callbacks)
	else :
		model.fit([tr_limages, tr_rimages], tr_gtimages, batch_size = batch_size, epochs = epochs, validation_split = SPLIT_RATIO, callbacks = callbacks)
	print "Training Complete" 


def trainModel(train_data, callbacks, weight_path = None, lr = 0.001, epochs = 10, batch_size = 1, val_data = None, resnet = True):
	left_img = Input(shape = (C, H, W))
        right_img = Input(shape = (C, H, W))
        model = createGCNetwork(left_img, right_img)
	if weight_path:
		print 'load pretrained weight'
		model.load_weights(weight_path)
        model.compile(optimizer=optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0),
              		loss='mean_absolute_error',
              		metrics=['mae'])
	_train_(model, train_data, lr, epochs, batch_size, callbacks, val_data = val_data)

def genCallBacks(weight_save_path, log_save_path):
	callback_tb = TensorBoard(log_dir=log_save_path, histogram_freq=0, write_graph=True, write_images=True)
        callback_mc = ModelCheckpoint(weight_save_path, verbose = 1, save_best_only = True, save_weights_only = True, period = 1)
	return [callback_tb, callback_mc]
