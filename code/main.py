import argparse
from network_utils import *
import numpy as np
from keras.models import load_model
from gcnetwork import *
from config import *
parser = argparse.ArgumentParser()
parser.add_argument('-mode', help = '0:prediction, otherwise training model', type = int)
parser.add_argument('-data', help = 'Training data or data used for prediction')
parser.add_argument('-wpath', '--weight_path', help = 'weight path for pretrained model', default = None)
parser.add_argument('-bs', '--batch_size', help = 'batch size', type = int, default = 1)
parser.add_argument('-lr', '--learning_rate', help = 'learning rate for gradient descent method', type = float, default = 0.001)
parser.add_argument('-ep','--epochs', help = 'number of epochs for training', type = int, default = 10)
parser.add_argument('-wspath','--weight_save_path', help = 'path for saving trained weight',default = '{epoch:02d}-{val_loss:.2f}.hdf5')
parser.add_argument('-lspath', '--log_save_path', help = 'path for saving log file. Using for Tensorboard visualization', default = 'log')
parser.add_argument('-vdata','--validation_data', help = 'validation data', default = None)
parser.add_argument('-pspath', '--prediction_save_path', help = 'path for saving predicted result')
args = parser.parse_args()
data = np.load(args.data)
limages = data['limages']
rimages = data['rimages']
mode = args.mode
if mode == 0:
	print 'mode is 0, predict the data'
	left = Input((C,H,W))
	right = Input((C,H,W))
	output = createGCNetwork(left, right)
	model = Model([left, right], output)
	weight_path = args.weight_path
	model.load_weights(weight_path)
	prediction = model.predict([limages, rimages], args.batch_size)
	print "prediction save path is ", args.prediction_save_path
	np.savez(args.prediction_save_path, prediction = prediction)
	print "prediction complete"
else:
	gtimages = data['gtimages']
	val_data = None
	model = None
	if not args.validation_data == None:
		vdata = np.load(args.validation_data)
		val_limages = vdata['limages']
		val_rimages = vdata['rimages']
		val_gtimages = vdata['gtimages']
		val_data = [val_limages, val_rimages, val_gtimages]
	callbacks = genCallBacks(args.weight_save_path, args.log_save_path) 
	print 'model will be saved as {}'.format(args.weight_save_path)
	weight_path = args.weight_path
	trainModel(train_data = [limages, rimages, gtimages], callbacks = callbacks, weight_path = weight_path, lr = args.learning_rate, epochs = args.epochs, batch_size = args.batch_size, val_data = val_data)
