import argparse
from end_endlearning import *
import numpy as np
from keras.models import load_model
parser = argparse.ArgumentParser()
parser.add_argument('mode', help = '0:prediction, 1:training with existing model, 2:training with new model', type = int)
parser.add_argument('data', help = 'path for training data or data used for prediction')
parser.add_argument('-mp', '--model_path', help = 'model path for pretrained model', default = None)
parser.add_argument('-bs', '--batch_size', help = 'batch size', type = int, default = 1)
parser.add_argument('-lr', '--learning_rate', help = 'learning rate for gradient descent method', type = float, default = 0.001)
parser.add_argument('-ep','--epochs', help = 'number of epochs for training', type = int, default = 10)
parser.add_argument('-mspath','--model_save_path', help = 'path for saving trained model',default = '{epoch:02d}-{val_loss:.2f}')
parser.add_argument('-lspath', '--log_save_path', help = 'path for saving log file. Using for Tensorboard visualization', default = 'log')
parser.add_argument('-vdata','--validation_data', help = 'path for validation data', default = None)
parser.add_argument('-pspath', '--prediction_save_file', help = 'path for saving predicted result')
args = parser.parse_args()
data = np.load(args.data)
limages = data['limages']
rimages = data['rimages']
mode = args.mode
if mode == 0:
	print 'mode is 0, predict the data'
	model = load_model(args.model_path)
	prediction = predict(model, [limages, rimages])
	np.savez('prediction', args.prediction_save_path)
	print "prediction complete"
elif mode in [1,2]:
	gtimages = data['gtimages']
	val_data = None
	model = None
	if not args.validation_data == None:
		vdata = np.load(args.validation_data)
		print vdata
		val_limages = vdata['limages']
		val_rimages = vdata['rimages']
		val_gtimages = vdata['gtimages']
		val_data = [val_limages, val_rimages, val_gtimages]
	if mode == 1 and args.model_path == None:
		raise Exception('Training with existing model(mode = 1). model path should be provided')
	elif mode == 1:
		model = load_model(args.model_path)
	callbacks = genCallBacks(args.model_save_path, args.log_save_path) 
	trainModel(train_data = [limages, rimages, gtimages], callbacks = callbacks, model = model, lr = args.learning_rate, epochs = args.epochs, batch_size = args.batch_size, val_data = val_data)
else:
	raise Exception('mode should be 0, 1 or 2 only')
