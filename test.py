import sys
sys.path.append('src')
import numpy as np
import argparse
import parse_arguments
from gcnetwork import *
import glob
import os
from generator import *
def _predictFromArrays_(model, left, right, bs):
	return model.predict([left, right], bs)

def _predictFromGenerator_(model, generator, steps, max_q_size):
	return model.predict_generator(generator, steps, max_q_size)

if __name__ == "__main__":
	hp, tp, pp, up, env = parse_arguments.parseArguments()
        weight = tp['weight_save_path']
	pspath = pp['pspath']
	parser = argparse.ArgumentParser()
	parser.add_argument('-wpath', help = 'weight path of pretrained model', default = weight)
	parser.add_argument('-pspath', help = 'path for saving prediction result', default = pspath)
	parser.add_argument('-data', help = 'data used for prediction', required = True, default = None)
	parser.add_argument('-bs', help = 'batch size or steps', default = pp['batch_size'])
	args = parser.parse_args()
	weight_path = args.wpath
	pred_path = args.pspath
	ext = up['file_extension']
	model = createGCNetwork(hp)
	model.load_weights(weight_path)
	data_path = args.data
	bs = args.bs
	max_q_size = pp['max_q_size']
	verbose = pp['verbose']
	if data_path.endswith('npz'):
		images = np.load(data_path)
		print "Predict data using arrays"
		pred = _predictFromArrays(model, images[left], images[1], images[2], bs)
		np.save(pspath, pred)
	else:
		q_size = pp['max_q_size']
		left_path = os.path.join(data_path, 'left')
		right_path = os.path.join(data_path, 'right')
		left_images = glob.glob(left_path + "/*.{}".format(ext))
		right_images = glob.glob(right_path + "/*.{}".format(ext))
		generator = generate_arrays_from_file(left_images, right_images, up)
		print "Predict data using generator..."
		pred = model.predict_generator(generator, max_queue_size = max_q_size, steps = bs, verbose = verbose)
		np.save(pspath, pred)
	
