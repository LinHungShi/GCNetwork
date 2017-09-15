import argparse
import sys
sys.path.append('src')
from parse_arguments import *
import os
from data_utils import *
from gcnetwork import *
from generator import *
from losses import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import optimizers
import math
import random

def trainSceneFlowData(hp, tp, up, env, callbacks, weight_path = None):
        lr = tp['learning_rate']
        epochs = tp['epochs']
        batch_size = tp['batch_size']
        q_size = tp['max_q_size']
	epsilon = tp['epsilon']
	rho = tp['rho']
	decay = tp['decay']
	loss = tp['loss_function']
        sceneflow_root = env['sceneflow_root']
        driving_root = env['driving_root']
        driving_train = env['driving_train']
        driving_label = env['driving_label']
	train_all = env['train_all']
	train_driving = env['train_driving']
	train_monkaa = env['train_monkaa']
        val_ratio = up['val_ratio']
	fraction = up['fraction']
        root = os.path.join(os.getcwd(), sceneflow_root)
        driving = os.path.join(root, driving_root)
        driving_data_path = os.path.join(driving, driving_train)
        driving_label_path = os.path.join(driving, driving_label)
        monkaa_root = env['monkaa_root']
        monkaa = os.path.join(root, monkaa_root)
        monkaa_train = env['monkaa_train']
        monkaa_label = env['monkaa_label']
        monkaa_data_path = os.path.join(monkaa, monkaa_train)
        monkaa_label_path = os.path.join(monkaa, monkaa_label)
	if train_all:
		train_list = [[driving_data_path, driving_label_path, genDrivingPath], [monkaa_data_path, monkaa_label_path, genMonkaaPath]]
	else:
		train_list = []
		if train_driving:
			train_list.append([driving_data_path, driving_label_path, genDrivingPath])
		if train_monkaa:
			train_list.append([monkaa_data_path, monkaa_label_path, genMonkaaPath])
	train_paths = map(lambda x: x[2](x[0], x[1]), train_list)
	agg_train_path = zip(*train_paths)
	left, right, disp = [reduce(lambda x, y: x + y, path) for path in agg_train_path]
        l_imgs, r_imgs, d_imgs = extractAllImage(left, right, disp)
        train, val = splitData(l_imgs, r_imgs, d_imgs, val_ratio, fraction)
        val_generator = generate_arrays_from_file(val[0], val[1], up,val[2])
        train_generator = generate_arrays_from_file(train[0], train[1], up, train[2])
        num_steps = math.ceil(len(train[0]) / batch_size)
        val_steps = math.ceil(len(val[0]) / batch_size)
        model = createGCNetwork(hp, tp)
        optimizer = optimizers.RMSprop(lr = lr, rho = rho, epsilon = epsilon, decay = decay)
        model.compile(optimizer = optimizer, loss = loss, metrics = [lessOneAccuracy, lessThreeAccuracy])
        model.fit_generator(train_generator, validation_data = val_generator, validation_steps = val_steps, steps_per_epoch = num_steps, max_q_size = q_size, epochs = epochs, callbacks = callbacks)
	print "Training Complete"
        cost = model.layers[-2].output
        cost_volume_weight_save_path = tp['cost_volume_weight_save_path']
        cost_model = Model(model.input, cost)
        cost_model.save_weights(cost_volume_weight_save_path)
	print "Save cost weight"
        if hp['output'] == 'linear':
                linear_output_weight_save_path = tp['linear_output_weight_save_path']
                linear_output = model.layers[-1]
                b, m, h, w = K.int_shape(cost)
                linear_input = Input((m, h, w))
                linear_model = Model(linear_input, linear_output(linear_input))
                linear_model.save_weights(linear_output_weight_save_path)
		print "Save linear output weight"
def genCallBacks(weight_save_path, log_save_path, save_best_only, period, verbose):
        callback_tb = TensorBoard(log_dir = log_save_path, histogram_freq = 0, write_graph = True, write_images = True)
        callback_mc = ModelCheckpoint(weight_save_path, verbose = verbose, save_best_only = save_best_only, save_weights_only = True, period = period)
        return [callback_tb, callback_mc]

if __name__ == '__main__':
        hp, tp, _, up, env = parseArguments()
        parser = argparse.ArgumentParser()
        parser.add_argument('-wpath', '--weight_path', help = 'weight path for pretrained model', default = tp['weight_path'])
        args = parser.parse_args()
        weight_save_path = tp['weight_save_path']
        log_save_path = tp['log_save_path']
        save_best_only = tp['save_best_only']
        period = tp['period']
        verbose = tp['verbose']
        weight_path = args.weight_path
        callbacks = genCallBacks(weight_save_path, log_save_path, save_best_only, period, verbose)
        trainSceneFlowData(hp, tp, up, env, callbacks, weight_path = weight_path)

