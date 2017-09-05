import argparse
import sys
sys.path.append('src')
from parse_arguments import *
import os
from data_utils import *
from gcnetwork import *
from generator import *
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
        val_ratio = up['val_ratio']
	fraction = up['fraction']
        root = os.path.join(os.getcwd(), sceneflow_root)
        driving = os.path.join(root, driving_root)
        driving_data_path = os.path.join(driving, driving_train)
        driving_label_path = os.path.join(driving, driving_label)
        d_left, d_right, d_disp = genDrivingPath(driving_data_path, driving_label_path)
        m_left = []
        m_right = []
        m_disp = []
        # comment the following lines to train with Monkaa dataset
        '''
        monkaa_root = env['monkaa_root']
        monkaa = os.path.join(root, monkaa_root)
        monkaa_train = env['monkaa_train']
        monkaa_label = env['monkaa_label']
        monkaa_data_path = os.path.join(monkaa, monkaa_train)
        monkaa_label_path = os.path.join(monkaa, monkaa_label)
        m_left, m_right, m_disp = genMonkaaPath(monkaa_data_path, monkaa_label_path)
        '''
        left = d_left + m_left
        right = d_right + m_right
        disp = d_disp + m_disp
        l_imgs, r_imgs, d_imgs = extractAllImage(left, right, disp)
        train, val = splitData(l_imgs, r_imgs, d_imgs, val_ratio, fraction)
        val_generator = generate_arrays_from_file(val[0], val[1], up,val[2])
        train_generator = generate_arrays_from_file(train[0], train[1], up, train[2])
        num_steps = math.ceil(len(train[0]) / batch_size)
        val_steps = math.ceil(len(val[0]) / batch_size)
        model = createGCNetwork(hp)
        if weight_path:
                print 'load pretrained weight'
                model.load_weights(weight_path)
        optimizer = optimizers.RMSprop(lr = lr, rho = rho, epsilon = epsilon, decay = decay)
        model.compile(optimizer = optimizer, loss = loss)
        model.fit_generator(train_generator, validation_data = val_generator, validation_steps = val_steps, steps_per_epoch = num_steps, max_q_size = q_size, epochs = epochs, callbacks = callbacks)
        print "Training Complete"

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

