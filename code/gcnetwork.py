from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv3D, Conv2DTranspose
from conv3dTranspose import Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras import backend as K
from keras.layers import Input, add, multiply
from keras.layers.core import Lambda, Permute, Reshape
from ipykernel import kernelapp as app
from config import *

def _resNetBlock_(input_shape, filters = 32, kernel_size = 3, strides = 1, padding = 'same', data_format = 'channels_first', act_func = 'relu'):

    conv1 = Conv2D(filters, kernel_size, strides = strides, input_shape = input_shape, padding = padding, data_format = data_format)
    bn1 = BatchNormalization(axis = 1)
    act1 = Activation(act_func)
    conv2 = Conv2D(filters,kernel_size, strides = strides, padding = padding,data_format = data_format)
    bn2 = BatchNormalization(axis = 1)
    act2 = Activation(act_func)    
    model = Sequential([conv1, bn1, act1, conv2, bn2, act2])
    return model

def _sharedResNet_(inputs, filters = 32, kernel_size = 3, strides = 1, padding = 'same', act_func = 'relu'):
    
    input_shape = K.int_shape(inputs[0]) 
    model = _resNetBlock_(input_shape[1:], filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, act_func = act_func)
    outputs = [add([model(input), input]) for input in inputs]
    return outputs

def _addConv3D_(input, filters = 32, kernel_size = 3, strides = 1, padding = 'same', data_format = 'channels_first', bn = True, act_func = 'relu'):
    
    conv = Conv3D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format)(input)
    if bn:
        conv = BatchNormalization(axis = 1)(conv)
    if act_func:
        conv = Activation(act_func)(conv)
    return conv

def _convDownSampling_(input, filters = 32, kernel_size = 3, ds_strides = 2, strides = 1):

    conv = _addConv3D_(input, filters = filters, kernel_size = kernel_size, strides = ds_strides, padding = 'same')
    conv = _addConv3D_(conv, filters= filters, kernel_size = kernel_size, strides = strides, padding = 'same')    
    conv = _addConv3D_(conv, filters= filters, kernel_size = kernel_size, strides = strides, padding = 'same')    
    return conv

def _createDeconv3D_(input, filters = 32, kernel_size = 3, strides = 1, padding = 'same', data_format = 'channels_first', bn = True, act_func = 'relu'):
    
    deconv = Conv3DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, data_format = data_format, padding = padding) (input)
    if bn:
        deconv = BatchNormalization(axis = 1)(deconv)
    if act_func:
        deconv = Activation(act_func)(deconv)
    return deconv

def _highwayBlock_(tensor):
	output = tensor[0]
	input = tensor[1]
	trans = tensor[2]
	return add([multiply([output, trans]), multiply([input, 1 - trans])])

def _getCostVolume_(inputs, d):

    def featureConcat(lf, states):
        b,f,h,w = lf.get_shape().as_list()
        rf = states[0]
        rfs = rf[:, :, :, :-1]
        disp_rfs = K.spatial_2d_padding(rfs, padding = ((0, 0), (1, 0)), data_format = 'channels_first')
        concat = K.concatenate([lf, rf], axis = 2)
        output = K.reshape(concat, (-1, 2*f, h, w))
        return output, [disp_rfs]
    left_feature = inputs[0]
    right_feature = inputs[1]
    left_feature = K.expand_dims(left_feature, axis = 1)
    left_feature = K.repeat_elements(left_feature, d, axis = 1)
    l,o,n = K.rnn(featureConcat, inputs = left_feature, initial_states = [right_feature], unroll = True)
    return K.permute_dimensions(o, (0,2,1,3,4));

def  _computeLinearScore_(cv, d):

        cv = K.squeeze(cv, axis = 1)
        disp_mat = K.expand_dims(K.arange(d, dtype = 'float32'), axis = 0)
        softmax = K.permute_dimensions(cv, (0,2,1,3))
        output = K.dot(disp_mat, softmax)
        return K.squeeze(output, axis = 0)

def _computeSoftArgMin_(cv, d):

    cv = K.squeeze(cv, axis = 1)
    permuted = K.permute_dimensions(cv, (0,2,3,1))
    softmax = K.softmax(permuted) 
    disp_mat = K.expand_dims(K.arange(d, dtype = 'float32'), axis = 0)
    softmax = K.permute_dimensions(softmax, (0,1,3,2))
    output = K.dot(disp_mat, softmax)
    return K.squeeze(output, axis = 0)

def getOutputFunction(output):

        if output == 'linear':
                return _computeLinearScore_
        if output == 'softargmin':
                return _computeSoftArgMin_

def _createUniFeature_(inputs):

    input_shape = K.int_shape(inputs[0])
    model = Sequential()
    conv1 = Conv2D(filters = BASE_NUM_FILTERS, kernel_size = FIRST_KERNEL_SIZE, strides = 2, padding = 'same', data_format = 'channels_first',input_shape = input_shape[1:])
    bn1 = BatchNormalization(axis = 1)
    act1 = Activation(ACT_FUNC)
    model.add(conv1)
    model.add(bn1)
    model.add(act1)
    outputs = [model(input) for input in inputs]
    for i in range(NUM_RES):
        outputs = _sharedResNet_(outputs, BASE_NUM_FILTERS)
    model2 = Sequential()
    input_shape2 = K.int_shape(outputs[0])
    conv2 = Conv2D(filters = BASE_NUM_FILTERS, kernel_size = KERNEL_SIZE, strides = 1, padding = 'same', data_format = 'channels_first', input_shape = input_shape2[1:])
    model2.add(conv2)
    outputs = [model2(output) for output in outputs]
    return outputs

def _LearnReg_(input):    

    down_convs = list()
    conv = _addConv3D_(input, filters = 2 * BASE_NUM_FILTERS) 
    conv = _addConv3D_(conv, filters = 2 * BASE_NUM_FILTERS)
    down_convs.insert(0, conv)
    if not RESNET:
    	trans_gates = list()
    	gate = _addConv3D_(conv, filters = 2 * BASE_NUM_FILTERS, act_func = HIGHWAY_ACTFUNC)
    	trans_gates.insert(0, gate)
    for i in range(NUM_DOWN_CONV):
	conv = _convDownSampling_(conv, filters = 2 * BASE_NUM_FILTERS)	 	
	down_convs.insert(0, conv)
	if not RESNET:
		gate = _addConv3D_(conv, filters = 2 * BASE_NUM_FILTERS)	
		trans_gates.insert(0, gate)
    up_convs = down_convs[0]
    for i in range(NUM_DOWN_CONV):    
	deconv = _createDeconv3D_(up_convs, 2 * BASE_NUM_FILTERS, strides = 2) 
	if not RESNET:
		up_convs = Lambda(_highwayBlock_)([deconv, down_convs[i+1], trans_gates[i+1]])
        else: 
		up_convs = add([deconv, down_convs[i+1]])
    output = _createDeconv3D_(up_convs, 1, bn = False, act_func = None, strides = 2)
    return output

def createGCNetwork(left_img, right_img):
    left_feature, right_feature = _createUniFeature_([left_img, right_img])
    cv = Lambda(_getCostVolume_, arguments = {'d':d/2})([left_feature, right_feature])
    print "Using resnet in second stage ? {}".format(RESNET)
    disp_map = _LearnReg_(cv)
    out_func = getOutputFunction(OUTPUT)
    output = Lambda(out_func, arguments = {'d':d})(disp_map)
    model = Model([left_img, right_img], output)
    return model

