from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D, Conv3D, Conv3DTranspose, ZeroPadding2D, ZeroPadding3D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Layer
from keras import backend as K
from keras.layers import Input, Dense
from keras import layers, optimizers
from keras.layers import add
from keras.layers.core import Lambda, Permute, Reshape
import numpy as np
from keras import regularizers
import cv2
print "Read all necessary libraries"
# Parameters
## H : Height of the image
## W : Width of the image
## C : Input Channel
## D : Maximum disparities
## data_format : The input shape for the first layer
## first_k_size : kernel size for the first layer
## k_size : kernel size for the remaining layers
## ds_stride : down-sampling stride
## norm_stride : stride for majority layers
dr = 0
H = 256
W = 512
C = 3
D = 192
base_num_filters = 32
data_format = 'channels_first'
input_shape = (H,W,C)
first_k_size = 5
k_size = 3
ds_stride = 2
norm_stride = 1

# Custom layer: Compute the uniary features for left image and right image
# inputs : a length of 2 list, inputs[0] is the left_feature maps where inputs[1] is the right feature maps
# The dimension of lfeatures and rfeatures is Height x Width x Features
def myWrapper(inputs, D):
    def featureConcat(lf, states):
        b,f,h,w = lf.get_shape().as_list()
        rf = states[0]
        rfs = rf[:, :, :, 1:]
        disp_rfs = K.spatial_2d_padding(rfs, padding = ((0, 0), (0, 1)), data_format = 'channels_first')
        concat = K.concatenate([lf, rf], axis = 2)
        output = K.reshape(concat, (-1, 2*f, h, w))
        return output, [disp_rfs]
    left_feature = inputs[0]
    right_feature = inputs[1]    
    left_feature = K.expand_dims(left_feature, axis = 1)
    left_feature = K.repeat_elements(left_feature, D, axis = 1)
    l,o,n = K.rnn(featureConcat, inputs = left_feature, initial_states = [right_feature], unroll = True)
    return K.permute_dimensions(o, (0,2,1,3,4));
# Custom layer: Compute the uniary features for left image and right image
# inputs : a length of 2 list, inputs[0] is the left_feature maps where inputs[1] is the right feature maps
# The dimension of lfeatures and rfeatures is Height x Width x Features
# The Output dimension is Disparity x Height x Width x Features
def CVLayer2(inputs, D):
    lfeatures = inputs[0]
    rfeatures = inputs[1]
    #assert(D < W), 'Maximum disparities must be smaller than the width of the image'
    output = []
    for disparity in np.arange(D):
        rfs = rfeatures[:, :, :, disparity:]
        disp_rfs = K.spatial_2d_padding(rfs, padding = ((0, 0), (0, disparity)), data_format = 'channels_first')
        output.append(K.concatenate([lfeatures, disp_rfs], axis = 1))
    return K.stack(output,axis = 2)

## Custom Layer: CV (Cost Volume) is a 3D, D by H by W matrix, where D represents number of disparities, H and W are 
## the height and  width of each regularized cost volume.
def softArgMin2(cv, D, H, W):
    permuted = K.permute_dimensions(cv, (0,2,3,1))
    softmax = K.softmax(permuted)
    disp_mat = K.expand_dims(K.arange(D, dtype = 'float32'), axis = 0)
    softmax = K.permute_dimensions(softmax, (0,1,3,2))
    output = K.dot(disp_mat, softmax)
    return K.squeeze(output, axis = 0)

def createDeconvLayer(inputs,
                filters, 
                ker_size,
                stride,
                data_format = 'channels_first',
                padding = 'same',
                batch_norm = True,
                act_func = True,
               ):          

    y = Conv3DTranspose(filters,
                        first_k_size,
                        strides = stride,
                        padding = padding,
                        data_format = data_format,
                        )
    shape = inputs.get_shape()
    output_shape = y.compute_output_shape(shape)
    x = y(inputs)
    if batch_norm:
        x = BatchNormalization()(x)
    if act_func:
        x = Activation('relu')(x)
    return x, output_shape

def createConvLayer (inputs,
                filters, 
                ker_size,
                stride,
                data_format = 'channels_first',
                padding = 'same',
                D_conv = 2, 
                input_shape = None,
                batch_norm = True,
                act_func = True,
               ):
    assert(D_conv == 2 or D_conv == 3), 'D_conv must be 2 or 3'
    if input_shape == None:
        if D_conv == 2:
            x = Conv2D(filters,
                       ker_size,
                       strides = stride,
                       data_format = data_format,
                       padding = padding,
                      )(inputs)
        elif D_conv == 3:
            x = Conv3D(filters,
                       ker_size,
                       strides = stride,
                       data_format = data_format,
                       padding = padding,
                      )(inputs)

    else:
        if D_conv == 2:
            x = Conv2D(base_num_filters,
                       first_k_size,
                       strides = stride,
                       padding = padding,
                       data_format = data_format,
                       )(inputs)
        elif D_conv == 3:
              x = Conv3D(base_num_filters,
                       first_k_size,
                       strides = stride,
                       padding = padding,
                       data_format = data_format,
                       )(inputs)                 
    if batch_norm:
        x = BatchNormalization()(x)
    if act_func:
        x = Activation('relu')(x)
    return x
    
def createUniFeature(inputs):
    conv1 = ZeroPadding2D(padding = (2,2), data_format = data_format)(inputs)
    conv1 = createConvLayer(conv1, base_num_filters, first_k_size, ds_stride, padding = 'valid')
    conv2 = createConvLayer(conv1, base_num_filters, k_size, norm_stride)
    conv3 = createConvLayer(conv2, base_num_filters, k_size, norm_stride)
    add4 = add([conv2, conv3])
    conv5 = createConvLayer(add4, base_num_filters, k_size, norm_stride)
    conv6 = createConvLayer(conv5, base_num_filters, k_size, norm_stride)
    add7 = add([add4, conv6])
    conv8 = createConvLayer(add7, base_num_filters, k_size, norm_stride)
    conv9 = createConvLayer(conv8, base_num_filters, k_size, norm_stride)
    add10 = add([add7, conv9])
    conv11 = createConvLayer(add10, base_num_filters, k_size, norm_stride)
    conv12 = createConvLayer(conv11, base_num_filters, k_size, norm_stride)
    add13 = add([add10, conv12])
    conv14 = createConvLayer(add13, base_num_filters, k_size, norm_stride)
    conv15 = createConvLayer(conv14, base_num_filters, k_size, norm_stride)
    add16 = add([add13, conv15])
    conv17 = createConvLayer(add16, base_num_filters, k_size, norm_stride)
    conv18 = createConvLayer(conv17, base_num_filters, k_size, norm_stride)
    add19 = add([add16, conv18])
    conv20 = createConvLayer(add19, base_num_filters, k_size, norm_stride)
    conv21 = createConvLayer(conv20, base_num_filters, k_size, norm_stride)
    add22 = add([add19, conv21])
    conv23 = createConvLayer(add22, base_num_filters, k_size, norm_stride)
    conv24 = createConvLayer(conv23, base_num_filters, k_size, norm_stride)
    add25 = add([add22, conv24])
    conv26 = createConvLayer(add25, base_num_filters, k_size, norm_stride, batch_norm = False, act_func = False)
    return conv26    



def LearnReg(input):
    conv1 = createConvLayer(input, base_num_filters, k_size, norm_stride, D_conv = 3)
    conv2 = createConvLayer(conv1, base_num_filters, k_size, norm_stride, D_conv = 3)
    conv3 = ZeroPadding3D(padding = (1,1,1), data_format = data_format)(conv2)
    conv3 = createConvLayer(conv3, 2*base_num_filters, k_size, ds_stride, padding = 'valid', D_conv = 3)
    conv4 = createConvLayer(conv3, 2*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv5 = createConvLayer(conv4, 2*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv6 = ZeroPadding3D(padding = (1,1,1), data_format = data_format)(conv5)
    conv6 = createConvLayer(conv6, 2*base_num_filters, k_size, ds_stride, padding = 'valid', D_conv = 3)
    conv7 = createConvLayer(conv6, 2*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv8 = createConvLayer(conv7, 2*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv9 = ZeroPadding3D(padding = (1,1,1), data_format = data_format)(conv8)    
    conv9 = createConvLayer(conv9, 2*base_num_filters, k_size, ds_stride, padding = 'valid', D_conv = 3)
    conv10 = createConvLayer(conv9, 2*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv11 = createConvLayer(conv10, 2*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv12 = ZeroPadding3D(padding = (1,1,1), data_format = data_format)(conv11)
    conv12 = createConvLayer(conv12, 4*base_num_filters, k_size, ds_stride, padding = 'valid', D_conv = 3)
    conv13 = createConvLayer(conv12, 4*base_num_filters, k_size, norm_stride, D_conv = 3)
    conv14 = createConvLayer(conv13, 4*base_num_filters, k_size, norm_stride, D_conv = 3)
    deconv15, output_shape15 = createDeconvLayer(conv14, 2 * base_num_filters, k_size, ds_stride)    
    add16 = add([deconv15, conv11])
    deconv17, output_shape17 = createDeconvLayer(add16, 2 * base_num_filters, k_size, ds_stride)
    add18 = add([deconv17, conv8])
    deconv19,output_shape19 = createDeconvLayer(add18, 2 * base_num_filters, k_size, ds_stride)    
    add20 = add([deconv19, conv5])
    deconv21,output_shape21 = createDeconvLayer(add20, base_num_filters, k_size, ds_stride)    
    add22 = add([deconv21, conv2])
    deconv23,output_shape23 = createDeconvLayer(add22, 1, k_size, ds_stride, batch_norm = False, act_func = False)
    return deconv23 
if __name__ == '__main__':
	print "Main function:..."
	left_img = Input(shape = (C, H, W))
	right_img = Input(shape = (C, H, W))
	left_feature = createUniFeature(left_img)
	right_feature = createUniFeature(right_img)
	cv = Lambda(myWrapper, arguments = {'D':D/2})([left_feature, right_feature])
        #cv = Lambda(CVLayer2, arguments = {'D':D/2})([left_feature, right_feature])
	disp_map = LearnReg(cv)
	disp_map_drop_f = Reshape((D,H,W))(disp_map)
	output = Lambda(softArgMin2, arguments = {'D':D, 'H':H, 'W':W})(disp_map_drop_f)
	model = Model(inputs = [left_img, right_img], outputs = [output])
	print "Model Compile success..."
	model.compile(optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
              loss='mean_absolute_error',
              metrics=['mae'])

	limage = cv2.resize(cv2.imread('left.png'), (512,256))
	rimage = cv2.resize(cv2.imread('right.png'), (512,256))
	gtimage = cv2.resize(cv2.imread('gt.png', cv2.IMREAD_GRAYSCALE), (512,256))
	limage = np.expand_dims(limage, 0)
	rimage = np.expand_dims(rimage, 0)
	gtimage = np.expand_dims(gtimage, 0)
	limage = np.transpose(limage, (0,3,1,2))
	rimage = np.transpose(rimage, (0,3,1,2))
	print "Start training model..."
	model.fit([limage, rimage], gtimage, batch_size = 1, epochs = 10)
	model.save('simpleModel.h5')
	test = model.predict([limage, rimage])
	cv2.imwrite('pred_disp.jpg', test[0])
