'''
C : an integer, Channel of the image
H : an integer, Height of the image
W : an integer, Width of the image
D : an integer, Maximum disparity
BASE_NUM_FILTERS : an integer, number of filters in all layers are the multiple of this number
FIRST_KERNEL_SIZE : an integer, kernel size for the first layer in GC-net
KERNEL_SIZE : an integer, kernel size for all layers except the first layer
NUM_RES : an integer, number of res-net blocks used in GC-net
NUM_DOWN_CONV : an integer, number of convoltion layer that downsampling feature maps in Learning-Regularization stage. 
DATA_FORMAT : a string, data format for layers. It shouldn't be changed
SPLIT_RATIO : a float number between 0 and 1, ratio of validation data for training
RESNET : a boolean value, using resnet or highway network
OUTPUT : a string, output Function for GCNetwork, choices: linear, softargmin
ACT_FUNC : a string, activation function
HIGHWAY_ACTFUNC : a string, activation function for transform gates in the highway block
'''

C = 3
H = 256 
W = 512
d = 64 
BASE_NUM_FILTERS= 32
FIRST_KERNEL_SIZE = 5
KERNEL_SIZE = 3
NUM_RES = 5
NUM_DOWN_CONV = 3
SPLIT_RATIO = 0.2
RESNET = True
OUTPUT = 'linear'
ACT_FUNC = 'relu'
HIGHWAY_ACTFUNC = 'sigmoid'

