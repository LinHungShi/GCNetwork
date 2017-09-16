import cv2
import numpy as np
import glob
import random
from load_pfm import *

def generate_arrays_from_file(lefts, rights, up, disps = None):
        crop_height = up['crop_height']
        crop_width = up['crop_width']
	train = True
	if disps == None:
		disps = np.arange(len(lefts))
		train = False
        while 1:
        	random.seed(up['seed'])
        	for ldata, rdata, ddata in zip(lefts, rights, disps):
			left_image = cv2.imread(ldata)
                        right_image = cv2.imread(rdata)
			if train == True:
                        	disp_image = load_pfm(open(ddata))
                        	h, w = left_image.shape[0:2]
                        	start_w = random.randint(0, w - crop_width)
                        	start_h = random.randint(0, h - crop_height)
                        	if crop_height > 0 and crop_width > 0:
					finish_h = start_h + crop_height
					finish_w = start_w + crop_width
                        		left_image = left_image[start_h:finish_h, start_w:finish_w]
                        	        right_image = right_image[start_h:finish_h, start_w:finish_w]
                        	        disp_image = disp_image[start_h:finish_h, start_w:finish_w]
                        	disp_image = np.expand_dims(disp_image, 0)
                        left_image = _centerImage_(left_image)
                        right_image = _centerImage_(right_image)
                        left_image = np.expand_dims(left_image, 0)
                        right_image = np.expand_dims(right_image, 0)
			if train == True:
                       		yield ([left_image, right_image], disp_image)
			else:
				yield ([left_image, right_image])
		if not train:
			break
					

def _centerImage_(img):
	img = img.astype(np.float32)
	var = np.var(img, axis = (0,1), keepdims = True)
        mean = np.mean(img, axis = (0,1), keepdims = True)
        return (img - mean) / np.sqrt(var) 

