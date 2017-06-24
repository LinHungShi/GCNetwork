import glob, re
import numpy as np
import cv2
import random
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', required = True, help = 's for scene_flow, k for kitti dataset')
parser.add_argument('--ldata', required = True, help = 'data path for left image')
parser.add_argument('--rdata', required = True, help = 'data path for right image')
parser.add_argument('--dispdata', required = True, help = 'data path for disparity')
parser.add_argument('--width', help = 'cropped image width', default = 512, type = int)
parser.add_argument('--height', help = 'cropped image height', default = 256, type = int)
parser.add_argument('--save_path', help = 'path for saving file', default = 'images.npz')
args = parser.parse_args()

def _normalizeImg_(input):
	output = np.array(input, dtype = np.float16)
	return (output / 128) - 1

def _normalizeDisp_(input):
	#disp = np.array(input, dtype = np.float16)
	#return disp/np.max(disp)
	return input

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().rstrip()
    if header == 'PF':
    	color = True    
    elif header == 'Pf':
    	color = False
    else:
    	raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
    	width, height = map(int, dim_match.groups())
    else:
    	raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
    	endian = '<'
    	scale = -scale
    else:
    	endian = '>' # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

def _crop_(img, start_h, length_h, start_w, length_w):
	end_h = start_h + length_h
	end_w = start_w + length_w
	if len(img.shape) == 3:
		return img[:, start_h:end_h, start_w:end_w]
	elif len(img.shape) == 2:
		return img[start_h:end_h, start_w:end_w]
	else:
		raise Exception('Dimension Error, should have two or three dimensions with 3 at first or final dimension, but get {}'.format(img.shape))
def _extractImage2_(left_images, right_images, disp_images, transpose = None, crop_h = None, crop_w = None, gt_pfm = False):
	if not len(left_images) == len(right_images) or not len(right_images) == len(disp_images):
		raise Exception('Length should be the same for three images, but get {}, {}, {}'.format(len(left_images), len(right_images), len(disp_images)))
	index = 0
	num_images = len(left_images)
	left_images.sort()
	right_images.sort()
	disp_images.sort()
	for lpath, rpath, gtpath in zip(left_images, right_images, disp_images):
		limage = cv2.imread(lpath)
		rimage = cv2.imread(rpath)
		if gt_pfm == False:
			gtimage = cv2.cvtColor(cv2.imread(gtpath), cv2.COLOR_BGR2GRAY)
		else:	
			gtimage = load_pfm(open(gtpath))
		if transpose:
			limage = np.transpose(limage, transpose)
			rimage = np.transpose(rimage, transpose)
			lc,lh,lw = limage.shape
			rc,rh,rw = rimage.shape
			gh,gw = gtimage.shape
		else:
                        lc,lh,lw = limage.shape
                        rc,rh,rw = rimage.shape
			gh,gw = gtimage.shape
		if not (lc,lh,lw)==(rc,rh,rw):
			raise Exception('Dimension Error, left and right should have the same dimension, but get {, and {}}'.format(limage.shape, rimage.shape))	
		if not (gh, gw) == (lh,lw):
			raise Exception('left image width and height are not compatible with disparity map, get{}, {}'.format(limage.shape[1:], gtimage.shape))
		if crop_h == None:
			crop_h = lh
		if crop_w == None:
			crop_w = lw
		if index == 0:
			limages = np.zeros((num_images, lc, crop_h, crop_w))
			rimages = np.zeros((num_images, rc, crop_h, crop_w))
			gtimages = np.zeros((num_images, crop_h, crop_w))
		max_start_h = lh - crop_h
		max_start_w = lw - crop_w
		start_h = random.randint(0, max_start_h)
		start_w = random.randint(0, max_start_w)
		limage = _crop_(limage, start_h, crop_h, start_w, crop_w)
		rimage = _crop_(rimage, start_h, crop_h, start_w, crop_w)
		gtimage = _crop_(gtimage, start_h, crop_h, start_w, crop_w)
		limage = _normalizeImg_(limage)
		rimage = _normalizeImg_(rimage)
		gtimage = _normalizeDisp_(gtimage)
		limages[index] = limage
		rimages[index] = rimage
		gtimages[index] = gtimage
		index += 1
	return limages, rimages, gtimages

if __name__ == '__main__':
	mode = args.mode
	crop_w = args.width
	crop_h = args.height
	if mode == 'k':
		leftpath = glob.glob(args.ldata + '*_10.png')
		rightpath = glob.glob(args.rdata + '*_10.png')
		disppath = glob.glob(args.dispdata + '*')
		limages, rimages, gtimages = _extractImage2_(leftpath, rightpath, disppath, (2,0,1), crop_h, crop_w)
	else:
		leftpath = glob.glob(args.ldata + '*')
                rightpath = glob.glob(args.rdata + '*')
                disppath = glob.glob(args.dispdata + '*')
                limages, rimages, gtimages = _extractImage2_(leftpath, rightpath, disppath, (2,0,1), crop_h,crop_w, gt_pfm = True)
	save_path = args.save_path
	np.savez(save_path, limages = limages, rimages = rimages, gtimages = gtimages)
	print "Success!"
