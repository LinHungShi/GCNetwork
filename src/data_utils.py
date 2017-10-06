import numpy as np
import cv2
import os
import glob
import random
import math
def genDrivingPath(x, y):
        l_paths = []
        r_paths = []
        y_paths = []
        focal_lengths = ["15mm_focallength", "35mm_focallength"]
        directions = ["scene_backwards", "scene_forwards"]
        types = ["fast", "slow"]
        sides = ["left", "right"]
        for focal_length in focal_lengths:
                for direction in directions:
                        for type in types:
                                l_paths.append(os.path.join(x, *[focal_length, direction, type, sides[0]]))
                                r_paths.append(os.path.join(x, *[focal_length, direction, type, sides[1]]))
                                y_paths.append(os.path.join(y, *[focal_length, direction, type, sides[0]]))
        return l_paths, r_paths, y_paths

def genMonkaaPath(x, y):
        l_paths = []
        r_paths = []
        y_paths = []
        scenes = sorted(os.listdir(x))
        sides = ["left", "right"]
        for scene in scenes:
                        l_paths.append(os.path.join(x, *[scene, sides[0]]))
                        r_paths.append(os.path.join(x, *[scene, sides[1]]))
                        y_paths.append(os.path.join(y, *[scene, sides[0]]))
	return l_paths, r_paths, y_paths

def extractAllImage(lefts, rights, disps):
        left_images = []
        right_images = []
        disp_images = []
        for left_path, right_path, disp_path in zip(lefts, rights, disps):
                left_data = sorted(glob.glob(left_path + "/*.png"))
                right_data = sorted(glob.glob(right_path + "/*.png"))
                disps_data = sorted(glob.glob(disp_path + "/*.pfm"))
                left_images = left_images + left_data
                right_images = right_images + right_data
                disp_images = disp_images + disps_data
        return left_images, right_images, disp_images


def splitData(l, r, d, val_ratio, fraction = 1):
	tmp = zip(l, r, d)
	random.shuffle(tmp)
	num_samples = len(l)
	num_data = int(fraction * num_samples)
	tmp = tmp[0:num_data]
        val_samples = int(math.ceil(num_data * val_ratio))
	val = tmp[0:val_samples]
	train = tmp[val_samples:]
	l_val, r_val, d_val = zip(*val)
	l_train, r_train, d_train = zip(*train)
        return [l_train, r_train, d_train], [l_val, r_val, d_val]
