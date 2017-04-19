import sys
from keras.models import Model, load_model
from keras import optimizers
from matplotlib import pyplot as plt
import cv2
import numpy as np
model_path = sys.argv[1]
lr = float(sys.argv[2])
epochs = int(sys.argv[3])
left_img_path = sys.argv[4]
right_img_path = sys.argv[5]
gt_img_path = sys.argv[6]
model_save_path = sys.argv[7]
pred_img_name = sys.argv[8]
model = load_model(model_path)
model.compile(optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
              loss='mean_absolute_error',
              metrics=['mae'])
limage = cv2.resize(cv2.imread(left_img_path), (512,256))
rimage = cv2.resize(cv2.imread(right_img_path), (512,256))
gtimage = cv2.resize(cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE), (512,256))
limage = np.expand_dims(limage, 0)
rimage = np.expand_dims(rimage, 0)
gtimage = np.expand_dims(gtimage, 0)
limage = np.transpose(limage, (0,3,1,2)).astype(np.float32)
rimage = np.transpose(rimage, (0,3,1,2)).astype(np.float32)
model.fit([limage, rimage], gtimage, batch_size = 1, epochs = epochs)
test = model.predict([limage, rimage])
np.savez('pred_disp.npz', images = test)
model.save(model_save_path)
