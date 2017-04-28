import sys
from keras import backend as K
from keras.models import Model, load_model
from keras import optimizers
from matplotlib import pyplot as plt
import cv2
import numpy as np
from end_endlearning import mean_absolute_error
model_path = sys.argv[1]
images = sys.argv[2]
lr = float(sys.argv[3])
epochs = int(sys.argv[4])
model_save_path = sys.argv[5]
pred_file = sys.argv[6]
batch_size = int(sys.argv[7])
print "Loading model..."
#K.set_learning_phase(1)
model = load_model(model_path)
print "Compiling model..."
model.compile(optimizer=optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0),
              loss=mean_absolute_error,
              metrics=['mae'])
data = np.load(images)
limages = data['limages']
rimages = data['rimages']
if epochs >= 1:
	gtimages = data['gtimages']
	print "Start training model"
	model.fit([limages, rimages], gtimages, batch_size = batch_size, epochs = epochs)
	model.save(model_save_path)
print "Predicting training data..."
prediction = model.predict([limages, rimages],batch_size = batch_size)
#test = prediction[0]
#print "mean absolute error for first image: ", np.mean(abs(test - gtimages[0]))
np.savez(pred_file, prediction = prediction)
print "Prediction complete"
