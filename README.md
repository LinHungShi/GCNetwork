#This project implements the GCNetwork developed by Kendal, et al(2017)

#TODO: Test if download.sh works. It maybe work.

#TODO: Test if training with Monkaa dataset in SceneFlow works

#TODO: Test if highway net and linear score work. Yes it works.

#TODO: Test if test.py works

#TODO: Test if we can cancel loading weight in command line. Yes We can do it by using python train.py -wpath “”

#Pretrained Weight : SceneFlow_Driving_DrivingFinalPass

Preprocessing:
	We crop training patches with size of 256x256 (different from that in the paper) from training images and normalize each channel.

1) To download Driving dataset, create subdirectories sceneflow/driving in data. Then download and tar driving_final pass and driving_disparity from <here>. You can also run the command “sh download.sh”, which will create subdirectories and download datasets.

2) Train the model by running “python train.py”
 
3) (Optional) Specify the pretrained weight while by adding -wpath <path to pretrained weight>, or you can set it in train_params.py

To enable training with Monkaa dataset, uncomment the relevant snippet in train.py.
