{\rtf1\ansi\ansicpg1252\cocoartf1504\cocoasubrtf830
{\fonttbl\f0\fnil\fcharset0 LucidaGrande;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww24820\viewh14920\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #TODO: Test if download.sh works. It maybe work.\
#TODO: Test if training with Monkaa dataset in SceneFlow works\
#TODO: Test if highway net and linear score work. Yes it works.\
#TODO: Test if test.py works\
#TODO: Test if we can cancel loading weight in command line. Yes We can do it by using python train.py -wpath \'93\'94\
#Pretrained Weight : SceneFlow_Driving_DrivingFinalPass\
\
Preprocessing:\
	We crop training patches with size of 256x256 (different from that in the paper) from training images and normalize each channel.\
\
1) To download Driving dataset, create subdirectories sceneflow/driving in data. Then download and tar driving_final pass and driving_disparity from <here>. You can also run the command \'93sh download.sh\'94, which will create subdirectories and download datasets.\
\
2) Train the model by running \'93python train.py\'94\
 \
3) (Optional) Specify the pretrained weight while by adding -wpath <path to pretrained weight>, or you can set it in train_params.py\
\
To enable training with Monkaa dataset, uncomment the relevant snippet in train.py.\
\
}