# Geometry and Context Network
   A keras implementation of GC Network by HungShi Lin(hl2997@columbia.edu). The paper can be found [here](https://arxiv.org/abs/1703.04309).
I do some modifications by adding a linear output function and enable training highway block at the second stage.

### Software Requirement
   tensorflow([install from here](https://www.tensorflow.org/install/)), keras([install from here](https://keras.io/#installation))

### Data used for training model  
   I trained my model with [drivingfinalpass dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), which contains more than 4000 stereo images with 2 epochs.

### Preprocessing
   We crop training patches with size of 256x256 (different from that in the paper) from training images and normalize each channel.

### Download
   Run the following command:
####   
      git clone https://github.com/LinHungShi/GCNetwork.git
   
### Two ways to download driving dataset:  
  1. create subdirectories sceneflow/driving in data, download and tar driving_final pass and driving_disparity from [here](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). 
  2. You can also issue command 
####
      “sh download.sh” 
   which will create subdirectories and download datasets.
   
### Train the model by running:
   runt the following command: 
####
      python train.py
 
### (Optional) Specify the pretrained weight by
   1. Set it in train_params.py
   2. python train.py -wpath <path to the pretrained weight>

##### To enable training with Monkaa dataset, uncomment the relevant snippet in src/train.py.

##### All hyperparameters used for training can be found in src/hyperparams.json

### Reference :
   Kendall, Alex, et al. "End-to-End Learning of Geometry and Context for Deep Stereo Regression." arXiv preprint arXiv:1703.04309 (2017).
