# Geometry and Context Network
This project reimplements the GCNetwork developed by Kendal, et al(2017). Currently, we only train with the Middlebury 2014 dataset for indoor object depth localization.

### Directories:

  code : contains main function (main.py), core function(end_endlearning.py) and helper functions(conv3dTranspose and pfm loader)
  
  data : stores middlebury data
  
  log : stores log file, which is useful for visualization
  
  model : trained model.


### Running code by calling main.py. Arguments for main.py:

  mode : 0 for prediction, 1 for training with existing model, 2 for training with new model
  
  data : path for training data
  
  -wpath : pretrained model path. Provided when mode is 0 for 1.
  
  -bs : batch_size. default = 1
  
  -lr : learning_rate. default = 0.001
  
  -ep : epochs. default = 10
  
  -wspath : model_save_path. used when mode is 1 or 2
  
  -lspath : log_save_path. used when mode is 1 or 2. This is the log file used in Tensorboard.
  
  -vdata : path for validation path
  
  -pspath : file for saving predicted result. Used when mode is 0.

  ex: srun --pty python code/main.py 2 data/mb_data/mb_train.npz \\-mspath model/mb_model/mbModel.h5 -lspath log/mb_log/log -vdata data/mb_data/mb_val.npz --epochs 
  
  Or simply: python code/main.py -mode 2 -data data/mb_data/mb_train.npz -wspath model/mb_model/mbModel.h5 -lspath log/mb_log/log -vdata data/mb_data/mb_val.npz

#### Reference: Kendall, Alex, et al. "End-to-End Learning of Geometry and Context for Deep Stereo Regression." arXiv preprint arXiv:1703.04309 (2017).
