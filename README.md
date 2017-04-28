# GCNetwork
# Please use main function to run the model.
# Arguments:
#  mode : 0 for prediction, 1 for training with existing model, 2 for training with new model
#  data : path for training data
#  -mpath : pretrained model path. Provided when mode is 0 for 1.
#  -bs : batch_size. default = 1
#  -lr : learning_rate. default = 0.001
#  -ep : epochs. default = 10
#  -mspath : model_save_path. used when mode is 1 or 2
#  -lspath : log_save_path. used when mode is 1 or 2. This is the log file used in Tensorboard.
#  -vdata : path for validation path
#  -pspath : file for saving predicted result. Used when mode is 0.

# ex: srun --pty python code/main.py 2 data/mb_data/mb_train.npz \\-mspath model/mb_model/mbModel.h5 -lspath log/mb_log/log -vdata data/mb_data/mb_val.npz --epochs 
