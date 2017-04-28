srun --pty python code/main.py 2 data/mb_data/mb_train.npz -mspath model/mb_model/mbModel.h5 -lspath log/mb_log/log -vdata data/mb_data/mb_val.npz --epochs 1
