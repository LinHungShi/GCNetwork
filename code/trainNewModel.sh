srun --pty python code/main.py 2 data/mb_data/data128x256/mb_train128x256.npz  -lspath log/mb_log/log -vdata data/mb_data/data128x256/mb_val128x256.npz --epochs 5 -bs 2
