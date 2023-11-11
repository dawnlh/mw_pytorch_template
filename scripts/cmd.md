## file operations

file transfer: `scp -r -P 22 local_path zzh@166.111.72.183:/remote_path`
file num in current dir: `ls -l | grep "^-" | wc -l`
file_num in current and sub dir: `ls -lR | grep "^-" | wc -l`

## pytorch
tensorboard: `tensorboard --logdir ./ --reload_multifile true --port 6008`

## hardware
GPU monitor: `gpustat -cpu -i 3`