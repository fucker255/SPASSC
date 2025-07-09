source /opt/conda/bin/activate base


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --config_path configs/semantickitti_base.py --log_folder /tmp/model/semantickitti_base --seed 7240 --log_every_n_steps 100 ;

# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --eval --ckpt_path /tmp/model/semantickitti_base/tensorboard/version_0/checkpoints/best.ckpt --config_path configs/semantickitti_base.py --log_folder /tmp/model/semantickitti_base_eval --seed 7240 --log_every_n_steps 100 --save_path /tmp/model/pred ;

# python /mnt/vepfs/datasets-v2/tmp/wyh/gpu_load.py;