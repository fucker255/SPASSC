
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--eval \
--ckpt_path /tmp/model/semantickitti_base/tensorboard/version_0/checkpoints/best.ckpt \
--config_path configs/semantickitti_base.py \
--log_folder /tmp/model/semantickitti_base_eval \
--seed 7240 \
--log_every_n_steps 100 