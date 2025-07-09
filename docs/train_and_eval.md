## Training Details

We train SPASSC for 21 epochs on 8 NVIDIA A100 GPUs, with a batch size per GPU of 2.  Before starting training, download the corresponding pretrained checkpoint ([efficientnet](https://github.com/pkqbajng/CGFormer/releases/download/v1.0/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth)) and put it under the folder pretrain.

## Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config_path configs/semantickitti_base.py --log_folder /tmp/model/semantickitti_base --seed 7240 --log_every_n_steps 100 
```

The training logs and checkpoints will be saved under the log_folder、

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the ckpts folder.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --eval --ckpt_path /tmp/model/semantickitti_base/tensorboard/version_0/checkpoints/best.ckpt --config_path configs/semantickitti_base.py --log_folder /tmp/model/semantickitti_base_eval --seed 7240 --log_every_n_steps 100
```

## Evaluation with Saving the Results

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --eval --ckpt_path /tmp/model/semantickitti_base/tensorboard/version_0/checkpoints/best.ckpt --config_path configs/semantickitti_base.py --log_folder /tmp/model/semantickitti_base_eval --seed 7240 --log_every_n_steps 100 --save_path pred
```

The results will be saved into the save_path.

## Submission

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--eval --ckpt_path /tmp/model/semantickitti_base/tensorboard/version_0/checkpoints/best.ckpt \
--config_path configs/semantickitti_base.py \
--log_folder /tmp/model/semantickitti_base_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --test_mapping
```

```

```
