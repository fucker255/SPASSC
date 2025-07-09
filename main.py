import os
import misc
import torch
from mmcv import Config
import pytorch_lightning as pl
from argparse import ArgumentParser
from tools.pl_model import pl_model
from tools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from ray import tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
        
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

def train_tune(config_tune, base_cfg, num_gpu, profiler):
    os.chdir('/tmp/algorithm/spassc')

    # 更新超参数
    base_cfg.lr_schedulers[0]['max_lr'] = config_tune["learning_rate"]
    base_cfg.optim_wrapper['optimizer']["lr"] = config_tune["learning_rate"]
    # base_cfg.optim_wrapper['optimizer']["weight_decay"] = config_tune["weight_decay"]
    # base_cfg.lr_schedulers[0]["pct_start"] = config_tune["pct_start"]
    # base_cfg.lr_schedulers[0]["cycle_momentum"] = config_tune["cycle_momentum"]
    base_cfg.model['occ_encoder_backbone']['far_aggregator']["far_encoder_backbone"]['num_layers'] = config_tune["num_layers"]

    model = pl_model(base_cfg)
    data_dm = DataModule(base_cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        save_last=True,
        filename='best'
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="/tmp/model/both",
        name='tune'
    )

    trainer = pl.Trainer(
        devices="auto",  # 这里用整数
        accelerator='auto',
        strategy=RayDDPStrategy(find_unused_parameters=False),
        callbacks=[RayTrainReportCallback(), checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        plugins=[RayLightningEnvironment()],
        max_steps=base_cfg.training_steps,
        logger=tb_logger,
        profiler=profiler,
        sync_batchnorm=True,
        log_every_n_steps=base_cfg['log_every_n_steps'],
        check_val_every_n_epoch=base_cfg['check_val_every_n_epoch'],
        gradient_clip_val=35,
        gradient_clip_algorithm="norm"
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model=model, datamodule=data_dm)
    
from ray.tune.callback import Callback
class EarlyStopMiOUCallback(Callback):
    def __init__(self, threshold=0.17, check_step=1000):
        self.threshold = threshold
        self.check_step = check_step

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # 假设 result 里有 'training_iteration' 和 'val/bestmIoU'
        if result.get("training_iteration", 0) >= self.check_step:
            if result.get("val/bestmIoU", 0) < self.threshold:
                print(f"Trial {trial} stopped early: bestmIoU={result.get('val/bestmIoU', 0)} < {self.threshold}")
                tune.request_stop(trial)

if __name__ == '__main__':
    args, config = parse_config()
    log_folder = os.path.join('logs', config['log_folder'])
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    model = pl_model(config)
    
    data_dm = DataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        save_last=True,
        filename='best')
    
    if not config.eval:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            max_steps=config.training_steps,
            resume_from_checkpoint=None,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step')
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch'],
            gradient_clip_val=35,  # 对应 max_norm
            gradient_clip_algorithm="norm"  # 对应 norm_type
        )
        trainer.fit(model=model, datamodule=data_dm)
    else:
    # if config.eval:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            logger=tb_logger,
            profiler=profiler
        )
        trainer.test(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])

# ray tune

    # import ray
    # ray.init()
    # search_space = {
    #     "learning_rate": tune.grid_search([3e-4]),
    #     # "cycle_momentum": tune.choice([True, False]),
    #     # "weight_decay": tune.uniform(0.02, 0.08),
    #     # "pct_start": tune.uniform(0.07, 0.2),
    #     # "loss_depth_weight": tune.uniform(0.3, 0.8),
    #     "num_layers": tune.grid_search([4, 5]),
    # }
    # from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    # scaling_config = ScalingConfig(
    #     num_workers=8, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
    # )

    # run_config = RunConfig(
    #     checkpoint_config=CheckpointConfig(
    #         num_to_keep=2,
    #         checkpoint_score_attribute="val/bestmIoU",
    #         checkpoint_score_order="max",
    #     ),
    #     callbacks=[EarlyStopMiOUCallback(threshold=0.17, check_step=8000)]
    # )
    # from ray.train.torch import TorchTrainer

    # # Define a TorchTrainer without hyper-parameters for Tuner
    # ray_trainer = TorchTrainer(
    #     tune.with_parameters(train_tune, base_cfg=config, num_gpu=num_gpu, profiler=profiler),
    #     scaling_config=scaling_config,
    #     run_config=run_config,
    # )
    # analysis = tune.Tuner(
    #     ray_trainer,
    #     param_space={"train_loop_config": search_space},
    #     tune_config=tune.TuneConfig(
    #         metric="val/bestmIoU",
    #         mode="max",
    #         num_samples=2,
    #     ),
    # )
    # results = analysis.fit()
    # results.get_best_result(metric="val/bestmIoU", mode="max")



