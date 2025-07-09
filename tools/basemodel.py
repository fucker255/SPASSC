import torch
import pytorch_lightning as pl

class LightningBaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def configure_optimizers(self):
        # 优先使用 optim_wrapper
        if 'optim_wrapper' in self.config:
            opt_cfg = self.config['optim_wrapper']['optimizer']
            base_lr = self.config['optim_wrapper']['optimizer']['lr']

            # paramwise_cfg 支持
            paramwise_cfg = self.config['optim_wrapper'].get('paramwise_cfg', {})
            custom_keys = paramwise_cfg.get('custom_keys', {}) if paramwise_cfg else {}

            params = []
            max_lrs = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                group = {'params': [param]}
                lr = base_lr
                # 设置不同的学习率倍数
                for key, cfg in custom_keys.items():
                    if key in name:
                        lr = base_lr * cfg.get('lr_mult', 1.0)
                        break
                group['lr'] = lr
                params.append(group) 
                max_lrs.append(lr)           
            if opt_cfg['type'] == 'AdamW':
                optimizer = torch.optim.AdamW(
                    params,
                    lr=opt_cfg['lr'],
                    weight_decay=opt_cfg.get('weight_decay', 0.0)
                )
            else:
                raise NotImplementedError(f"Unsupported optimizer type: {opt_cfg['type']}")


        else:
            # 兼容旧配置
            if self.config['optimizer']['type'] == 'AdamW':
                optimizer = torch.optim.AdamW(
                    [param for param in self.model.parameters() if param.requires_grad],
                    lr=self.config['optimizer']['lr'],
                    weight_decay=self.config['optimizer']['weight_decay']
                )
            else:
                raise NotImplementedError

        # 定义多个学习率调度器
        schedulers = []

        # 遍历配置中的调度器
        for scheduler_cfg in self.config.get('lr_schedulers', []):
            if scheduler_cfg['type'] == 'OneCycleLR':
                schedulers.append({
                    'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=max_lrs if isinstance(max_lrs, list) else scheduler_cfg['max_lr'],
                        total_steps=scheduler_cfg['total_steps'],
                        pct_start=scheduler_cfg['pct_start'],
                        cycle_momentum=scheduler_cfg['cycle_momentum'],
                        anneal_strategy=scheduler_cfg['anneal_strategy']
                    ),
                    'interval': scheduler_cfg['interval'],  # 调度器的调用间隔（'step' 或 'epoch'）
                    'frequency': scheduler_cfg['frequency']  # 调度器的调用频率
                })
            elif scheduler_cfg['type'] == 'CosineAnnealingLR':
                schedulers.append({
                    'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=scheduler_cfg['T_max'],
                        eta_min=scheduler_cfg['eta_min']
                    ),
                    'interval': scheduler_cfg['interval'],
                    'frequency': scheduler_cfg['frequency']
                })
            elif scheduler_cfg['type'] == 'ReduceLROnPlateau':
                schedulers.append({
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode=scheduler_cfg['mode'],
                        factor=scheduler_cfg['factor'],
                        patience=scheduler_cfg['patience'],
                        threshold=scheduler_cfg['threshold'],
                        threshold_mode=scheduler_cfg['threshold_mode'],
                        cooldown=scheduler_cfg['cooldown'],
                        min_lr=scheduler_cfg['min_lr']
                    ),
                    'interval': scheduler_cfg['interval'],
                    'frequency': scheduler_cfg['frequency'],
                    'monitor': scheduler_cfg.get('monitor', 'val_loss')  # 监控的指标
                })
            elif self.config['lr_scheduler']['type'] == 'LinearLR':
                schedulers.append({
                'scheduler': torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.config['lr_scheduler']['start_factor'],
                    total_iters=self.config['lr_scheduler']['total_iters']
                ),
                'interval': scheduler_cfg['interval'],
                'frequency': scheduler_cfg['frequency']
                })
            else:
                raise NotImplementedError(f"Unsupported scheduler type: {scheduler_cfg['type']}")

        # 返回优化器和调度器
        return [optimizer],schedulers