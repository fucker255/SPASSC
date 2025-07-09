from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F

@BACKBONES.register_module()
class Scan1dFormer(BaseModule):
    def __init__(
        self,
        embed_dims=128,
        num_layers=2,
        with_cp=True,
    ):    
        super().__init__()
        self.former=nn.ModuleList([Scan1dFormer_onelayer(embed_dims) for _ in range(num_layers)])
        self.with_cp=with_cp
        
    def forward(self, x):
        for layer in self.former:
            if self.with_cp:
                x=torch.utils.checkpoint.checkpoint(layer,x)
            else:
                x=layer(x)
        return x
    
    
    
    
class Scan1dFormer_onelayer(BaseModule):
    def __init__(
        self,
        embed_dims=128,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.attention = nn.MultiheadAttention(embed_dims, num_heads=8)#参数量偏少不知道一维attention是否合理,参数少但是占显存多，多层甚至需要checkpoint方式训练
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims)
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        device = x.device  # 获取输入张量所在的设备
        B, C, H, W, Z = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(B * W * Z, C, H)  # 合并W和Z到B上

        # 创建mask
        mask = torch.triu(torch.ones(H, H), diagonal=1).bool().to(device)  # 前面的特征不能看到后面的特征
        mask[:H//2, :H//2] = False  # 前50%可以互相看到 False是可参与

        # 一维attention提取H方向的信息
        x = self.norm1(x.permute(2, 0, 1))  # (H, B*W*Z, C)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        attn_output = attn_output.permute(1, 2, 0)  # (B*W*Z, C, H)
        x = x.permute(1,2,0)
        # 残差连接和层归一化
        x = self.norm2(attn_output + x)


        # 前馈神经网络
        ffn_output = self.ffn(x)
        
        # 残差连接和层归一化
        x = ffn_output + x

        x = x.reshape(B, W, Z, C, H).permute(0, 3, 4, 1, 2)  # 恢复原始形状

        return x