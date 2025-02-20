import math
import torch
import torch
import torch.nn as nn
# from torch.distributions.normal import Normal
import numpy as np
from layers.AMS_sn import AMS
# from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
# from functools import reduce
# from operator import mul


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.sn_list = configs.sn_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(configs, self.seq_len, self.seq_len, self.num_experts_list[num], k=self.k,
                    num_nodes=self.num_nodes, sn_list=self.sn_list[num], noisy_gating=True, 
                     residual_connection=self.residual_connection)
            )
            
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len, self.pre_len),
            nn.Tanh(),  # 使输出值规范化在-1到1之间
            nn.Linear(self.pre_len, 2 * self.pre_len),
        )

    def forward(self, x):
        x = x[:,:,:,0]      # 取温度数据
        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        # out = self.start_fc(x.unsqueeze(-1))#B、T、N、1 -> B、T、N、d_model

        out = x
        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0,2,1).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).reshape(batch_size, self.num_nodes, self.pre_len, 2).transpose(2, 1)

        mean = out[..., 0]
        std = nn.functional.softplus(out[:,:,:,1]) # 分别获取均值和对数方差
        eps = 1e-6
        std = std + eps
        # var = torch.exp(log_var)  # 方差需要通过指数函数恢复


        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return mean, balance_loss, std


