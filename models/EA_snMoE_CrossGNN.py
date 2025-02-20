import math
import torch
import torch
import torch.nn as nn
# from torch.distributions.normal import Normal
import numpy as np
from layers.AMS_EA_sn import AMS
# from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
# from functools import reduce
# from operator import mul
import torch.nn.functional as F

class GlobalEmbedding(nn.Module):
    def __init__(self, val_dim, time_dim, global_hidden):
        super(GlobalEmbedding, self).__init__()
        self.val_to_one = nn.Linear(val_dim, 1)  # 将val维压缩到1
        self.time_to_global = nn.Linear(time_dim, global_hidden)  # 将Dim投影到global_hidden

    def forward(self, x):
        batch, time, val, dim = x.shape
        # 先处理val维度
        x = x.permute(0, 1, 3, 2)  # 转换维度为[Batch, time, Dim, val]
        x = x.reshape(batch * time * dim, val)  # 为线性层准备数据
        x = self.val_to_one(x)  # 将val维压缩到1
        x = x.view(batch, time, dim, -1)  # 重新组织回[Batch, time, Dim, 1]

        # 处理Dim维度
        x = x.squeeze(-1)  # 移除最后一个维度[Batch, time, Dim]
        x = x.permute(0, 1, 2 )  # 转换维度为[Batch, Dim, time]
        
        # x = nn.Tanh(x)
        x = x.reshape(batch * dim, time)  # 为线性层准备数据
        x = self.time_to_global(x)  # 将Dim投影到global_hidden
        x = x.view(batch, dim, -1)  # 重新组织回[Batch, time, global_hidden]
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, global_hidden):
        super(CrossAttention, self).__init__()
        # 初始化线性变换层
        self.query = nn.Linear(global_hidden, global_hidden)
        self.key = nn.Linear(global_hidden, global_hidden)
        self.value = nn.Linear(global_hidden, global_hidden)

    def forward(self, x):
        # x is assumed to be [batch, dim, global_hidden]
        # 应用线性变换
        query = self.query(x)  # [Batch, Dim, global_hidden]
        key = self.key(x)      # [Batch, Dim, global_hidden]
        value = self.value(x)  # [Batch, Dim, global_hidden]

        # 计算注意力分数
        scores = torch.einsum('bik,bjk->bij', query, key)  # [Batch, Dim, Dim]
        attention = F.softmax(scores, dim=-1)  # Softmax over the last dimension (j)

        # 应用注意力分数到value
        attended = torch.einsum('bij,bjk->bik', attention, value)  # [Batch, Dim, global_hidden]
        return attended

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
        self.channels = configs.enc_in
        self.global_hidden = configs.global_hidden
        self.dim = 3
        self.global_embed = GlobalEmbedding(self.channels, self.seq_len, self.global_hidden)  # 
        self.cross_attention = CrossAttention(self.dim, self.global_hidden)

    def forward(self, x):

        global_features = self.global_embed(x) 
        cross_attended = self.cross_attention(global_features)
        sst_attended = cross_attended[:,0,:] # [b,global_hidden] 


        x = x[:,:,:,0]      # 取温度数据
        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        # out = self.start_fc(x.unsqueeze(-1))#B、T、N、1 -> B、T、N、d_model

        out = x
        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out,sst_attended,use_attend=0)
            
            balance_loss += aux_loss

        out = out.permute(0,2,1).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).reshape(batch_size, self.num_nodes, self.pre_len, 2).transpose(2, 1)
        mean = out[..., 0]
        std = nn.functional.softplus(out[:,:,:,1]) # 分别获取均值和对数方差
        eps = 1e-6
        std = std + eps

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return mean, balance_loss, std


