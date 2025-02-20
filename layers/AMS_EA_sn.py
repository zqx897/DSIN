import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers.Layer import Transformer_Layer
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
import torch.nn.functional as F
import numpy as np
import math

def FFT_for_Period(x, k):
    """
    使用快速傅里叶变换（FFT）寻找数据的主要频率
    :param x: 输入数据，形状为 [Batch, Time, Channels]
    :param k: 要提取的顶部频率的数量
    :return: 主要频率的周期列表，以及相应频率的幅度
    """

    if(x.shape[0] == 0):
        print('x.shape',  x.shape)
        print('x:',x)
    if x.size(1) == 0:
        raise ValueError("Input size for dimension 1 is zero.")

    xf = torch.fft.rfft(x, dim=1)
    # print("输入x形状：", x.shape, 'k:', k)
    # print("开始傅里叶变换")
    # print(xf.shape)
    # print("结束傅里叶变换")
    # FFT的输出解释
    # FFT的输出是一个复数数组，其中每个元素代表一个特定频率的幅度和相位。
    # 在这个数组中，第一个元素（即索引为0的元素）特殊，它代表了信号的直流分量（DC分量）。直流分量是信号的平均值，不随时间变化。
    
    # k = select_frequencies_based_on_energy(xf, threshold=0.8) 
    # k = select_frequencies_by_threshold(xf, threshold=0.6)
    
    # print('k:', k.item())
        # 检查输入张量在指定维度上的大小



    frequency_list = abs(xf).mean(0).mean(-1)   # find period by amplitudes
    # print('frequency_list:', frequency_list.shape, frequency_list[:10])

    frequency_list[0] = float('-inf')
    # print('frequency_list:', frequency_list.shape, frequency_list[:10])
    _, top_list = torch.topk(frequency_list, k)

    top_list = top_list.detach().cpu().numpy()
    # print('top_list:', top_list.shape, top_list)
    period=[1]
    for top in top_list:
        #print(x.shape[1],top,x.shape[1] / top)
        period = np.concatenate((period,[math.ceil(x.shape[1] / top)])) #  
    # print(period.shape,period)  
    return period, abs(xf).mean(-1)[:, top_list] #

class moving_avg(nn.Module):
    """
    Moving average block
    """
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, padding=0) 

    def forward(self, x):
        # batch seq_len channel
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x #batch seq_len channel

class multi_scale_data(nn.Module):
    '''
    多尺度数据拼接模块
    :param kernel_size: 各尺度的核大小
    :param return_len: 返回数据的长度
    '''
    def __init__(self, kernel_size,return_len):
        super(multi_scale_data, self).__init__()
        self.kernel_size = kernel_size
        self.max_len = return_len
        self.moving_avg = [moving_avg(kernel) for kernel in kernel_size]
        
    def forward(self, x):
        """
        前向传播函数
        :param x: 输入数据，形状为: [Batch, Input length, Variables]
        :return: 不同尺度拼接后的数据
        """
        # batch seq_len channel
        different_scale_x = []
        for func in self.moving_avg:
            moving_avg = func(x)    # 应用移动平均
            different_scale_x.append(moving_avg)
            #print(moving_avg.shape)
        multi_scale_x=torch.cat(different_scale_x,dim=1)     # 拼接不同尺度的数据
        # ensure fixed shape: [batch, max_len, variables]
        if multi_scale_x.shape[1]<self.max_len: #padding    填充
            padding = torch.zeros([x.shape[0], (self.max_len - (multi_scale_x.shape[1])), x.shape[2]]).to(x.device)
            multi_scale_x = torch.cat([multi_scale_x,padding],dim=1)
        elif multi_scale_x.shape[1]>self.max_len: #trunc    截断
            multi_scale_x = multi_scale_x [:,:self.max_len,:]
        #print(multi_scale_x.shape)
        return multi_scale_x

class nconv(nn.Module):
    """
    正规化卷积模块
    """
    def __init__(self,gnn_type):
        """
        :param gnn_type: 图神经网络的类型
        """
        super(nconv,self).__init__()
        self.gnn_type = gnn_type
    def forward(self,x, A):
        """
        前向传播函数
        :param x: 输入特征
        :param A: 邻接矩阵
        :return: 卷积后的特征
        """
        if self.gnn_type =='time':
            x = torch.einsum('btdc,tw->bwdc',(x,A))     # 时间维度的卷积
        else:
            x = torch.einsum('btdc,dw->btwc',(x,A))     # 节点维度的卷积
        return x.contiguous()   # 返回连续的张量
    
class gcn(nn.Module):
    """
    图卷积网络模块
    """
    def __init__(self,c_in,c_out,dropout,gnn_type,order=2):
        """
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param dropout: Dropout比例
        :param gnn_type: 图神经网络的类型
        :param order: 图卷积的阶数
        """
        super(gcn,self).__init__()
        self.nconv = nconv(gnn_type)    # 正规化卷积模块
        self.gnn_type=gnn_type
        self.c_in = (order+1)*c_in      # (2+1)*d_model=configs.hidden=8
        self.mlp = nn.Linear(self.c_in,c_out)   #c_out": d_model=configs.hidden=8
        self.dropout = dropout
        self.order = order
        self.act = nn.GELU()
    def forward(self,x,a):
        # in: b t dim d_model
        # out: b t dim d_model
        out = [x]   # 初始化输出列表

        # print(x)
        # print(x.shape)
        # print(a)
        # print(a.shape)
        x1 = self.nconv(x,a)         # 应用正规化卷积
        out.append(x1)  # 添加到输出列表
        for k in range(2, self.order + 1):      # 应用正规化卷积
            x2 = self.nconv(x1,a)
            out.append(x2)
            x1 = x2
        h=torch.cat(out,dim=-1)
        h=self.mlp(h)
        h=self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h#.unsqueeze(1)

class FeatureExpander(nn.Module):
    def __init__(self, val):
        super(FeatureExpander, self).__init__()
        self.val = val

    def forward(self, global_x, original_x):
        # 扩展x以匹配[Batch, time, val]
        # print("Shape of global_x:", global_x.shape)
        # print("Shape of original_x:", original_x.shape)
        global_x = global_x.unsqueeze(-1).expand(-1, -1, self.val)  # [Batch, global_hidden]扩展维度至[Batch, global_hidden, val]
        return torch.cat([original_x, global_x], dim=1)  # 在时间维度上拼接

class single_scale_gnn(nn.Module):
    def __init__(self, configs, scale):
        super(single_scale_gnn, self).__init__()
        self.scale_number=scale     # 10    跨尺度邻居的数量
        self.tk=configs.tk # 尺度数
        self.use_tgcn=configs.use_tgcn  #   1   是否使用时间图卷积网络
        self.use_ngcn=configs.use_ngcn  #   1   是否使用节点图卷积网络
        self.init_seq_len = configs.seq_len     #  初始序列长度
        self.pred_len = configs.pred_len    #  预测长度
        self.ln = nn.ModuleList()
        self.channels = configs.enc_in      # default=7,  # 编码器输入大小=实际变量数
        self.individual = configs.individual    # default=False, 是否对每个变量（通道）单独使用一个线性层1
        self.dropout=configs.dropout    
        self.device='cuda:'+str(configs.gpu)
        self.GraphforPre = False                # 是否用于预测的图
        self.tvechidden = configs.tvechidden    # default=1, help='scale vec dim'   # 时间向量的维度
        self.nvechidden = configs.nvechidden
        self.tanh=nn.Tanh()
        self.d_model = configs.hidden           # default=8, help='channel dim'     # 通道维度
        self.start_linear = nn.Linear(1,self.d_model)       # 起始线性层
        self.global_hidden = configs.global_hidden    #全局token维度
        self.dim = 3        #多变量
        self.muti_scale_len = self.init_seq_len+self.init_seq_len# max_len (i.e., multi-scale shape)   最大长度（多尺度形状）
        self.seq_len =  self.muti_scale_len + self.global_hidden
        # print("A:", self.seq_len)
        self.expander = FeatureExpander(self.channels)
        
        # 时间向量
        self.timevec1 = nn.Parameter(torch.randn(self.seq_len, self.tvechidden).to(self.device), requires_grad=True).to(self.device) 
        self.timevec2 = nn.Parameter(torch.randn(self.tvechidden, self.seq_len).to(self.device), requires_grad=True).to(self.device)
        self.tgcn = gcn(self.d_model,self.d_model,self.dropout,gnn_type='time')
        # 节点向量
        self.nodevec1 = nn.Parameter(torch.randn(self.channels,  self.nvechidden).to(self.device), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn(self.nvechidden, self.channels).to(self.device), requires_grad=True).to(self.device)
        self.gconv = gcn(self.d_model,self.d_model,self.dropout,gnn_type='nodes')
        # 层正则化
        self.layer_norm = nn.LayerNorm(self.channels) 
        self.grang_emb_len = math.ceil(self.d_model//4)
        self.graph_mlp = nn.Linear(2*self.tvechidden,self.grang_emb_len)
        self.act = nn.Tanh()
        # 决定序列的维度
        if self.use_tgcn:
            dim_seq = 2*self.d_model
            if self.GraphforPre:
                dim_seq = 2*self.d_model+self.grang_emb_len#2*self.seq_len+self.grang_emb_len
        else:
            dim_seq = 2*self.seq_len   #2*self.seq_len   
        # 最终的线性层
        self.Linear = nn.Linear(dim_seq, 1) # map to intial scale
        # 后面是各种函数，主要用于构造和调整图的连接，以及进行图卷积网络的前向传播。
    def logits_warper_softmax(self,adj,indices_to_remove,filter_value=-float("Inf")):
        """
        对邻接矩阵应用Softmax，同时根据提供的索引移除某些值。
        :param adj: 邻接矩阵
        :param indices_to_remove: 要移除的索引
        :param filter_value: 用于替换移除值的数值，默认为负无穷
        :return: 处理后的邻接矩阵
        """
        adj = F.softmax(adj.masked_fill(indices_to_remove,filter_value),dim=0)
        return adj
    def logits_warper(self,adj,indices_to_remove,mask_pos,mask_neg,filter_value=-float("Inf")):
        """
        对邻接矩阵进行处理，将某些值替换为Softmax的输出，用于生成修改后的邻接矩阵。
        :param adj: 原始邻接矩阵
        :param indices_to_remove: 要处理的索引
        :param mask_pos: 正向掩码
        :param mask_neg: 负向掩码
        :param filter_value: 用于替换的值，默认为负无穷
        :return: 处理后的邻接矩阵
        """
        #print('adj:',adj)
        mask_pos_inverse = ~mask_pos
        mask_neg_inverse = ~mask_neg
        # Replace values for mask_pos rows
        processed_pos =  mask_pos * F.softmax(adj.masked_fill(mask_pos_inverse,filter_value),dim=-1) 
        # Replace values for mask_neg rows
        processed_neg = -1 * mask_neg * F.softmax((1/(adj+1)).masked_fill(mask_neg_inverse,filter_value),dim=-1) 
        # Combine processed rows for both cases
        processed_adj = processed_pos + processed_neg
        return processed_adj
    def add_adjecent_connect(self,mask):
        """
        为邻接矩阵添加相邻连接。
        :param mask: 邻接矩阵的掩码
        :return: 更新后的掩码
        """
        s=np.arange(0,self.seq_len-1) # torch.arange(start=0,end=self.seq_len-1)
        e=np.arange(1,self.seq_len)
        forahead = np.stack([s,e],0)
        back = np.stack([e,s],0)
        all = np.concatenate([forahead,back],1)
        mask[all] = False
        return mask
    def add_cross_scale_connect(self,adj,periods):
        """
        为邻接矩阵添加跨尺度连接。
        :param adj: 邻接矩阵
        :param periods: 周期列表
        :return: 更新后的掩码
        """
        max_L = self.seq_len            # 设置最大长度为序列长度 self.seq_len。
        mask=torch.tensor([],dtype=bool).to(adj.device)     # 创建一个空的布尔型张量 mask，用于记录哪些连接需要被添加到邻接矩阵中。
        k=self.tk           # 表示在每个尺度上最多考虑的邻居节点数量。
        min_total_corss_scale_neighbors = 5 #  number在每个尺度上至少考虑的邻居节点数量。
        start = 0
        end = 0
        for period in periods:
            ls=self.init_seq_len//period # time node number at this scale 计算在当前周期下的时间节点数量。
            # print(f'ls :{ls} = {self.init_seq_len}//{period}')
            end=start+ls #  计算结束索引。
            if end > max_L: #   如果 end 超出了最大长度 max_L，则将其调整为 max_L，并相应地调整 ls。
                end = max_L #
                ls = max_L-start #+
            kp=k//period 
            kp=max(kp,min_total_corss_scale_neighbors)
            kp=min(kp,ls) # prevent kp exceeding ls
            # print(f'kp :{kp} = {k}//{period}')
            mask = torch.cat([mask,adj[:,start:end] < torch.topk(adj[:,start:end], k=kp)[0][..., -1, None]],dim=1) 
            #更新掩码:
            # 对于每个周期，使用 torch.topk 函数找到 adj 中每个节点在当前尺度（即 start:end 范围内）下的 kp 个最强连接。
            # 然后，将 mask 更新为包含这些最强连接的位置。这是通过将 adj 在 start:end 范围内小于这些最强连接值的位置标记为 True 实现的。
            start=end
            if start==max_L:
                break  
        if start<max_L:
            mask=torch.cat([mask,torch.zeros(self.seq_len,max_L-start,dtype=bool).to(mask.device)],dim=1)
        return mask
    def add_cross_var_adj(self,adj):
        """
        为邻接矩阵添加跨变量连接。该方法基于邻接矩阵中的值确定哪些节点应该连接。
        :param adj: 邻接矩阵
        :return: 生成的三个掩码：普通掩码、正向掩码、负向掩码
        """
        k=3  # 设置邻居的数量
        k=min(k,adj.shape[0])   # 确保k不超过邻接矩阵的大小
        # 普通掩码：筛选出邻接矩阵中既不是最大的k个值也不是最小的k个值的元素
        mask = (adj < torch.topk(adj, k=adj.shape[0]-k)[0][..., -1, None]) * (adj > torch.topk(adj, k=adj.shape[0]-k)[0][..., -1, None])
        # 正向掩码：筛选出邻接矩阵中最大的k个值
        mask_pos = adj >= torch.topk(adj, k=k)[0][..., -1, None] 
        # 负向掩码：筛选出邻接矩阵中最小的k个值
        mask_neg = adj <= torch.kthvalue(adj, k=k)[0][..., -1, None]
        return mask,mask_pos,mask_neg
    
    def get_time_adj(self,periods):
        """

        基于时间向量生成时间邻接矩阵，并根据周期调整连接。
        :param periods: 周期列表
        :return: 处理后的邻接矩阵
        """
        # print("Shape of self.timevec1:", self.timevec1.shape)
        # print("Shape of self.timevec2:", self.timevec2.shape)
        adj=F.relu(torch.einsum('td,dm->tm',self.timevec1,self.timevec2))
        mask = self.add_cross_scale_connect(adj,periods)
        mask = self.add_adjecent_connect(mask)
        # 应用softmax来标准化邻接矩阵
        adj = self.logits_warper_softmax(adj=adj,indices_to_remove=mask)
        return adj
    def get_var_adj(self):
        """
        生成变量间的邻接矩阵。
        :return: 变量间邻接矩阵
        """
        adj=F.relu(torch.einsum('td,dm->tm',self.nodevec1,self.nodevec2))
        mask,mask_pos,mask_neg=self.add_cross_var_adj(adj)
        # 调整邻接矩阵
        adj = self.logits_warper(adj,mask,mask_pos,mask_neg)
        return adj
    def get_time_adj_embedding(self,b):
        """
        生成时间邻接矩阵的嵌入表示。
        :param b: 批次大小
        :return: 时间邻接矩阵的嵌入表示
        """
        graph_embedding = torch.cat([self.timevec1,self.timevec2.transpose(0,1)],dim=1) 
        graph_embedding = self.graph_mlp(graph_embedding)
        # 调整形状以匹配输入数据的维度
        graph_embedding = graph_embedding.unsqueeze(0).unsqueeze(2).expand([b,-1,self.channels,-1])
        # 这里首先在第 0 维和第 2 维上增加一个新的维度，将 graph_embedding 的形状从 [192, F] 改变为 [1, 192, 1, F]。
        # 然后，使用 expand 方法重复这个张量，以匹配批次大小 b 和通道数 self.channels。假设 self.channels 为 C，则最终的形状将是 [b, 192, C, F]。
        return graph_embedding
    def expand_channel(self,x):
        # x: batch seq_len dim 
        # out: batch seq dim d_model
        x=x.unsqueeze(-1)
        x=self.start_linear(x)
        return x
    def forward(self, x, sst_attended):
        # x: [Batch, Input, Dim]
        # print("Shape of x:", x.shape)
        periods,_=FFT_for_Period(x,self.scale_number)
        multi_scale_func = multi_scale_data(kernel_size=periods,return_len=self.muti_scale_len)        #batch(32) *return_len(2*96)  
        
        x = multi_scale_func(x)  # Batch 2*seq_len channel
        # print("Shape of x:", x.shape)
        x = self.expander( sst_attended, x) 
        # print("Shape of x:", x.shape)
        x =self.expand_channel(x)   # batch 192 8
        batch_size=x.shape[0]
        x_ = x

        # print('专家第一步x的形状:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())

        if self.use_ngcn:
            gcn_adp =  self.get_var_adj()

            # print('use_ngcn:')
            # print('gcn_adp:', gcn_adp, gcn_adp.shape)
            x = self.gconv(x, gcn_adp)+x    ##########
            # print('gconv输出:', x.shape)
        # print('专家第二步x的形状:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())
        if self.use_tgcn:
            time_adp =  self.get_time_adj(periods)
            # print('use_tgcn:')
            # print('time_adp:', time_adp, time_adp.shape)
            # print("Shape of time_adp:", time_adp.shape)
            x = self.tgcn(x,time_adp)+x
            # print('tgcn 输出:', x.shape)
        # print('专家第三步x的形状:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())


        x = torch.cat([x_ , x],dim=-1)
        # print('专家第四步x的形状:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())
        if self.use_tgcn and self.GraphforPre:
            graph_embedding = self.get_time_adj_embedding(b=batch_size)
            x=torch.cat([x,graph_embedding],dim=-1)
            # print('专家第五（也可能没有）步x的形状:',x.shape)
            # print("Min:", torch.min(x).item())
            # print("Max:", torch.max(x).item())
            # print("Mean:", torch.mean(x).item())
            # print("Std:", torch.std(x).item())
        x = self.Linear(x).squeeze(-1)
        # print('专家第六步x的形状:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())
        x = F.dropout(x,p=self.dropout,training=self.training)
        # print('专家第七步x的形状:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())
        return x[:,:self.init_seq_len,:] # [Batch, init_seq_len(96), variables]


class AMS(nn.Module):
    def __init__(self, configs, input_size, output_size, num_experts, num_nodes, 
                 sn_list, noisy_gating=False, k=2, residual_connection=1):
        super(AMS, self).__init__()
        self.num_experts = num_experts      # num_experts_list[num] 4
        self.output_size = output_size      #seq_len
        self.input_size = input_size        #seq_len
        self.k = k  # 2

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()     #?没用上
        for scale in sn_list:
            self.experts.append(single_scale_gnn(configs, scale))

    
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)        # 96 4
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating        #默认开启，需要去 AMS()中调
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        # 打印调试信息
        # print("clean_values:", clean_values)
        # print("threshold_if_in:", threshold_if_in)
        # print("noise_stddev:", noise_stddev)
        if torch.any(torch.isnan(clean_values)) or torch.any(torch.isnan(threshold_if_in)) or torch.any(torch.isnan(noise_stddev)):
            raise ValueError("Input contains NaN values")

        if torch.any(torch.isinf(clean_values)) or torch.any(torch.isinf(threshold_if_in)) or torch.any(torch.isinf(noise_stddev)):
            raise ValueError("Input contains infinite values")
            
        if torch.any(noise_stddev == 0):
            raise ValueError("noise_stddev should not be zero")

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        #BTN
        # x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        #x.shape:B、T、N
        x = self.start_linear(x).squeeze(-1)
        #x.shape: B、T  (32,96)
        
        clean_logits = x @ self.w_gate  #(32,96)(96,4)
        if torch.any(torch.isnan(clean_logits)):
            eps = 1e-20
            clean_logits += eps 
        if torch.any(torch.isnan(clean_logits)):
            raise ValueError("clean_logits contains NaN  values")
        if torch.any(torch.isinf(clean_logits)):
            raise ValueError("clean_logits contains infinite values")
        # clean_logits:B、num_expert    每个batch对当前层每个专家的权重(32,4)
        if self.noisy_gating and train:     #对应论文公式6
    
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            if torch.any(torch.isnan(clean_logits)):
                raise ValueError("clean_logits contains NaN  values")
            if torch.any(torch.isinf(clean_logits)):
                raise ValueError("clean_logits contains infinite values")
            
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            if torch.any(torch.isnan(clean_logits)):
                raise ValueError("clean_logits contains NaN  values")
            if torch.any(torch.isinf(clean_logits)):
                raise ValueError("clean_logits contains infinite values")
            logits = noisy_logits      
        else:
            logits = clean_logits
        #logits(32,4)
        # calculate topk + 1 that will be needed for the noisy gates
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            raise ValueError("logits contains NaN or infinite values")
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)   #返回一个batch中每个专家(patch尺寸)的使用次数[4]
        return gates + 1e-30, load

    def forward(self, x, sst_attended, loss_coef=1e-2, use_attend = 0):
        if torch.any(torch.isnan(x)):
            raise ValueError("Input x contains NaN values")
        if torch.any(torch.isinf(x)):
            raise ValueError("Input x contains infinite values")

        # x.shape:B、T、N、
        new_x = self.seasonality_and_trend_decompose(x)
        # new_x.shape:B、T、N
        #multi-scale router
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        # print('gates:',gates)
        # print('gates.size:',gates.shape)
        # print('load:',load)
        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        # print('分发前x:')
        # # if expert_inputs[i].size(0) > 0:  # 检查输入是否为空
        # print(f"x")
        # print('x:',x.shape)
        # print("Min:", torch.min(x).item())
        # print("Max:", torch.max(x).item())
        # print("median:", torch.median(x).item())
        # print("Mean:", torch.mean(x).item())
        # print("Std:", torch.std(x).item())
        expert_inputs = dispatcher.dispatch(x)
        # transformer
        # 将分配给第i个专家的数据输入到experts[i]中，取模型返回的预测数据[0]
        expert_outputs = []
        for i in range(self.num_experts):
            # print('专家', i)
            # if expert_inputs[i].size(0) > 0:  # 检查输入是否为空
            # print(f"expert_inputs[{i}]")
            # print('expert_output:',expert_inputs[i].shape)
            # print("Min:", torch.min(expert_inputs[i]).item())
            # print("Max:", torch.max(expert_inputs[i]).item())
            # print("median:", torch.median(expert_inputs[i]).item())
            # print("Mean:", torch.mean(expert_inputs[i]).item())
            # print("Std:", torch.std(expert_inputs[i]).item())

            expert_output = self.experts[i](expert_inputs[i],sst_attended)
            # print("expert_output:")
            # print('expert_output:',expert_output.shape)
            # print("Min:", torch.min(expert_output).item())
            # print("Max:", torch.max(expert_output).item())
            # print("median:", torch.median(expert_output).item())
            # print("Mean:", torch.mean(expert_output).item())
            # print("Std:", torch.std(expert_output).item())
            expert_outputs.append(expert_output)
            # else:
            #     # 使用默认值，例如零张量
            #     expert_outputs.append(torch.zeros_like(expert_inputs[i]))
        output = dispatcher.combine(expert_outputs)
        
        if self.residual_connection:
            output = output + x
        return output, balance_loss





