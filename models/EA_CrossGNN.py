import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def FFT_for_Period(x, k=5):
    """
    使用快速傅里叶变换（FFT）寻找数据的主要频率
    :param x: 输入数据，形状为 [Batch, Time, Channels]
    :param k: 要提取的顶部频率的数量
    :return: 主要频率的周期列表，以及相应频率的幅度
    """
    xf = torch.fft.rfft(x, dim=1)
    # print("输入x形状：", x.shape, 'k:', k)
    # print("开始傅里叶变换")
    # print(xf.shape)
    # print("结束傅里叶变换")
    # FFT的输出解释
    # FFT的输出是一个复数数组，其中每个元素代表一个特定频率的幅度和相位。
    # 在这个数组中，第一个元素（即索引为0的元素）特殊，它代表了信号的直流分量（DC分量）。直流分量是信号的平均值，不随时间变化。
    
    # print('k:', k.item())

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

# class GlobalEmbedding(nn.Module):
#     def __init__(self, seq_len, global_hidden):
#         super(GlobalEmbedding, self).__init__()
#         self.fc = nn.Linear(seq_len, global_hidden)  # 假设这里我们只处理一维特征


#     def forward(self, x):
#         # x: [batch, time, features]
#         x = x.mean(dim=1)  # Average over time to create a global embedding
#         return self.fc(x)  # [batch, embed_dim]
    

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
        x = x.permute(0, 1, 2)  # 转换维度为[Batch, Dim, time]
        x = x.reshape(batch * dim, time)  # 为线性层准备数据
        x = self.time_to_global(x)  # 将Dim投影到global_hidden
        x = x.view(batch, dim, -1)  # 重新组织回[Batch, time, global_hidden]
        return x


# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, key_dim, value_dim):
#         super(CrossAttention, self).__init__()
#         self.query = nn.Linear(query_dim, value_dim)
#         self.key = nn.Linear(key_dim, value_dim)
#         self.value = nn.Linear(value_dim, value_dim)

#     def forward(self, query, key, value):
#         # query, key, value are all [batch, spatial, features]
#         scores = torch.einsum('bik,bjk->bij', self.query(query), self.key(key))
#         attention = F.softmax(scores, dim=-1)
#         attended = torch.einsum('bij,bjk->bik', attention, self.value(value))
#         return attended
    

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
    
class FeatureExpander(nn.Module):
    def __init__(self, val):
        super(FeatureExpander, self).__init__()
        self.val = val

    def forward(self, global_x, original_x):
        # 扩展x以匹配[Batch, time, val]
        global_x = global_x.unsqueeze(-1).expand(-1, -1, self.val)  # [Batch, global_hidden]扩展维度至[Batch, global_hidden, val]
        return torch.cat([original_x, global_x], dim=1)  # 在时间维度上拼接

class single_scale_gnn(nn.Module):
    def __init__(self, configs):
        super(single_scale_gnn, self).__init__()
        self.tk=configs.tk      # 10    跨尺度邻居的数量
        self.scale_number=configs.scale_number  # 4 尺度数
        self.use_tgcn=configs.use_tgcn  #   1   是否使用时间图卷积网络
        self.use_ngcn=configs.use_ngcn  #   1   是否使用节点图卷积网络
        self.init_seq_len = configs.seq_len     #  初始序列长度
        self.pred_len = configs.pred_len    #  预测长度
        self.ln = nn.ModuleList()
        self.channels = configs.enc_in       # default=112,  # 编码器输入大小=实际变量数
        self.individual = configs.individual    # default=False, 是否对每个变量（通道）单独使用一个线性层
        self.dropout=configs.dropout    
        self.device='cuda:'+str(configs.gpu)
        self.GraphforPre = False                # 是否用于预测的图
        self.tvechidden = configs.tvechidden    # default=1, help='scale vec dim'   # 时间向量的维度
        self.tanh=nn.Tanh()
        self.d_model = configs.hidden           # default=8, help='channel dim'     # 通道维度
        self.start_linear = nn.Linear(1,self.d_model)       # 起始线性层
        self.global_hidden = configs.global_hidden      #全局token维度
        self.dim = 3        #多变量
        self.muti_scale_len = self.init_seq_len+self.init_seq_len# max_len (i.e., multi-scale shape)   最大长度（多尺度形状）
        self.seq_len =  self.muti_scale_len + self.global_hidden
        
        self.expander = FeatureExpander(self.channels)

        # 时间向量
        self.timevec1 = nn.Parameter(torch.randn(self.seq_len, configs.tvechidden).to(self.device), requires_grad=True).to(self.device) 
        self.timevec2 = nn.Parameter(torch.randn(configs.tvechidden, self.seq_len).to(self.device), requires_grad=True).to(self.device)
        self.tgcn = gcn(self.d_model,self.d_model,self.dropout,gnn_type='time')
        # 节点向量
        self.nodevec1 = nn.Parameter(torch.randn(self.channels,  configs.nvechidden).to(self.device), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn( configs.nvechidden, self.channels).to(self.device), requires_grad=True).to(self.device)
        self.gconv = gcn(self.d_model,self.d_model,self.dropout,gnn_type='nodes')
        # 层正则化
        self.layer_norm = nn.LayerNorm(self.channels) 
        self.grang_emb_len = math.ceil(self.d_model//4)
        self.graph_mlp = nn.Linear(2*self.tvechidden, self.grang_emb_len)
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
        用于时间维度邻接矩阵的生成
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
        用于变量维度邻接矩阵的生成
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
        用于时间维度邻接矩阵的生成
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
        用于时间维度邻接矩阵的生成
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
            end=start+ls #  计算结束索引。
            if end > max_L: #   如果 end 超出了最大长度 max_L，则将其调整为 max_L，并相应地调整 ls。
                end = max_L #
                ls = max_L-start #+
            kp=k//period 
            kp=max(kp,min_total_corss_scale_neighbors)
            kp=min(kp,ls) # prevent kp exceeding ls
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
        用于变量维度邻接矩阵的生成
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
        graph_embedding = torch.cat([self.timevec1, self.timevec2.transpose(0,1)],dim=1) 
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
    def forward(self, x, sst_attended,use_attend = 1):
        


        # x: [Batch, time, val, Dim]                # torch.Size([32, 96, 112, 3])
        # First apply global embeddings across the time dimension
        # print("single_scale_gnn(forward)x.shape:", x.shape)
        

        periods,_=FFT_for_Period(x,self.scale_number)
        multi_scale_func = multi_scale_data(kernel_size=periods,return_len=self.muti_scale_len)        #batch(32) *return_len(2*96)  创建对象

        x = multi_scale_func(x)  # Batch 2*seq_len channel  得到多尺度数据
        # print('拼接前x_sst的维度：',x.shape)    # 拼接前x_sst的维度： torch.Size([32, 192, 112])

        # print('拼接前 sst_attended 的维度：',sst_attended.shape)    # 拼接前 sst_attended 的维度： torch.Size([32, 5])

        x = self.expander( sst_attended, x)        # 将注意力结果拼接到原始数据
        # print('拼接后x_sst的维度：',x.shape)    # 拼接后x_sst的维度： torch.Size([32, 197, 112])

        x =self.expand_channel(x)   # batch 192+global_hidden 8
        # print('扩展后的维度：',x.shape)     # 扩展后的维度： torch.Size([32, 197, 112, 8])
        batch_size=x.shape[0]
        x_ = x
        if self.use_tgcn:
            time_adp =  self.get_time_adj(periods)
            # print('use_tgcn:')
            # print('time_adp:', time_adp, time_adp.shape)

            x = self.tgcn(x,time_adp)+x
            # print('tgcn 输出:', x.shape)
        if self.use_ngcn:
            gcn_adp =  self.get_var_adj()

            # print('use_ngcn:')
            # print('gcn_adp:', gcn_adp, gcn_adp.shape)
            x = self.gconv(x, gcn_adp)+x    ##########
            # print('gconv输出:', x.shape)
        x = torch.cat([x_ , x],dim=-1)
        if self.use_tgcn and self.GraphforPre:
            graph_embedding = self.get_time_adj_embedding(b=batch_size)
            x=torch.cat([x,graph_embedding],dim=-1)
        x = self.Linear(x).squeeze(-1)
        x = F.dropout(x,p=self.dropout,training=self.training)
        # print("end_single_scale_gnn(forward)x.shape:", x.shape)
        return x[:,:self.init_seq_len,:] # [Batch, init_seq_len(96), variables]

class Model(nn.Module):
    '''
    EA_CrossGNN
    '''
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len          # 96
        self.pred_len = configs.pred_len        # 96...
        self.graph_encs = nn.ModuleList()       # 多个single_scale_gnn
        self.enc_layers = configs.e_layers      # 1
        self.anti_ood = configs.anti_ood        # 1
        for i in range(self.enc_layers):        # 1
            self.graph_encs.append(single_scale_gnn(configs=configs))
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.channels = configs.enc_in
        self.global_hidden = configs.global_hidden
        self.dim = 3
        self.global_embed = GlobalEmbedding(self.channels, self.seq_len, self.global_hidden)  # 
        self.cross_attention = CrossAttention(self.dim, self.global_hidden)

    def forward(self, x):
        # x: [Batch, Input length, Variables]
        # 如果启用了防过度拟合设置

        # print("Model:x.shape",x.shape)        Model:x.shape torch.Size([32, 96, 112, 3])
        # print("Model:enc_layers",self.enc_layers)         Model:enc_layers 2

        global_features = self.global_embed(x)  
        # Then apply the cross-attention mechanism
        cross_attended = self.cross_attention(global_features)
        sst_attended = cross_attended[:,0,:] # [b,global_hidden] 
        x_sst = x[:,:,:,0]  # 取温度数据

        if self.anti_ood:
            seq_last = x_sst[:,-1:,:].detach()  # 获取并分离最后一个时间步
            x_sst = x_sst - seq_last    # 从输入中减去最后一个时间步

        # 遍历所有的图编码器
        for i in range(self.enc_layers):
            # print("Model_i:",i,x_sst.shape)
            x = self.graph_encs[i](x_sst, sst_attended,use_attend = 0)   # 应用图编码器

        # 应用线性层，调整输出维度
        pred_x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        if self.anti_ood:
                pred_x = pred_x  + seq_last # 将最后一个时间步加回到预测中
        return pred_x # [Batch, Output length, Variables]