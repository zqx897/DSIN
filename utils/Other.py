import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.fft as fft
from einops import rearrange, reduce, repeat


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # print('self._gates.shape',self._gates.shape)
        # print('self._gates',self._gates)

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)#取gate中不为0的索引，然后按照batch从小到大排序
        # print('sorted_experts.shape',sorted_experts.shape)
        # print('sorted_experts',sorted_experts)
        # print('index_sorted_experts.shape',index_sorted_experts.shape)
        # print('index_sorted_experts',index_sorted_experts)

        _, self._expert_index = sorted_experts.split(1, dim=1)
        # print('self._expert_index.shape',self._expert_index.shape)
        # print('self._expert_index',self._expert_index)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # print('self._batch_index.shape',self._batch_index.shape)
        # print('self._batch_index',self._batch_index)
        
        self._part_sizes = (gates > 0).sum(0).tolist()
        # print('self._part_sizes',self._part_sizes)

        a = self._batch_index.flatten()
        # print('a.shape',a.shape)
        # print('a',a)

        gates_exp = gates[a]
        # print('gates_exp.shape',gates_exp.shape)
        # print('gates_exp',gates_exp)

        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        # print('self._nonzero_gates.shape',self._nonzero_gates.shape)
        # print('self._nonzero_gates',self._nonzero_gates)

    def dispatch(self, inp):
        # print('inp.shape',inp.shape)
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index]
        # print('inp_exp.shape',inp_exp.shape)
        # print('inp_exp',inp_exp)
        a = torch.split(inp_exp, self._part_sizes, dim=0)

        # print('a[0].shape',a[0].shape)
        return a

    # def combine(self, expert_out, multiply_by_gates=True):
    #     # apply exp to expert outputs, so we are not longer in log space
    #     # stitched = torch.clamp(torch.cat(expert_out, 0), max=10).exp()  # 限制最大值为10
    #     cat = torch.cat(expert_out, 0)

    #     print('cat:',cat.shape)
    #     print("Min:", torch.min(cat).item())
    #     print("Max:", torch.max(cat).item())
    #     print("median:", torch.median(cat).item())
    #     print("Mean:", torch.mean(cat).item())
    #     print("Std:", torch.std(cat).item())
    #     # # 绘制原始数据直方图
    #     # plt.figure(figsize=(12, 6))
    #     # plt.subplot(1, 2, 1)
    #     # plt.hist(cat.flatten().cpu().detach().numpy(), bins=50, alpha=0.75)
    #     # plt.title('Histogram of x')



    #     cat = torch.clamp(cat, max=10)
    #     # # 应用指数变换并绘制直方图
    #     # cat_exp = np.exp(cat.cpu().detach().numpy())
    #     # plt.subplot(1, 2, 2)
    #     # plt.hist(cat_exp.flatten(), bins=50, alpha=0.75)
    #     # plt.title('Histogram of exp(x)')
    #     # plt.savefig('Histogram.png')
    #     print("cat:")
    #     print('cat:',cat.shape)
    #     print("Min:", torch.min(cat).item())
    #     print("Max:", torch.max(cat).item())
    #     print("median:", torch.median(cat).item())
    #     print("Mean:", torch.mean(cat).item())
    #     print("Std:", torch.std(cat).item())
    #     stitched = cat.exp()

    #     print("cat.exp():")
    #     print('stitched:',stitched.shape)
    #     print("Min:", torch.min(stitched).item())
    #     print("Max:", torch.max(stitched).item())
    #     print("median:", torch.median(stitched).item())
    #     print("Mean:", torch.mean(stitched).item())
    #     print("Std:", torch.std(stitched).item())


    #     print("Nonzero gates stats:")
    #     print('self._nonzero_gates:',self._nonzero_gates.shape)
    #     print("Min:", torch.min(self._nonzero_gates).item())
    #     print("Max:", torch.max(self._nonzero_gates).item())
    #     print("median:", torch.median(self._nonzero_gates).item())
    #     print("Mean:", torch.mean(self._nonzero_gates).item())
    #     print("Std:", torch.std(self._nonzero_gates).item())

    #     if multiply_by_gates:
    #         stitched = torch.einsum("ijk,ik -> ijk", stitched, self._nonzero_gates)

    #     print("smultiply_by_gates后stitched stats:")
    #     print('smultiply_by_gates后stitched:',stitched.shape)
    #     print("Min:", torch.min(stitched).item())
    #     print("Max:", torch.max(stitched).item())
    #     print("median:", torch.median(stitched).item())
    #     print("Mean:", torch.mean(stitched).item())
    #     print("Std:", torch.std(stitched).item())

    #     print('self._gates:',self._gates.shape)
    #     zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2),
    #                         requires_grad=True, device=stitched.device)
    #     # combine samples that have been processed by the same k experts



    #     combined = zeros.index_add(0, self._batch_index, stitched.float())
    #     print("combined:")
    #     print('combined:',combined.shape)
    #     print("Min:", torch.min(combined).item())
    #     print("Max:", torch.max(combined).item())
    #     print("median:", torch.median(combined).item())
    #     print("Mean:", torch.mean(combined).item())
    #     print("Std:", torch.std(combined).item())

    #     # add eps to all zero values in order to avoid nans when going back to log space
    #     combined[combined == 0] = np.finfo(float).eps
    #     # back to log space
        

    #     return torch.clamp(combined, min=1e-10) # .log()  # 防止取对数前值太小
    #     # return combined

    def combine(self, expert_out, multiply_by_gates=True):
        # Stitch the expert outputs together
        # stitched = torch.cat(expert_out, 0)
        stitched = torch.clamp(torch.cat(expert_out, 0), max=10).exp()  # 限制最大值为10
        if multiply_by_gates:
            stitched = torch.einsum("ijk,ik -> ijk", stitched, self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2),
                            requires_grad=True, device=stitched.device)
        # Combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return torch.clamp(combined, min=1e-10).log()  # 防止取对数前值太小


   
    
   
        
    
                
    
    


    
    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=(1, 1),
                             bias=True)

    def forward(self, x):
        out = self.fc(x)
        return out


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = np.exp(2 * math.pi * 1j / T)
        idft_mat = (np.power(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')