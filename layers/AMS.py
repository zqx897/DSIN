import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers.Layer import Transformer_Layer
from utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
from models import CrossGNN

class AMS(nn.Module):
    def __init__(self, input_size, output_size, num_experts, device, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size=[8, 6, 4, 2], noisy_gating=True, k=4, layer_number=1, residual_connection=1):
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
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(Transformer_Layer(device=device, d_model=d_model, d_ff=d_ff,
                                      dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                                      patch_size=patch, factorized=True, layer_number=layer_number))

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
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        #x.shape:B、T、N
        x = self.start_linear(x).squeeze(-1)
        #x.shape: B、T  (32,96)
        clean_logits = x @ self.w_gate  #(32,96)(96,4)
        if torch.any(torch.isnan(clean_logits)) or torch.any(torch.isinf(clean_logits)):
            raise ValueError("clean_logits contains NaN or infinite values")
        # clean_logits:B、num_expert    每个batch对当前层每个专家的权重(32,4)
        if self.noisy_gating and train:     #对应论文公式6
    
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            if torch.any(torch.isnan(noise_stddev)) or torch.any(torch.isinf(noise_stddev)):
                raise ValueError("noise_stddev contains NaN or infinite values")
            
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            if torch.any(torch.isnan(noisy_logits)) or torch.any(torch.isinf(noisy_logits)):
                raise ValueError("noisy_logits contains NaN or infinite values")
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
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            raise ValueError("Input x contains NaN or infinite values")
        # x.shape:B、T、N、d_model
        new_x = self.seasonality_and_trend_decompose(x)

        #multi-scale router
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        # print('gates:',gates)
        # print('load:',load)
        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        # transformer
        # 将分配给第i个专家的数据输入到experts[i]中，取模型返回的预测数据[0]
        expert_outputs = [self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + x
        return output, balance_loss





