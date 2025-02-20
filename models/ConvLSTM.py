import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    #这里面全都是数，衡量后面输入数据的维度/通道尺寸
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        #卷积核为一个数组
        self.kernel_size = kernel_size
        #填充为高和宽分别填充的尺寸
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # 保证在传递过程中 （h,w）不变
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # i门，f门，o门，g门放在一起计算，然后在split开 ----4* 是因为后面输出时要切4片
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state    # 每个timestamp包含两个状态张量：h和c
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis # 把输入张量与h状态张量沿通道维度串联
        
        combined_conv = self.conv(combined) # i门，f门，o门，g门放在一起计算，然后在split开
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        #torch.sigmoid(),激活函数--
        #nn.functional中的函数仅仅定义了一些具体的基本操作，
        #不能构成PyTorch中的一个layer
        #torch.nn.Sigmoid()(input)等价于torch.sigmoid(input)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        #这里的乘是矩阵对应元素相乘，哈达玛乘积
        c_next = f * c_cur + i * g  # c状态张量更新
        h_next = o * torch.tanh(c_next) # h状态张量更新

        return h_next, c_next   # 输出当前timestamp的两个状态张量

    def init_hidden(self, batch_size, shape):
        """
        初始状态张量初始化.第一个timestamp的状态张量0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = shape
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (init_h,init_c)



class Model(nn.Module):
    
    """
    Parameters:参数介绍
        input_dim: Number of channels in input# 输入张量的通道数
        hidden_dim: Number of hidden channels # h,c两个状态张量的通道数，可以是一个列表
        kernel_size: Size of kernel in convolutions # 卷积核的尺寸，默认所有层的卷积核尺寸都是一样的,也可以设定不通lstm层的卷积核尺寸不同
        num_layers: Number of LSTM layers stacked on each other # 卷积层的层数，需要与len(hidden_dim)相等
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers # 是否返回所有lstm层的h状态
        Note: Will do same padding. # 相同的卷积核尺寸，相同的padding尺寸
    Input:输入介绍
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]# 需要是5维的
    Output:输出介绍
        返回的是两个列表：layer_output_list，last_state_list
        列表0：layer_output_list--单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[B,T,hidden_dim,H,W]
        列表1：last_state_list--双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个timestamp的输出状态[h,c],h.size=c.size = [B,hidden_dim,H,W]
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:使用示例
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = configs.input_dim
        self.hidden_dim = [16, 32,]
        self.kernel_size = [(3, 3),(3,3)]
        self.num_layers = 2
        self.batch_first = True
        self.bias = True
        self.return_all_layers = False
        self.output_dim = configs.pred_len

        self._check_kernel_size_consistency(self.kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        self.kernel_size = self._extend_for_multilayer(self.kernel_size, self.num_layers) # 转为列表
        self.hidden_dim = self._extend_for_multilayer(self.hidden_dim, self.num_layers) # 转为列表
        if not len(self.kernel_size) == len(self.hidden_dim) == self.num_layers: # 判断一致性
            raise ValueError('Inconsistent list length.')
        print("kernel_size", self.kernel_size)
        print("hidden_dim", self.hidden_dim)

        
        # Stack multiple ConvLSTMCells
        #为了储存每一层的参数尺寸
        cell_list = []
        for i in range(self.num_layers):
            # 当前LSTM层的输入维度
            # if i==0:
            #     cur_input_dim = self.input_dim
            # else:
            #     cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # 与上等价
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        #注意这里用了ModuLelist函数，模块化列表
        self.layers = nn.ModuleList(cell_list) # 把定义的多个LSTM层串联成网络模型

        
        # self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer for output
        # 输出层
        self.out_conv = nn.Conv2d(in_channels=self.hidden_dim[-1], out_channels=self.output_dim, kernel_size=1)

    #这里forward有两个输入参数，input_tensor 是一个五维数据
    #（t时间步,b输入batch_ize,c输出数据通道数--维度,h,w图像高乘宽）
    #hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
    def forward(self, x, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        # 假设 x 是你的原始数据，形状为 [batch_size, seq_length, height, width]
        # 添加通道维度
        x = x.unsqueeze(2)  # 新的形状为 [batch_size, seq_length, 1, height, width]
        # Initialize the states
        batch_size, seq_len, _, height, width = x.size()
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            #取出图片的数据，供下面初始化使用
            # Since the init is done in forward. Can send image size here
            b, _, _, height, width = x.size()  # 自动获取 b,h,w信息
            hidden_state = self._init_hidden(batch_size=b,image_size=(height, width))
        
        #储存输出数据的列表
        layer_output_list = []
        last_state_list = []
 
        seq_len = x.size(1) # 根据输入张量获取lstm的长度
        cur_layer_input = x  # 初始化输入数据
 
        for layer_idx in range(self.num_layers): # 逐层计算
 
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): # 逐个stamp计算
                #每一个时间步都更新 h,c
                #注意这里self.cell_list是一个模块(容器)
                h, c = self.layers[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],cur_state=[h, c])
                output_inner.append(h) # 第 layer_idx 层的第t个stamp的输出状态  储存输出，注意这里 h 就是此时间步的输出
 
            layer_output = torch.stack(output_inner, dim=1) # 第 layer_idx 层的第所有stamp的输出状态串联
            cur_layer_input = layer_output # 准备第layer_idx+1层的输入张量
 
            layer_output_list.append(layer_output) # 当前层的所有timestamp的h状态的串联
            last_state_list.append([h, c]) # 当前层的最后一个stamp的输出状态的[h,c]

        # 选择要输出所有数据，还是输出最后一层的数据
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        last_layer_output = layer_output_list[-1]
        # last_layer_output = layer_output_list[-1]
        
        # 输出层处理
        out = self.out_conv(last_layer_output[:,-1,...])  # 应用卷积输出层
        # out = self.fc(h[-1][:, -1, :, :].view(batch_size, -1))  # Flatten and pass through FC
        return out

    def _init_hidden(self, batch_size, image_size):
        """
        所有lstm层的第一个timestamp的输入状态0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.layers[i].init_hidden(batch_size, image_size))
        return init_states
 
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
 
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        扩展到多层lstm情况
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
