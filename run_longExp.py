import torch
import numpy as np
import random
from exp.exp_main import Exp_Main
import argparse
import os
import time


fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')     # --is_training 1
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation an.d testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id') # --model_id zqx01
parser.add_argument('--model', type=str, required=True, default='Autoformer', help='model name, options: [Autoformer, Informer, Transformer]')   # --model CrossGNN

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--type', type=str, default='vaild', help='data file')
# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# FC_LSTM&ConvLSTM
parser.add_argument('--input_dim', type=int, default=1, help='input feature:sst')
parser.add_argument('--hidden_dim', type=int, default=64, help='lstm cell hidden_dim')

# ConvLSTM
# parser.add_argument('--kernel_size', type=int, default=1, help='convkstm cell conv kernel_size')
# parser.add_argument('--num_layers', type=int, default=2, help='convlstm num_layers')

parser.add_argument('--global_hidden', type=int, default=24, help='global_hidden')
# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')  #重名
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')      #重名
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')      #重复与drop
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')   #重复
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')  #重复

#pathformer model
parser.add_argument('--num_nodes', type=int, default=21)    ########    同 enc_in=112
parser.add_argument('--layer_nums', type=int, default=2)    ########    3
parser.add_argument('--k', type=int, default=4, help='choose the Top K patch size at the every layer ')     # 2
parser.add_argument('--num_experts_list', type=list, default=[4,4])
parser.add_argument('--tk_list', nargs='+', type=int, default=[3, 5, 10, 15, 3, 5, 10, 15])
parser.add_argument('--sn_list', nargs='+', type=int, default=[1, 2, 4, 8, 1, 2, 4, 8])
parser.add_argument('--revin', type=int, default=0, help='whether to apply RevIN')
parser.add_argument('--residual_connection', type=int, default=0)
# parser.add_argument('--metric', type=str, default='mae')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='nll', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# path’s optimization
parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
# GNN
parser.add_argument('--tvechidden', type=int, default=1, help='scale vec dim')
parser.add_argument('--nvechidden', type=int, default=1, help='variable vec dim')
parser.add_argument('--use_tgcn', type=int, default=1, help='use cross-scale gnn')
parser.add_argument('--use_ngcn', type=int, default=1, help='use cross-variable gnn')
parser.add_argument('--anti_ood', type=int, default=1, help='simple strategy to solve data shift')
parser.add_argument('--scale_number', type=int, default=4, help='scale number')
parser.add_argument('--hidden', type=int, default=8, help='channel dim')
parser.add_argument('--tk', type=int, default=5, help='constant w.r.t corss-scale neighbors')  #跨尺度邻居数量

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.tk_list = np.array(args.tk_list).reshape(args.layer_nums, -1).tolist()
args.sn_list = np.array(args.sn_list).reshape(args.layer_nums, -1).tolist()


print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_lr{}_ngcn{}_tgcn{}_ood{}_td{}_nd{}_loss{}_layer{}_scale{}_hidden{}_tk{}_gh{}'.format(
            args.model_id, args.model, args.data,
            args.features,  args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii,
            args.learning_rate,
            args.use_ngcn,
            args.use_tgcn,
            args.anti_ood,
            args.tvechidden,
            args.nvechidden,
            args.loss,
            args.e_layers,
            args.scale_number,
            args.hidden,
            args.tk,
            args.global_hidden
            )

        exp = Exp(args)  # set experiments
        time_now = time.time()
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('train time: ', time.time() - time_now)
        
        if not args.train_only:
            time_now = time.time()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            print('Inference time: ', time.time() - time_now)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_lr{}_ngcn{}_tgcn{}_ood{}_td{}_nd{}_loss{}_layer{}_scale{}_hidden{}_tk{}_gh{}'.format(
        args.model_id, args.model, args.data,
        args.features,  args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii,
        args.learning_rate,
        args.use_ngcn,
        args.use_tgcn,
        args.anti_ood,
        args.tvechidden,
        args.nvechidden,
        args.loss,
        args.e_layers,
        args.scale_number,
        args.hidden,
        args.tk,
        args.global_hidden,
        )

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
