from models import CrossGNN, EA_CrossGNN, EM_CrossGNN, snMoE_CrossGNN, EA_snMoE_CrossGNN, PathFormer, FC_LSTM, ConvLSTM, Transformer, TimesNet, FEDformer, Autoformer, iTransformer, FiLM, SegRNN, PatchTST #, DLinear, Linear, NLinear, 
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import math
import os
import time
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

class GaussianNLLLoss(nn.Module):
    def __init__(self):
        super(GaussianNLLLoss, self).__init__()

    def forward(self, mean, std, target):

        if torch.any(std <= 0):
            raise ValueError("标准差必须大于零")

        mean = torch.flatten(mean)
        std = torch.flatten(std)
        target = torch.flatten(target)

        d = torch.distributions.normal.Normal(mean, std)
        loss = d.log_prob(target)
        # nll = 0.5 * torch.log(2 * math.pi * var) + (target - mean) ** 2 / (2 * var)
        return -torch.mean(loss)

class WeightedLoss(nn.Module):
    def __init__(self, weight_nll=0.5, weight_mse=0.5):
        super(WeightedLoss, self).__init__()
        self.weight_nll = weight_nll
        self.weight_mse = weight_mse
        self.gaussian_nll_loss = GaussianNLLLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, mean, std, target):
        nll_loss = self.gaussian_nll_loss(mean, std, target)
        mse_loss = self.mse_loss(mean, target)
        total_loss = self.weight_nll * nll_loss + self.weight_mse * mse_loss
        return total_loss          

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.loss_selection=args.loss
        self.type = args.type
        
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'FEDformer': FEDformer,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
            # 'Informer': Informer,
            # 'DLinear': DLinear,
            # 'NLinear': NLinear,
            # 'Linear': Linear,
            'SegRNN': SegRNN,
            'FiLM': FiLM,
            'TimesNet': TimesNet,
            'FC_LSTM':FC_LSTM,
            'ConvLSTM':ConvLSTM,
            'PathFormer':PathFormer,
            'CrossGNN': CrossGNN,
            'EA_CrossGNN': EA_CrossGNN,
            'EM_CrossGNN': EM_CrossGNN,
            'snMoE_CrossGNN': snMoE_CrossGNN,
            'EA_snMoE_CrossGNN': EA_snMoE_CrossGNN,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.loss_selection == 'mae':
            criterion = nn.L1Loss()
        elif self.loss_selection == 'SmoothL1Loss' :
            criterion = nn.SmoothL1Loss()
        elif self.loss_selection == 'nll':
            criterion = GaussianNLLLoss()
        elif self.loss_selection == 'weighted':
            criterion = WeightedLoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp: #基本没提到这个东西
                    with torch.cuda.amp.autocast():
                        if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                            outputs, balance_loss, std = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'GNN' in self.args.model or 'FC_LSTM' in self.args.model or 'ConvLSTM' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, None, dec_inp, None)
                            else:
                                outputs = self.model(batch_x, None, dec_inp, None)
                else:
                    if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                        outputs, balance_loss, std = self.model(batch_x)
                    elif 'Linear' in self.args.model or 'GNN' in self.args.model or 'FC_LSTM' in self.args.model or 'ConvLSTM' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, None, dec_inp, None)
                        else:
                            outputs = self.model(batch_x, None, dec_inp, None)
                if 'ConvLSTM'  in self.args.model:
                    outputs = outputs[:, -self.args.pred_len:, :, :] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                    batch_y = batch_y[:, -self.args.pred_len:, :, :].to(self.device)  #因为batch_y :  torch.Size([32, 240, 139])，要选择后pred_len的长度算损失
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0  # feature是预测模式，默认是多变量预测多变量
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  #因为batch_y :  torch.Size([32, 240, 139])，要选择后pred_len的长度算损失
                if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                    std = std[:, -self.args.pred_len:, f_dim:]
                    std = std.detach().cpu()

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                if(self.loss_selection in ['nll', 'weighted']):
                    loss = criterion(pred, std, true)
                else:
                    loss = criterion(pred, true)
                # print("loss：", loss)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        print("total_loss:", total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        total_num = sum(p.numel() for p in self.model.parameters())
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        #先遵守PathFormer源论文的设置
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=self.args.pct_start,
                                        epochs=self.args.train_epochs,
                                        max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # print('batch_x.shape:',batch_x.shape)   #[B, T, C]
                # print('batch_y.shape:',batch_y.shape)
                # print('batch_x.shape:',batch_x.shape)
                # print("Min:", torch.min(batch_x).item())
                # print("Max:", torch.max(batch_x).item())
                # print("median:", torch.median(batch_x).item())
                # print("Mean:", torch.mean(batch_x).item())
                # print("Std:", torch.std(batch_x).item())  


                # print('batch_y.shape:',batch_y.shape)
                # print("Min:", torch.min(batch_y).item())
                # print("Max:", torch.max(batch_y).item())
                # print("median:", torch.median(batch_y).item())
                # print("Mean:", torch.mean(batch_y).item())
                # print("Std:", torch.std(batch_y).item())  
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # print('decoder input: ', dec_inp.shape) # decoder input:  torch.Size([32, 192, 139])

                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # print('decoder input: ', dec_inp.shape) # decoder input:  torch.Size([32, 240, 139])

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                            outputs, balance_loss, std = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'GNN' in self.args.model or 'FC_LSTM' in self.args.model or 'ConvLSTM' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, None, dec_inp, None)
                            else:
                                outputs = self.model(batch_x, None, dec_inp, None)

                        if 'ConvLSTM'  in self.args.model:
                            outputs = outputs[:, -self.args.pred_len:, :, :] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                            batch_y = batch_y[:, -self.args.pred_len:, :, :].to(self.device)  #因为batch_y :  torch.Size([32, 240, 139])，要选择后pred_len的长度算损失
                        else:
                            f_dim = -1 if self.args.features == 'MS' else 0  # feature是预测模式，默认是多变量预测多变量
                            outputs = outputs[:, -self.args.pred_len:, f_dim:] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                        # print('self.args.model:',self.args.model)
                        # print('进入PathFormer or MoE')
                        outputs, balance_loss ,std= self.model(batch_x)
                        if torch.any(std <= 0):
                            raise ValueError("标准差必须大于零")
                        # print('一次forward')
                    elif 'Linear'in self.args.model or 'GNN' in self.args.model or 'FC_LSTM' in self.args.model or'ConvLSTM' in self.args.model:
                            outputs = self.model(batch_x)
                            # print(i,'outputs.shape:', outputs.shape)   # outputs.shape: torch.Size([32, 192, 139])
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, None, dec_inp, None)
                            
                        else:
                            outputs = self.model(batch_x, None, dec_inp, None)
                    # print(outputs.shape,batch_y.shape)
                    if 'ConvLSTM'  in self.args.model:
                        outputs = outputs[:, -self.args.pred_len:, :, :] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                        batch_y = batch_y[:, -self.args.pred_len:, :, :].to(self.device)  #因为batch_y :  torch.Size([32, 240, 139])，要选择后pred_len的长度算损失
                    else:
                        f_dim = -1 if self.args.features == 'MS' else 0  # feature是预测模式，默认是多变量预测多变量
                        outputs = outputs[:, -self.args.pred_len:, f_dim:] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 



                    if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                        std = std[:, -self.args.pred_len:, f_dim:]
                        # print("balance_loss:", balance_loss)
                    if(self.loss_selection in ['nll', 'weighted']):
                        loss = criterion(outputs, std, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)
                        # print('outputs:',outputs.shape)
                        # print("Min:", torch.min(outputs).item())
                        # print("Max:", torch.max(outputs).item())
                        # print("median:", torch.median(outputs).item())
                        # print("Mean:", torch.mean(outputs).item())
                        # print("Std:", torch.std(outputs).item())        
                        # for i in range(std.shape[1]):
                            # print('std:',std[:, i, :].shape)
                            # print("Min:", torch.min(std).item())
                            # print("Max:", torch.max(std).item())
                            # print("median:", torch.median(std).item())
                            # print("Mean:", torch.mean(std).item())
                            # print("Std:", torch.std(std).item())  
                    
                    # print("loss：", loss)
                    if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                        loss = loss + balance_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                # print("vali(vali_data)")
                vali_loss = self.vali(vali_data, vali_loader, criterion)    #######
                # print("vali(test_data)")
                test_loss = self.vali(test_data, test_loader, criterion)    ######

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format( epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format( epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # 如果test参数为真，则加载训练好的模型
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 初始化用于存储预测结果和真实值的列表
        preds = []
        trues = []
        inputx = []
        stds = []
        # 创建用于存放测试结果的文件夹
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # print('batch_x.shape:',batch_x.shape)       # batch_x.shape: torch.Size([32, 90, 112])   if type=="mv" ： batch_x.shape: torch.Size([9, 90, 112, 3])
                # print('batch_y.shape:',batch_y.shape)       # batch_y.shape: torch.Size([32, 30, 112])                    batch_y.shape: torch.Size([9, 30, 112])
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                            outputs, balance_loss, std = self.model(batch_x)
                        elif 'Linear' in self.args.model or 'GNN' in self.args.model or 'FC_LSTM' in self.args.model or 'ConvLSTM' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, None, dec_inp, None)
                            else:
                                outputs = self.model(batch_x, None, dec_inp, None)
                else:
                    if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                        outputs, balance_loss ,std= self.model(batch_x)
                        # print('test():outputs.shape：', outputs.shape)
                        # print('test():batch_y.shape：', batch_y.shape)
                        # print('test():std.shape：', std.shape)
                        if torch.any(std <= 0):
                            raise ValueError("标准差必须大于零")
                    elif 'Linear' in self.args.model or 'GNN' in self.args.model or 'FC_LSTM' in self.args.model or 'ConvLSTM' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, None, dec_inp, None)
                        else:
                            outputs = self.model(batch_x, None, dec_inp, None)
                if 'ConvLSTM'  in self.args.model:
                    outputs = outputs[:, -self.args.pred_len:, :, :] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                    batch_y = batch_y[:, -self.args.pred_len:, :, :].to(self.device)  #因为batch_y :  torch.Size([32, 240, 139])，要选择后pred_len的长度算损失
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0  # feature是预测模式，默认是多变量预测多变量
                    outputs = outputs[:, -self.args.pred_len:, f_dim:] # 感觉这个第二维度的选择相当于:，本来模型的linear就设置好了
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 

                if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
                    std = std[:, -self.args.pred_len:, f_dim:]
                    std = std.detach().cpu().numpy()
                    stds.append(std)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                input = batch_x.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                #print(pred.shape) (32, 96, 7)
                preds.append(pred)
                trues.append(true)
                inputx.append(input)
                

                # print('test____i____', len(preds))

                if i % 20 == 0:
                    if(self.args.type == 'mv'):
                        input = input[:,:,:,0]

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        # 如果设置了测试flop，则进行测试并退出
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds,axis=0)
        trues = np.concatenate(trues,axis=0)
        inputx = np.concatenate(inputx,axis=0)
        print('test shape:', preds.shape, trues.shape, inputx.shape)
        

        if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
            stds = np.concatenate(stds, axis=0)
            print('test prob shape：', stds.shape)

        #print(preds.shape[-2])

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './prob_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # print('test_data shape:', test_data.shape)

        # preds = preds.reshape(preds.shape[0], -1)  # Flatten data to fit scaler
        preds = test_data.scaler.inverse_transform(preds, self.type)
        # preds = preds.reshape(preds_shape) 

        # trues = trues.reshape(trues.shape[0], -1)  # Flatten data to fit scaler
        trues = test_data.scaler.inverse_transform(trues, self.type)
        # trues = trues.reshape(trues_shape) 
        # stds = test_data.scaler.inverse_transform(stds, self.type)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('corr：{}, mse:{}, rmse:{}, mae:{}'.format(corr, mse, rmse, mae))
        # 打印每个指标的形状信息
        # print(f"mae shape: {np.array(mae).shape if isinstance(mae, (list, np.ndarray)) else 'scalar'}")
        # print(f"mse shape: {np.array(mse).shape if isinstance(mse, (list, np.ndarray)) else 'scalar'}")
        # print(f"rmse shape: {np.array(rmse).shape if isinstance(rmse, (list, np.ndarray)) else 'scalar'}")
        # print(f"mape shape: {np.array(mape).shape if isinstance(mape, (list, np.ndarray)) else 'scalar'}")
        # print(f"mspe shape: {np.array(mspe).shape if isinstance(mspe, (list, np.ndarray)) else 'scalar'}")
        # print(f"rse shape: {np.array(rse).shape if isinstance(rse, (list, np.ndarray)) else 'scalar'}")
        # print(f"corr shape: {np.array(corr).shape if isinstance(corr, (list, np.ndarray)) else 'scalar'}")
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, rse:{}, corr:{}'.format(mse, mae, rmse, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'inputx.npy', inputx)
        if 'PathFormer' in self.args.model or 'MoE' in self.args.model:
            np.save(folder_path + 'std.npy', stds)

        return

    def predict(self, setting, load=False):     #未改
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                print('decoder input: ', dec_inp.shape)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                print('decoder input: ', dec_inp.shape)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear'  or 'GNN' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' or 'GNN'  in self.args.model:
                        outputs = self.model(batch_x)
                        print('outputs.shape:', outputs.shape)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds, type)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
