import torch
import os
import numpy as np
# print(torch.cuda.device_count())
path = r'/data2/zqx/???/datasets/SST/data_south_sea_mv.npy'
data = np.load(path)
data = data[0]
print('data:',data.shape)
print("Min:", np.min(data).item())
print("Max:", np.max(data).item())
print("median:", np.median(data).item())
print("Mean:", np.mean(data).item())
print("Std:", np.std(data).item())