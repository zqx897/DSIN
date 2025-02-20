import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt


# 方法名称列表
methods = ['FC-LSTM', 'DSIN-EA-DR', 'DSIN-EA', 'DSIN-DR', 'DSIN']
forecast_lengths = [30, 90, 180, 365]

# 文件路径
mat_file_path = r'/data2/zqx/A_DSIN/datasets/SST/data_bohai_37_117.mat'
data = sio.loadmat(mat_file_path)
sst = data['sst'].astype('float32')  # 海表温度

# 获取有效点的坐标
valid_mask = (sst != -999)  # 基于sst的无效值掩码
valid_indices = np.array(np.where(valid_mask[0]))  # 获取有效点的坐标（i, j）
valid_indices = valid_indices.T  # 转置为 (112, 2) 形状，每一行是 (x, y) 坐标
print(valid_indices.shape)

# 用于存储每个方法、每个步长的最后一天的RMSE值
rmse_results = {}

# 可视化：绘制每个方法在不同预测步长下的RMSE
fig, axes = plt.subplots(len(methods), len(forecast_lengths), figsize=(15, 10))

# 调整子图布局
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 遍历每个方法
for i, method in enumerate(methods):
    rmse_results[method] = {}
    
    # 遍历每个预测步长
    for j, length in enumerate(forecast_lengths):
        # 构建文件路径
        pred_path = f'/data2/zqx/A_DSIN/prob_results/bohai/{method}/{length}/pred.npy'
        true_path = f'/data2/zqx/A_DSIN/prob_results/bohai/{method}/{length}/true.npy'
        
        # 加载预测和真实值数据
        pred = np.load(pred_path)  # 预测数据，形状为 (N, T, 112)
        true = np.load(true_path)  # 真实数据，形状为 (N, T, 112)
        
        # 确保数据形状一致
        assert pred.shape == true.shape, f"数据形状不一致：{method}, {length}"
        
        # 获取最后一天的预测和真实值
        pred_last_day = pred[:, -1, :]  # 获取最后一天的预测值，形状为 (N, 112)
        true_last_day = true[:, -1, :]  # 获取最后一天的真实值，形状为 (N, 112)
        print(pred_last_day.shape, true_last_day.shape)

        # 计算每个有效点的RMSE
        rmse_per_point = np.sqrt(np.mean((pred_last_day - true_last_day) ** 2, axis=0))  # 对112个有效点计算RMSE
        print(f"{method} - {length} last day RMSE: {rmse_per_point},shape{rmse_per_point.shape}")

        # 存储结果
        rmse_results[method][length] = rmse_per_point
        
        # 将RMSE还原到16×16网格
        rmse_map = np.full((16, 16), np.nan)  # 初始化为NaN，表示无效点
        for k, (x, y) in enumerate(valid_indices):
            rmse_map[x, y] = rmse_per_point[k]  # 将RMSE值填充到有效点的位置

        # 绘制RMSE热图
        ax = axes[i, j]
        c = ax.imshow(rmse_map, cmap='viridis', interpolation='nearest')
        ax.set_title(f'{method} - {length} days')
        ax.set_xticks([0, 15])
        ax.set_yticks([0, 15])
        if j == 0:
            ax.set_ylabel(method)

# 添加共享的colorbar
fig.colorbar(c, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

plt.tight_layout()
# 保存图像
plt.savefig('/data2/zqx/A_DSIN/prob_results/bohai_rmse_heatmap.png')
print(f"RMSE heatmap for bohai days) saved succeed")

# 打印每个方法在不同步长下的RMSE
for method in rmse_results:
    for length in rmse_results[method]:
        print(f"{method} - {length} days RMSE: {rmse_results[method][length]},shape{rmse_results[method][length].shape}")


