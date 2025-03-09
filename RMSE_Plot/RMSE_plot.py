import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义 LSTM-GPR 的样本长度 l
l = 2116  # 假设 l 为 2116，您可以根据实际情况修改

# 定义 .mat 文件的路径列表
mat_files = [
    '/data2/zqx/zqx0413/store/a_LSTM-GPR/MDPI/svmd_SVR.mat',
    '/data2/zqx/zqx0413/store/a_LSTM-GPR/MDPI/svmd_GPR.mat',
    '/data2/zqx/zqx0413/store/a_LSTM-GPR/MDPI/svmd_LSTM.mat',
    '/data2/zqx/zqx0413/store/a_LSTM-GPR/MDPI/svmd_LSTM_GPR.mat'
    # 添加更多 .mat 文件路径
]

def plot_rmse_heatmap(day, mat_files, l):
    # 存储每个模型的预测结果和真实值
    predictions = []
    ground_truth = []

    # 读取 svmd_LSTM_GPR.mat 文件获取第day天的真实值
    mat_file = scipy.io.loadmat(mat_files[3])
    sst_real = mat_file['sst_real'][-l:, :, :, day - 1]
    sst_real[sst_real == -999] = np.nan

    # 创建一个与 sst_real 形状相同的布尔数组，其中 True 表示应该掩盖的区域（即 sst_real 中的 -999 或 nan）
    mask = np.isnan(sst_real[0])

    # 存储每个天数的RMSE结果
    day_predictions = []

    # 遍历每个 .mat 文件
    for mat_file_path in mat_files:
        # 读取 .mat 文件
        mat_file = scipy.io.loadmat(mat_file_path)

        # 获取 sst_pred 数据
        sst_pred = mat_file['sst_pred'][-l:, :, :, day - 1]

        # 排除陆地数据（值为 -999 的元素）
        sst_pred[sst_pred == -999] = np.nan

        # 计算每个点的 RMSE
        rmse = np.sqrt(np.mean((sst_pred - sst_real) ** 2, axis=0))

        # 存储 RMSE 结果
        day_predictions.append(rmse)

    # 计算最后一天预测结果的最小值和最大值，用于统一colorbar的范围
    vmin = np.min([np.nanmin(pred) for pred in day_predictions])
    vmax = np.max([np.nanmax(pred) for pred in day_predictions])

    # 绘制指定天数的RMSE热图
    fig, axes = plt.subplots(nrows=1, ncols=len(day_predictions), figsize=(16, 4))
    
    for j, prediction in enumerate(day_predictions):
        # 在heatmap中使用自定义颜色映射
        cmap_c = 'viridis' # plasma
        sns.heatmap(prediction, cmap=cmap_c, ax=axes[j], vmin=vmin, vmax=vmax, square=True, cbar=False, mask=mask, cbar_kws={'label': 'RMSE'})
        axes[j].invert_yaxis()  # 反转y轴
    
        # 提取文件名中的方法名作为标题
        method_name = mat_file_path.split('/')[-1].split('.')[0].split('_')[-1]
        axes[j].axis('off')  # 移除横纵坐标
    
    # 添加子图边框
    for spine in axes[j].spines.values():
        spine.set_visible(True)
        spine.set_color('black')  # 设置边框颜色
        spine.set_linewidth(2)    # 设置边框宽度
    
    # 添加统一的colorbar
    cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label('RMSE')
    
    # 保存指定天数的热图
    plt.savefig(f'RMSE_heatmap_group_day_{day}_{cmap_c}.png')
    plt.close()

# 调用函数绘制指定天数的热图
plot_rmse_heatmap(day=1, mat_files=mat_files, l=l)  # 这里以绘制第3天为例，您可以根据需要修改day的值
