import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_mask(region_name):
    """根据区域名称加载对应的二维mask"""
    if "bohai" in region_name:
        mask_path = "/data2/zqx/A_DSIN/datasets/SST/data_bohai_37_117_mask.npy"
    else:
        mask_path = "/data2/zqx/A_DSIN/datasets/SST/data_south_sea_60_28_mask.npy"
    return np.load(mask_path)

def calculate_rmse(pred, true):
    # 输入形状: (num_samples, pred_length, num_points)
    # 合并样本和预测步长维度
    pred_2d = pred.reshape(-1, pred.shape[2])  # [num_samples*pred_length, num_points]
    true_2d = true.reshape(-1, true.shape[2])
    return np.sqrt(np.mean((pred_2d - true_2d)**2, axis=0))  # [num_points]

def plot_heatmaps(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 新的数据结构：{region: {model: {pred_length: rmse_map}}}
    results = {}
    
    folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for folder in tqdm(folders):
        # 解析新的目录结构（示例：south_sea_90_180_Autoformer）
        parts = folder.split('_')
        region = '_'.join(parts[:2])  # 获取区域（south_sea 或 bohai）
        pred_length = int(parts[3])   # 获取预测步长（30/90/180/365）
        model_name = parts[4]         # 获取模型名称
        
        # 加载数据
        pred_path = os.path.join(root_dir, folder, 'pred.npy')
        true_path = os.path.join(root_dir, folder, 'true.npy')
        if not (os.path.exists(pred_path) and os.path.exists(true_path)):
            continue
            
        try:
            pred = np.load(pred_path)
            true = np.load(true_path)
        except:
            continue

        # 计算RMSE (形状: [num_points])
        rmse_1d = calculate_rmse(pred, true)
        
        # 获取二维mask
        mask = load_mask(region)
        h, w = mask.shape
        
        # 还原到二维空间
        rmse_2d = np.full((h, w), np.nan)
        valid_points = np.where(mask == 0)
        
        # 确保维度匹配
        if rmse_1d.shape[0] != valid_points[0].size:
            print(f"维度不匹配: {folder} 的valid_points数量{valid_points[0].size}与RMSE结果数量{rmse_1d.shape[0]}不一致")
            continue
            
        rmse_2d[valid_points] = rmse_1d  # 直接赋值一维结果

        # 存储结果
        if region not in results:
            results[region] = {}
        if model_name not in results[region]:
            results[region][model_name] = {}
        results[region][model_name][pred_length] = rmse_2d
    
    # 绘制热力图
    for region in results:
        models = results[region]
        pred_lengths = [30, 90, 180, 365]  # 固定的预测步长顺序
        
        # 创建画布：每行一个模型，四列对应四个预测步长
        fig, axs = plt.subplots(len(models), 4, 
                              figsize=(20, 5*len(models)),
                              sharex=True, sharey=True)
        
        # 获取全局颜色范围
        all_rmse = np.concatenate(
            [rmse[None] for model in models.values() for rmse in model.values()]
        )
        vmin, vmax = np.nanpercentile(all_rmse, [2, 98])
        
        for row_idx, (model_name, pred_data) in enumerate(models.items()):
            for col_idx, length in enumerate(pred_lengths):
                ax = axs[row_idx, col_idx] if len(models) > 1 else axs[col_idx]
                
                # 提取对应步长的数据
                rmse_map = pred_data.get(length, np.full((h, w), np.nan))
                
                # 绘制热力图（添加origin和interpolation参数）
                im = ax.imshow(rmse_map, 
                             cmap='jet',  # 更柔和的颜色映射 jet viridis plasma
                             origin='lower',  # 反转y轴方向
                             interpolation='bicubic',  # 添加渐变过渡 gaussian bicubic
                             vmin=vmin, 
                             vmax=vmax)
                ax.set_title(f"{model_name} - {length} steps")
                ax.axis('off')
        
        # 添加公共colorbar
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, 
                    label='RMSE (°C)')
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f"{region}_heatmaps-5.png"),
                   bbox_inches='tight', dpi=300)
        plt.close()

# 运行参数
root_dir = "/data2/zqx/A_DSIN/RMSE_Plot"
output_dir = "/data2/zqx/A_DSIN/Plot/heatmaps"
plot_heatmaps(root_dir, output_dir)
