# 文件名：generate_semantic_map.py

import numpy as np
from scipy import ndimage

def generate_semantic_map(mean, slope, var, step_map=None, flatness_map=None,
                          resolution=0.4, slope_thresh=0.10, elev_thresh=0.3,
                          crit_slope=0.50, crit_step=0.35, crit_flat=0.50):
    h, w = mean.shape
    semantic_map = np.full((h, w), 255, dtype=np.uint8)  # 初始化为无效区域

    for i in range(h):
        for j in range(w):
            if var[i, j] > 1:
                continue
            # 水域判断：平坦+低高程+低不确定性
            is_flat = slope[i, j] < slope_thresh
            is_low = mean[i, j] < elev_thresh
            is_var_good = var[i, j] < 1.1
            if is_flat and is_low and is_var_good:
                semantic_map[i, j] = 1  # 水域（蓝）
                continue

            # 多特征障碍判断：任意一项超过阈值即为障碍物
            is_obstacle = False
            if np.arctan(slope[i, j]) > crit_slope:
                is_obstacle = True
            if step_map is not None and not np.isnan(step_map[i, j]) and step_map[i, j] > crit_step:
                is_obstacle = True
            if flatness_map is not None and not np.isnan(flatness_map[i, j]) and flatness_map[i, j] > crit_flat:
                is_obstacle = True

            if is_obstacle:
                semantic_map[i, j] = 0  # 障碍物（红）
            elif is_var_good:
                semantic_map[i, j] = 2  # 通行区域（绿）
            else:
                semantic_map[i, j] = 255  # 无效区域（灰）

    # 去除过小水域斑块
    water_mask = (semantic_map == 1)
    labeled, num_features = ndimage.label(water_mask)
    for label in range(1, num_features + 1):
        region = (labeled == label)
        if np.sum(region) < 8:
            semantic_map[region] = 2  # 设为通行区域

    return semantic_map

# ✅ 可视化函数：带颜色映射（红=障碍物，蓝=水域，绿=通行，灰=无效）
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_semantic_map(semantic_map, xx, yy, save_path="semantic_map.png"):
    cmap = ListedColormap(["red", "blue", "green", "lightgray"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 255.5], cmap.N)
    labels = ['Obstacle', 'Water', 'Land', 'Invalid']
    ticks = [0, 1, 2, 255]

    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.imshow(
        semantic_map.T,
        cmap=cmap,
        norm=norm,
        origin='lower',
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        interpolation='none'
    )
    cbar = fig.colorbar(c, ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("Semantic Label")

    ax.set_title("Semantic Map")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()
