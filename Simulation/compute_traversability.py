import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

def compute_traversability_map(step_map, flatness_map, slope_map, var_map, semantic_map,
                               w_step=0.3, w_flat=0.5, w_slope=0.2,
                               crit_step=0.3, crit_flat=0.50, crit_slope=0.50,
                               var_thresh=1.1):
    # 初始化
    trav_map = np.zeros_like(step_map, dtype=np.float32)
    h, w = step_map.shape

    for i in range(h):
        for j in range(w):
            # 水域区域强制赋最大通行代价
            if var_map[i, j] > 1:
                continue
            if semantic_map[i, j] == 1:
                trav_map[i, j] = 0.0
                continue
            if semantic_map[i, j] == 0:
                trav_map[i, j] = 1.0
                continue
            # 基础通行性代价（加权标准化）
            trav = (w_step * (step_map[i, j] / crit_step)) + \
                   (w_flat * (flatness_map[i, j] / crit_flat)) + \
                   (w_slope * (slope_map[i, j] / crit_slope))

            # 方差超限（低置信区域）
            if var_map[i, j] > var_thresh:
                trav = 1.0
            # 归一化到 [0, 1]
            if trav > 1:
                trav=1
            trav_map[i, j] = trav
    return trav_map

def visualize_traversability_map(trav_map, xx, yy, save_path="traversability_heatmap.png"):
    """
    通行性热力图（保留红黄渐变，NaN 区域设为灰色）

    参数：
        trav_map: 通行性代价图（0=好，1=差，np.nan=无扫描）
        xx, yy: meshgrid，用于坐标轴定位
        save_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 获取默认的 hot_r colormap
    base_cmap = cm.get_cmap('hot_r')
    # 创建新的 colormap，设置 NaN 显示为灰色
    new_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    new_cmap.set_bad(color='lightgray')  # NaN 区域设为灰色

    # 显示 trav_map，NaN 会显示为 lightgray
    c = ax.imshow(
        np.ma.masked_invalid(trav_map.T),  # 屏蔽 NaN
        cmap=new_cmap,
        origin='lower',
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        interpolation='none'
    )

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Traversability Cost (0=Good, 1=Bad)")

    ax.set_title("Traversability Heatmap")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()