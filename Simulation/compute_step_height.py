# 文件名：compute_step_height.py

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

def compute_step_height(elevation_grid, window_size=3):
    """
    基于高程图计算 Step Height 地图。

    参数：
        elevation_grid: 2D numpy.ndarray，输入的高程图（包含 np.nan 表示无效栅格）
        window_size: int，滑动窗口的尺寸，必须为奇数，默认 3

    返回：
        step_height_map: 2D numpy.ndarray，Step Height 图（已归一化）
    """
    if window_size % 2 == 0:
        raise ValueError("window_size 必须是奇数")

    # 定义最大最小值滤波器，忽略 nan
    max_elev = maximum_filter(np.nan_to_num(elevation_grid, nan=-np.inf), size=window_size, mode='nearest')
    min_elev = minimum_filter(np.nan_to_num(elevation_grid, nan=np.inf), size=window_size, mode='nearest')

    step_height = max_elev - min_elev

    # 将原始无效区域（nan）赋值为 nan
    step_height[np.isnan(elevation_grid)] = np.nan

    # 归一化（避免被异常值拉伸）
    valid_mask = ~np.isnan(step_height)
    # if np.any(valid_mask):
    #     max_val = np.nanmax(step_height)
    #     if max_val > 0:
    #         step_height[valid_mask] = step_height[valid_mask] / max_val

    return step_height
