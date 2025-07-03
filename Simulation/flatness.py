import numpy as np
from scipy.ndimage import uniform_filter

def compute_flatness(elevation_grid, resolution=0.4, physical_window=1.2):
    """
    计算每个栅格的地形平坦度指标（Flatness），基于高程的局部变化。
    
    参数：
        elevation_grid (np.ndarray): 高程二维数组
        resolution (float): 栅格尺寸（米），默认0.4m
        physical_window (float): 物理窗口大小（米），比如1.2米范围
    返回：
        flatness_map (np.ndarray): 平坦度图，归一化到0-1之间
    """
    # 计算实际需要的窗口格子数（必须是奇数）
    window_size = int(np.round(physical_window / resolution))
    if window_size % 2 == 0:
        window_size += 1  # 保证是奇数
    
    # 计算局部均值
    local_mean = uniform_filter(elevation_grid, size=window_size, mode='nearest')
    
    # 计算局部方差
    squared_diff = (elevation_grid - local_mean) ** 2
    local_var = uniform_filter(squared_diff, size=window_size, mode='nearest')
    
    # 归一化
    flatness_map = local_var / np.nanmax(local_var)
    
    return flatness_map
