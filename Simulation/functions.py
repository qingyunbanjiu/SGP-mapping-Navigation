import numpy as np
import plotly.graph_objects as go
import torch
import gpytorch
from time import time
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import splprep, splev, griddata
from map_utils import update_global_maps_from_sgp, visualize_local_map_with_path
from terrain_angle import car_angle
from scipy.ndimage import maximum_filter
from flatness import compute_flatness
from compute_step_height import compute_step_height
from generate_semantic_map import generate_semantic_map
from generate_semantic_map import generate_semantic_map, visualize_semantic_map
from compute_traversability import compute_traversability_map, visualize_traversability_map
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation
from dstar_lite import DStarLite
from point_utils import Point
from utils import world_to_grid, grid_to_world
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt, generic_filter
import matplotlib.patches as patches
import heapq

# 第一步：加载本地点云
file_path = "20250330.xyz"
xyz_points = np.loadtxt(file_path)
min_x = np.min(xyz_points[:, 0])
min_y = np.min(xyz_points[:, 1])
max_x = np.max(xyz_points[:, 0])
max_y = np.max(xyz_points[:, 1])
global_xx_list = []
global_yy_list = []
global_mean_list = []
global_slope_list = []
global_var_list = []
resolution = 0.4
# 构建栅格函数（保持不变）
large_size = 0.8
small_size = 0.4


# 1. 提取地图边界范围

# 3. 构建 X、Y 方向的栅格坐标轴
large_grid_x = np.arange(min_x, max_x + large_size, large_size)
large_grid_y = np.arange(min_y, max_y + large_size, large_size)
small_grid_x = np.arange(min_x, max_x + small_size, small_size)
small_grid_y = np.arange(min_y, max_y + small_size, small_size)

# 4. 构建 meshgrid 网格中心点坐标对（可用于可视化、点映射）
large_grid = np.array(np.meshgrid(large_grid_x, large_grid_y)).T.reshape(-1, 2)
small_grid = np.array(np.meshgrid(small_grid_x, small_grid_y)).T.reshape(-1, 2)

# 5. 获取主地图尺寸（X方向列数，Y方向行数）
num_x = int(np.ceil((max_x - min_x) / resolution)) + 1
num_y = int(np.ceil((max_y - min_y) / resolution)) + 1




# 第二步：SGP 模型定义
class SGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=torch.tensor([0.7, 0.7]))
        )
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel,
            inducing_points=inducing_points,
            likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Node:
    def __init__(self, point, parent=None, cost=0, relative_angle=0):
        self.point = point
        self.parent = parent
        self.cost = cost
        self.relative_angle = relative_angle
        

# 建立 (x,y) 到栅格坐标 (i,j) 的映射函数            ix, iy = int(x / self.resolution), int(y / self.resolution)


# 运行单步 SGP 建图流程
def run_sgp_mapping(x0, y0, local_range=5):
    #print(f"当前小车位置: ({x0}, {y0})")
    distance_sq = (xyz_points[:, 0] - x0) ** 2 + (xyz_points[:, 1] - y0) ** 2
    mask = distance_sq <= local_range ** 2
    local_points = xyz_points[mask]

    X_train_np = local_points[:, :2]
    y_train_np = local_points[:, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)

    inducing_num = min(500, X_train.shape[0])
    idx = np.linspace(0, X_train.shape[0] - 1, inducing_num, dtype=int)
    inducing_pts = X_train[idx, :]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SGPModel(X_train, y_train, likelihood, inducing_pts).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    start_time = time()
    for i in range(1):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    end_time = time()

    training_time = end_time - start_time
    #print(f"模型训练时间: {training_time:.3f} 秒")

    x_lin = np.linspace(x0 - local_range, x0 + local_range, 40)
    y_lin = np.linspace(y0 - local_range, y0 + local_range, 40)
    xx, yy = np.meshgrid(x_lin, y_lin)
    X_pred_np = np.column_stack([xx.ravel(), yy.ravel()])
    X_pred = torch.tensor(X_pred_np, dtype=torch.float32, requires_grad=True, device=device)

    model.eval()
    likelihood.eval()
    with torch.autograd.set_grad_enabled(True):
        preds = likelihood(model(X_pred))

    mean = preds.mean.detach().cpu().numpy().reshape(xx.shape)
    var = preds.variance.detach().cpu().numpy().reshape(xx.shape)
    grad_outputs = torch.ones_like(preds.mean)
    grad_mean = torch.autograd.grad(preds.mean, X_pred, grad_outputs=grad_outputs, retain_graph=True)[0]
    slope = torch.norm(grad_mean, dim=1).detach().cpu().numpy().reshape(xx.shape)
    #slope = np.arctan(slope)

    train_preds = likelihood(model(X_train)).mean.detach().cpu().numpy()
    train_y_np = y_train.cpu().numpy()
    mse = mean_squared_error(train_y_np, train_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(train_y_np, train_preds)
    print(f"训练集的RMSE为: {rmse:.4f}")
    print(f"训练集的R²为: {r2:.4f}")

     # --- 统一到 0.4m 栅格 ---
    resolution = 0.4
    grid_x = np.arange(xx.min(), xx.max() + 1e-6, resolution)
    grid_y = np.arange(yy.min(), yy.max() + 1e-6, resolution)
    xx_grid, yy_grid = np.meshgrid(grid_x, grid_y)

    mean_grid = griddata((xx.flatten(), yy.flatten()), mean.flatten(), (xx_grid, yy_grid), method='nearest')
    slope_grid = griddata((xx.flatten(), yy.flatten()), slope.flatten(), (xx_grid, yy_grid), method='nearest')
    var_grid = griddata((xx.flatten(), yy.flatten()), var.flatten(), (xx_grid, yy_grid), method='nearest')

    # --- 计算局部特征图 ---
    flatness_map = compute_flatness(mean_grid, resolution=0.4, physical_window=1.2)
    step_height_map = compute_step_height(mean_grid, window_size=3)

    # --- 生成局部语义地图 + 通行性地图 ---
    semantic_map = generate_semantic_map(
        mean=mean_grid,
        slope=slope_grid,
        var=var_grid,
        step_map=step_height_map,
        flatness_map=flatness_map,
        resolution=0.4
    )

    trav_map = compute_traversability_map(
        step_map=step_height_map,
        flatness_map=flatness_map,
        slope_map=slope_grid,
        var_map=var_grid,
        semantic_map=semantic_map,
        w_step=0.3,
        w_flat=0.5,
        w_slope=0.2
    )

    return xx_grid, yy_grid, mean_grid, slope_grid, var_grid, flatness_map, step_height_map, semantic_map, trav_map, grad_mean,rmse,r2



def update_global_maps_from_sgp(xx_grid, yy_grid, mean_grid, slope_grid, var_grid, semantic_map, trav_map, grad_mean,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map):
    """
    将局部的预测结果更新到全局地图。
    """
    margin = 0  
    center_x = np.mean(xx_grid)
    center_y = np.mean(yy_grid)
    radius = np.min([xx_grid.max() - center_x, yy_grid.max() - center_y]) - 0.5
    geo_mask = ((xx_grid - center_x)**2 + (yy_grid - center_y)**2) <= radius**2

    for i in range(margin, xx_grid.shape[0] - margin):
        for j in range(margin, xx_grid.shape[1] - margin):
            if not geo_mask[i, j]:
                continue
            x_coord = xx_grid[i, j]
            y_coord = yy_grid[i, j]
            grid_i, grid_j = world_to_grid(x_coord, y_coord)

            if 0 <= grid_i < global_mean_map.shape[0] and 0 <= grid_j < global_mean_map.shape[1]:
                if np.isnan(mean_grid[i, j]):
                    print(f"📛 mean_grid 为 NaN：world=({x_coord:.2f}, {y_coord:.2f}) → grid=({grid_i}, {grid_j})，跳过写入")
                    continue
                if global_frozen_map[grid_i, grid_j]:
                    continue  # 冻结点不更新
                global_mean_map[grid_i, grid_j] = mean_grid[i, j]
                global_slope_map[grid_i, grid_j] = slope_grid[i, j]
                global_roughness_map[grid_i, grid_j] = var_grid[i, j]
                global_envrionment_map[grid_i, grid_j] = semantic_map[i, j]
                global_traversability_map[grid_i, grid_j] = trav_map[i, j]
                global_frozen_map[grid_i, grid_j] = True

                                # ✅ 额外更新法向量地图（如果提供了grad_mean）
                if grad_mean is not None:
                    # grad_mean shape 是 (H*W, 2)，所以要转换一下索引
                    idx_flat = i * xx_grid.shape[1] + j
                    dzdx = grad_mean[idx_flat, 0].item()
                    dzdy = grad_mean[idx_flat, 1].item()

                    normal_vec = np.array([-dzdx, -dzdy, 1.0])
                    normal_norm = np.linalg.norm(normal_vec)

                    if normal_norm > 1e-6:  # 避免除以0
                        normal_vec /= normal_norm
                        global_normal_map[grid_i, grid_j] = normal_vec
                    else:
                        # 如果梯度异常，就设成默认垂直向上
                        global_normal_map[grid_i, grid_j] = np.array([0.0, 0.0, 1.0])
            else:
                print(f"⚠️ 越界坐标 ({x_coord:.2f}, {y_coord:.2f}) → (i={grid_i}, j={grid_j})，地图尺寸=({global_mean_map.shape[0]}, {global_mean_map.shape[1]})")


def is_path_fully_explored(path_indices, traversability_map):
    for i, j in path_indices:
        if i < 0 or i >= traversability_map.shape[0] or j < 0 or j >= traversability_map.shape[1]:
            return False  # 越界视为未探索
        if np.isnan(traversability_map[i, j]):
            return False
    return True


def extract_water_edge_from_semantic_map(semantic_map, traversability_map=None, water_label=1):
    assert semantic_map.ndim == 2, "语义图必须是二维的"

    # 1. 提取水域掩码
    water_mask = (semantic_map == water_label)

    # 2. 四邻域结构（只膨胀上下左右）
    # structure = np.array([[0, 1, 0],
    #                       [1, 1, 1],
    #                       [0, 1, 0]])
    structure = np.ones((3, 3))
    # 3. 水域四邻域膨胀（相邻区域）
    dilated = binary_dilation(water_mask, structure=structure)

    # 4. 非水域 且 与水域相邻
    candidate_edge = np.logical_and(dilated, ~water_mask)

    # 5. 如果有通行性地图，则要求边缘像素为“已知”（不是NaN）
    if traversability_map is not None:
        known_mask = ~np.isnan(traversability_map)
        water_edge_mask = np.logical_and(candidate_edge, known_mask)
    else:
        water_edge_mask = candidate_edge

    return water_edge_mask




def detect_structural_blockage_from_grid(
    current_grid, next_grid,
    edge_points, resolution=0.4,
    min_x=0, min_y=0,
    R=4, alpha_deg=120, bin_count=12
):
    
    if edge_points is None or len(edge_points) == 0:
        return False, 0, np.empty((0, 2), dtype=int), 0, 0

    # 1. 坐标转换
    x0 = min_x + current_grid[0] * resolution
    y0 = min_y + current_grid[1] * resolution
    x1 = min_x + next_grid[0] * resolution
    y1 = min_y + next_grid[1] * resolution

    # 2. 水域边缘点 → 世界坐标
    edge_points_world = np.array([
        [min_x + ix * resolution, min_y + iy * resolution]
        for ix, iy in edge_points
    ])

    # 3. 计算前进方向（heading）
    heading = np.arctan2(y1 - y0, x1 - x0)

    # 4. 向量差值
    dx = edge_points_world[:, 0] - x0
    dy = edge_points_world[:, 1] - y0
    distances = np.sqrt(dx**2 + dy**2)

    # 5. 半径限制
    mask = distances <= R
    dx = dx[mask]
    dy = dy[mask]
    if len(dx) == 0:
        return False, 0, np.empty((0, 2), dtype=int), 0, 0

    # 6. 所有点相对于 heading 的角度差（单位：度）
    relative_angles = np.arctan2(dy, dx) - heading
    relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi
    relative_angles_deg = np.degrees(relative_angles)

    # 7. 限制在 ±(alpha/2) 扇形区域
    alpha_half = alpha_deg / 2.0
    sector_mask = (relative_angles_deg >= -alpha_half) & (relative_angles_deg <= alpha_half)
    angles_in_sector = relative_angles_deg[sector_mask]
    if len(angles_in_sector) == 0:
        return False, 0, np.empty((0, 2), dtype=int), 0, 0

    # 8. 将角度映射到 bin
    bin_edges = np.linspace(-alpha_half, alpha_half, bin_count + 1)
    bin_counts, _ = np.histogram(angles_in_sector, bins=bin_edges)
    
    angle_span = np.ptp(angles_in_sector)           # ≡ max - min
    span_blocked = angle_span > 70     # bool
    
    # 9. 判定结构性阻碍（大于等于8个 bin 被占用）
    bin_coverage = np.sum(bin_counts > 0)
    is_blocked = (bin_coverage >= 8) or span_blocked 

    # ✅ 9.5 统计左右两侧 bin 占用情况
    half = bin_count // 2
    left_occupied_bins = np.sum(bin_counts[:half] > 0)
    right_occupied_bins = np.sum(bin_counts[half:] > 0)

    # 10. 提取扇形内边缘点
    sector_indices = np.where(sector_mask)[0]
    candidate_cross_grids = edge_points[mask][sector_indices]

    return is_blocked, bin_coverage, candidate_cross_grids, left_occupied_bins, right_occupied_bins






def extract_candidate_cross_points_in_radius(
    current_grid,
    edge_points,
    global_envrionment_map,
    resolution=0.4,
    radius=4.0
):
    if edge_points is None or len(edge_points) == 0:
        return np.empty((0, 2), dtype=int)

    dist_cells = radius / resolution
    diffs = edge_points - np.array(current_grid)
    dists = np.linalg.norm(diffs, axis=1)
    mask_radius = dists <= dist_cells

    filtered_points = edge_points[mask_radius]

    # 再排除障碍物（global_envrionment_map 中为 0 的）
    valid_mask = []
    for pt in filtered_points:
        ix, iy = pt
        if 0 <= ix < global_envrionment_map.shape[0] and 0 <= iy < global_envrionment_map.shape[1]:
            if global_envrionment_map[ix, iy] != 0:
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        else:
            valid_mask.append(False)

    valid_edge_points = filtered_points[np.array(valid_mask)]
    return valid_edge_points


def compute_true_path_length(path_indices, resolution=0.4):
    path_length = 0.0
    for k in range(1, len(path_indices)):
        dx = path_indices[k][0] - path_indices[k-1][0]
        dy = path_indices[k][1] - path_indices[k-1][1]
        step_dist = np.sqrt(dx**2 + dy**2) * resolution
        path_length += step_dist
    return path_length


def car_angle(point_normal, t_in=0.0):
    point_normal = np.array(point_normal).reshape(3, 1)  # 转为列向量

    # 初始化 X/Y 朝向向量
    x_1 = np.array([[np.cos(t_in)], [np.sin(t_in)], [0.0]])
    y_1 = np.array([[-np.sin(t_in)], [np.cos(t_in)], [0.0]])

    if np.allclose(x_1, 0):
        x_1 = np.array([[1.0], [0.0], [0.0]])

    normal_size = np.dot(x_1.T, point_normal)
    normal_dir = -1 if normal_size > 0 else 1

    z_2 = point_normal
    fenzi = np.cross(y_1.T, z_2.T).T
    x_2 = fenzi / (np.linalg.norm(fenzi) + 1e-8)
    y_2 = np.cross(z_2.T, x_2.T).T

    rotation_matrix = np.hstack((x_2, y_2, z_2))  # 3x3 matrix

    # rotm2eul: ZYX
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Roll
        y = np.arctan2(-rotation_matrix[2, 0], sy)                    # Pitch
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Yaw
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    # 修正范围限制为 [-pi/2, pi/2]
    if abs(y) > np.pi / 2:
        y = np.sign(y) * (np.pi - abs(y))
    if abs(x) > np.pi / 2:
        x = np.sign(x) * (np.pi - abs(x))

    euler_angle = normal_dir * np.array([z, y, x])
    return euler_angle  # 返回 ZYX 欧拉角



def compute_path_total_cost(
    path_indices,
    cross_point,
    goal_grid,
    global_traversability_map,
    global_envrionment_map,
    global_mean_map,
    risk,
    water_edge_mask,
    resolution=0.4,
    alpha1=1.0,  # 路径长度
    alpha2=2.0,  # 通行性代价
    alpha3=3.0,  # 水域边缘惩罚
    gamma=1.5    # 目标距离
):
    path_length = compute_true_path_length(path_indices=path_indices, resolution=0.4)

    # 通行性代价总和
    trav_cost = 0.0
    edge_penalty = 0.0
    for i, j in path_indices:
        if 0 <= i < global_traversability_map.shape[0] and 0 <= j < global_traversability_map.shape[1]:
            if not np.isnan(global_traversability_map[i, j]):
                trav_cost += global_traversability_map[i, j]
            if water_edge_mask[i, j]:  # 靠近水域边缘区域
                edge_penalty += 1.0

    # 跨域代价
    cross_cost = risk

    # 目标距离代价
    goal_dist = np.linalg.norm((np.array(cross_point) - np.array(goal_grid)) * resolution)

    # 总路径代价
    path_cost = alpha1 * path_length + alpha2 * trav_cost + alpha3 * edge_penalty
    total_cost = path_cost + cross_cost + goal_dist

    return total_cost



def compute_cross_domain_risk(pitch, roll, elevation_diff):
    term_pitch = 1.0 / (1.0 + np.exp(-((abs(pitch) * (60.0 / np.pi)) - 8.0)))
    term_roll = 1.0 / (1.0 + np.exp(-((abs(roll) * (70.0 / np.pi)) - 9.0)))
    term_elevation = 0.2 * elevation_diff

    total_risk = term_pitch + term_roll + term_elevation
    return total_risk

def get_euler_from_normal(normal_vec, direction_angle=0.0):
    return car_angle(normal_vec, direction_angle)

def filter_cross_points_by_angle(
    global_mean_map,
    current_grid,
    candidate_cross_grids,
    global_normal_map,
    global_envrionment_map,
    resolution=0.4,
    pitch_thresh=0.45,
    roll_thresh=0.38
):
    pitch_dict = {}
    roll_dict = {}
    risk_dict = {}
    valid_points = []
    H, W, _ = global_normal_map.shape
    ci, cj = current_grid
    env_curr = global_envrionment_map[ci, cj]

    for pt in candidate_cross_grids:
        i, j = pt
        env_target = global_envrionment_map[i, j]

        if env_target == env_curr and env_target  in [1, 2]:
            print("🎯 起点在陆地")
        elif env_target != env_curr and env_target  in [1, 2]:
            print("🎯 起点在水域")
        else:
            continue
        normal_vec = global_normal_map[i, j]
        if np.linalg.norm(normal_vec) < 1e-3 or np.isnan(normal_vec).any():
            continue
        has_valid_direction = False  # 是否真的检测过
        max_pitch, max_roll,elevation_diff = 0.0, 0.0,0.0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_env = global_envrionment_map[ni, nj]
                    if neighbor_env != env_target and neighbor_env == 1:
                        elevation = abs(global_mean_map[i, j] - global_mean_map[ni, nj])
                        has_valid_direction = True
                        direction_vec = np.array([ni - i, nj - j]) 
                        direction_angle = np.arctan2(direction_vec[1], direction_vec[0])
                        euler = get_euler_from_normal(normal_vec, direction_angle)
                        pitch, roll = abs(euler[1]), abs(euler[2])
                        max_pitch = max(max_pitch, pitch)
                        max_roll = max(max_roll, roll)
                        elevation_diff=max(elevation_diff,elevation)

        if has_valid_direction and max_pitch <= pitch_thresh and max_roll <= roll_thresh:
            cross_risk = compute_cross_domain_risk(max_pitch, max_roll, elevation_diff)
            pitch_dict[(i, j)] = max_pitch
            roll_dict[(i, j)] = max_roll
            risk_dict[(i, j)] = cross_risk
            valid_points.append([i, j])

    return np.array(valid_points), pitch_dict, roll_dict, risk_dict











def select_virtual_goal_adaptive(current_xy, goal_xy,direction_vec, water_edge_mask, resolution=0.4, k=1.2):
    
    current_xy = grid_to_world(current_xy[0], current_xy[1], min_x, min_y, resolution)
    goal_xy = grid_to_world(goal_xy[0], goal_xy[1], min_x, min_y, resolution)
    x1, y1 = current_xy
    x2, y2 = goal_xy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l = np.linalg.norm([x2 - x1, y2 - y1])
    a = k * l / 2
    b = np.sqrt(k**2 - 1) * l / 2
    angle = np.arctan2(y2 - y1, x2 - x1)
    mid_lower = (cx + b * np.sin(angle), cy - b * np.cos(angle))  # 原来的 upper
    mid_upper = (cx - b * np.sin(angle), cy + b * np.cos(angle))  # 原来的 lower
    direction_vec = direction_vec / (np.linalg.norm(direction_vec) + 1e-8)
    radius = 5.0  
    edge_indices = np.argwhere(water_edge_mask)
    edge_points = []
    for i, j in edge_indices:
        x = i * resolution + min_x
        y = j * resolution + min_y
        if np.linalg.norm([x - x1, y - y1]) <= radius:
            edge_points.append([x, y])
    edge_points = np.array(edge_points)

    up_angles = []
    down_angles = []
    for pt in edge_points:
        vec = pt - np.array([x1, y1])
        if np.linalg.norm(vec) < 1e-3:
            continue
        vec_unit = vec / (np.linalg.norm(vec) + 1e-8)
        dot = np.clip(np.dot(vec_unit, direction_vec), -1.0, 1.0)
        angle_deg = np.arccos(dot) * 180 / np.pi

        # 判断是上/下（使用向量叉乘方向）
        cross = np.cross(direction_vec, vec_unit)
        if cross > 0:
            up_angles.append(angle_deg)
        else:
            down_angles.append(angle_deg)

    # 比较最大夹角
    max_up = max(up_angles) if up_angles else 180
    max_down = max(down_angles) if down_angles else 180

    if max_up < max_down:
        chosen_x, chosen_y = mid_upper
    else:
        chosen_x, chosen_y = mid_lower
    print(f"up_angles: {up_angles}")
    print(f"down_angles: {down_angles}")
    print(f"Max Up Angle: {max_up:.2f}°, Max Down Angle: {max_down:.2f}°")
    i, j=world_to_grid(chosen_x,chosen_y)
    return i, j





def select_virtual_goal_refined(current_grid, traversability_map, water_edge_mask,
                                resolution=0.4, search_radius=5.2,
                                min_water_dist=2.0, max_water_dist=3.2):
    h, w = traversability_map.shape
    ci, cj = current_grid
    r_cells = int(search_radius / resolution)

    # ✅ 计算局部窗口范围
    i_min = max(ci - r_cells, 0)
    i_max = min(ci + r_cells + 1, h)
    j_min = max(cj - r_cells, 0)
    j_max = min(cj + r_cells + 1, w)

    # ✅ 裁剪局部地图
    local_trav = traversability_map[i_min:i_max, j_min:j_max]
    local_water = (local_trav == 0)

    # ✅ 计算当前位置到局部所有点的欧式距离（单位：米）
    xx, yy = np.meshgrid(np.arange(i_min, i_max), np.arange(j_min, j_max), indexing='ij')
    dist_to_current = np.sqrt((xx - ci)**2 + (yy - cj)**2) * resolution
    radius_mask = dist_to_current <= search_radius

    # ✅ 通行性有效（0 < cost < 1）
    valid_cost_mask = (local_trav > 0) & (local_trav < 1)

    # ✅ 四邻域中是否包含 NaN（代表邻接未知区域）
    nan_neighbor_mask = ~np.isnan(local_trav) & generic_filter(
        np.isnan(local_trav).astype(int), 
        lambda x: any(x[[1, 3, 5, 7]]),  # 上下左右位置
        size=3,
        mode='constant',
        cval=0
    ).astype(bool)

    # ✅ 组合得到候选点掩码
    candidate_mask = valid_cost_mask & nan_neighbor_mask & radius_mask

    if not np.any(candidate_mask):
        print("❌ 没有满足邻接未知区域的候选点")
        return None
    candidate_indices = np.argwhere(candidate_mask)
    print(f"✅ 满足条件的候选点数量: {len(candidate_indices)}")

    for idx in candidate_indices:
        global_i = idx[0] + i_min
        global_j = idx[1] + j_min
        print("候选点栅格坐标 (i, j):", (global_i, global_j))
    # ✅ 距离最近水域边缘距离图（单位：米）
    
    dist_to_water = distance_transform_edt(1 - local_water) * resolution
    print("dist_to_water min/max:", dist_to_water.min(), dist_to_water.max())
    water_dist_mask = (dist_to_water >= min_water_dist) & (dist_to_water <= max_water_dist)

    # ✅ 最终候选点：同时满足三重条件
    final_mask = candidate_mask & water_dist_mask
    final_indices = np.argwhere(final_mask)
    if final_indices.size > 0:
        for idx in final_indices:
            real_i = idx[0] + i_min
            real_j = idx[1] + j_min
            print("✅ 候选虚拟目标点栅格坐标:", (real_i, real_j))
    else:
        print("❌ 没有满足条件的虚拟目标点")
    if not np.any(final_mask):
        print("❌ 没有满足距离水域条件的候选点")
        return None

    # ✅ 从最终候选点中选择通行性最小的作为虚拟目标
    final_indices = np.where(final_mask)
    best_idx = np.argmin(local_trav[final_indices])
    best_i = final_indices[0][best_idx] + i_min
    best_j = final_indices[1][best_idx] + j_min

    return (best_i, best_j)


def try_virtual_goal_path(current_grid, virtual_goal_grid,
                          global_traversability_map, 
                          global_envrionment_map, global_normal_map):
    # ✅ 直接传入栅格坐标，不再使用 Point 或 world_to_grid
    planner = DStarLite(current_grid, virtual_goal_grid,
                        global_traversability_map,
                        global_envrionment_map,
                        global_normal_map,
                        use_weighted=True)

    planner.compute_shortest_path()
    path_indices = planner.extract_path()

    # ✅ 判断路径是否可达
    if len(path_indices) < 2:
        return False, path_indices  # 无法生成路径

    # ✅ 判断是否路径中有未建图区域（NaN）
    if not is_path_fully_explored(path_indices, global_traversability_map):
        return False, path_indices

    return True, path_indices


def filter_points_by_obstacle_proximity(
    valid_cross_points,
    global_traversability_map,
    max_obstacle_count=2
):
    filtered_points = []
    H, W = global_traversability_map.shape

    for pt in valid_cross_points:
        i, j = pt
        obstacle_count = 0

        # 四邻域（上、下、左、右）
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                val = global_traversability_map[ni, nj]
                if val >= 1 or np.isnan(val):
                    obstacle_count += 1

        if obstacle_count <= max_obstacle_count:
            filtered_points.append([i, j])

    return np.array(filtered_points)


import heapq
import numpy as np

def compute_all_segment_costs(current, 
                              global_trav, global_env, water_edge_mask,candidate_points,
                              α1=1.0, α2=2.0, α3=3.0,
                              resolution=0.4,
                              max_radius_m=8.0):
    """
    从 current 格点出发做一次 Dijkstra，返回归一化后的 dist_map：
      dist_map[i,j] = minimal ∑[ α1*d + α2*trav + α3*edge ] 到 (i,j)
    只搜索以 current 为中心、半径 max_radius_m 米范围内的节点。
    """
    H, W = global_trav.shape
    # 8 邻域偏移及其欧氏距离（单位：格点数）
    neighs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
              (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),
              (1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]

    cx, cy = current
    # 提前计算半径对应的格点数（浮点数），并取平方
    max_radius_cells_sq = (max_radius_m / resolution) ** 2

    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    dist_map[cx, cy] = 0.0

    # 小顶堆： (累积代价, i, j)
    heap = [(0.0, cx, cy)]

    while heap:
        cost_u, ui, uj = heapq.heappop(heap)
        if cost_u > dist_map[ui, uj]:
            continue

        for di, dj, dlen in neighs:
            vi, vj = ui + di, uj + dj
            # 边界检查
            if not (0 <= vi < H and 0 <= vj < W):
                continue

            # 距离中心超出半径 5m？
            # 这里比较 (vi-cx)^2 + (vj-cy)^2 > (max_radius_cells)^2


            # 该格点必须有效且可通行
            if np.isnan(global_trav[vi, vj]):
                continue

            # 计算权重
            w = α1 * dlen \
                + α2 * global_trav[vi, vj] \
                + α2 * (1.0 if water_edge_mask[vi, vj] else 0.0)

            new_cost = cost_u + w
            if new_cost < dist_map[vi, vj]:
                dist_map[vi, vj] = new_cost
                heapq.heappush(heap, (new_cost, vi, vj))

    norm_val = None
    if candidate_points is not None:
        cand_costs = [dist_map[i, j]
                      for (i, j) in candidate_points
                      if 0 <= i < H and 0 <= j < W and np.isfinite(dist_map[i, j])]
        if cand_costs:                       # 至少有一个有效值
            norm_val = max(cand_costs)

    # 2) 如果候选点都无效，则用全图 finite 最大值
    if norm_val is None:
        finite_vals = dist_map[np.isfinite(dist_map)]
        if finite_vals.size:
            norm_val = finite_vals.max()

    # 3) 最终归一化
    if norm_val and norm_val > 0:
        dist_map /= norm_val

    return dist_map




def select_best_cross_point_batch(
    current_grid, initial_goal_grid,initial_start_grid, candidate_points,
    global_trav, global_env, water_edge_mask,
    risk_dict, α1=1.0, α2=2.0, α3=3.0, γ=1.5, resolution=0.4,obstacle_cost_dict=None
):
    # 1) 一次 Dijkstra，拿到所有段成本,已完成归一化
    print("开始寻找最佳跨域点")
    dist_map = compute_all_segment_costs(
        current_grid,
        global_trav, global_env, water_edge_mask,candidate_points,
        α1, α2, α3
    )
    # 检查是否存在有限值（非 inf 和非 NaN）
    has_valid_values = np.isfinite(dist_map).any()

    if has_valid_values:
        print("dist_map 不全为空，存在有效路径。")
    else:
        print("dist_map 全为空（所有值为 inf 或 NaN），没有找到有效路径。")
    print("开始寻找最佳跨域点1")

    best_pt, best_cost = None, np.inf
    gx, gy = initial_goal_grid
    sx,sy=initial_start_grid

    # 2) 批量读候选点成本
    for (ai, aj) in candidate_points:
        seg_cost = dist_map[ai, aj]
        
        print(seg_cost)

        if np.isinf(seg_cost):
            continue  # 不可达

        cross_cost = risk_dict[(ai, aj)]# 已完成近似归一化
        # 计算候选点到目标点的欧氏距离
        candidate_to_goal_dist = np.linalg.norm(np.array([ai, aj]) - np.array([gx, gy]))

        candidate_to_start_dist = np.linalg.norm(np.array([ai, aj]) - np.array([sx, sy]))
        # 计算起点到目标点的欧氏距离
        start_to_goal_dist = np.linalg.norm(np.array([sx, sy]) - np.array([gx, gy]))

        # 安全地计算距离比值，处理分母为零的情况
        if start_to_goal_dist > 0:
            start_goal_cost = (candidate_to_goal_dist) / start_to_goal_dist
        else:
            # 起点和目标点重合的特殊情况，可根据需求处理
            start_goal_cost = 1

        if obstacle_cost_dict is not None:
            total = cross_cost + seg_cost + obstacle_cost_dict[(ai, aj)] + start_goal_cost
        else: 
            total = cross_cost  + seg_cost + start_goal_cost
        print("total")
        print(total)

        if total < best_cost:
            best_cost, best_pt = total, (ai, aj)
    print("开始寻找最佳跨域点2")
    return best_pt


def compute_obstacle_influence_cost(
    current_grid, next_grid,
    obstacle_points,
    resolution=0.4,
    min_x=0, min_y=0,
    R=4,
    alpha_deg=120,
    bin_count=12,
    mid_point=6
):
    if obstacle_points is None or len(obstacle_points) == 0:
        return 0.0, 0, np.empty((0,2),dtype=int), 0, 0

    # 1. 当前点、目标点的世界坐标
    x0 = min_x + current_grid[0] * resolution
    y0 = min_y + current_grid[1] * resolution
    x1 = min_x + next_grid[0]    * resolution
    y1 = min_y + next_grid[1]    * resolution

    # 2. 障碍物格点 → 世界坐标
    pts_world = np.stack([
        min_x + obstacle_points[:,0]*resolution,
        min_y + obstacle_points[:,1]*resolution
    ], axis=1)

    # 3. 前进方向（heading）
    heading = np.arctan2(y1 - y0, x1 - x0)

    # 4. 向量差与距离
    dx = pts_world[:,0] - x0
    dy = pts_world[:,1] - y0
    dists = np.hypot(dx, dy)

    # 5. 半径筛选
    mask_r = (dists <= R)
    dx = dx[mask_r]; dy = dy[mask_r]
    pts_sel  = obstacle_points[mask_r]  # 对应的格点索引
    if dx.size == 0:
        return 0.0, 0, np.empty((0,2),dtype=int), 0, 0

    # 6. 计算相对角度（°），归一到 [-180,180]
    rel_ang = np.degrees(((np.arctan2(dy,dx) - heading + np.pi)%(2*np.pi)) - np.pi)

    # 7. 扇形区域筛选
    half = alpha_deg/2.0
    mask_sector = (rel_ang >= -half) & (rel_ang <= half)
    ang_in = rel_ang[mask_sector]
    pts_in = pts_sel[mask_sector]
    if ang_in.size == 0:
        return 0.0, 0, np.empty((0,2),dtype=int), 0, 0

    # 8. 角度分 bin 统计
    edges = np.linspace(-half, half, bin_count+1)
    counts, _ = np.histogram(ang_in, bins=edges)

    # 9. 计算 B(p) 及左右占用
    B_p = int((counts>0).sum())
    left_bins  = int((counts[:bin_count//2]>0).sum())
    right_bins = int((counts[bin_count//2:]>0).sum())

    # 10. 用 Sigmoid 映射成 [0,1]
    C_obst = 1.0 / (1.0 + np.exp(-(B_p - mid_point)))

    return C_obst, B_p, pts_in, left_bins, right_bins




def select_best_cross_domain_point(
    water_edge_mask,
    initial_goal_grid,
    initial_start_grid,
    current_grid,
    candidate_cross_grids,
    global_traversability_map,
    global_envrionment_map,
    global_normal_map,
    global_mean_map,
    weight_dist=1.0,
    weight_rough=2.0,
    resolution=0.4
):
    if candidate_cross_grids is None or len(candidate_cross_grids) == 0:
        return None
    safe_points = filter_points_by_obstacle_proximity(
    valid_cross_points=candidate_cross_grids,
    global_traversability_map=global_traversability_map,
    max_obstacle_count=2  # 超过两个邻居为障碍物就排除
    )
    print("safe_points")
    print(safe_points)

    if safe_points is not None:
        valid_cross_points, pitch_dict, roll_dict, risk_dict = filter_cross_points_by_angle(
            global_mean_map=global_mean_map,
            current_grid=current_grid,
            candidate_cross_grids=safe_points,
            global_normal_map=global_normal_map,
            global_envrionment_map=global_envrionment_map,
            resolution=0.4,              # 栅格分辨率（单位：米）
            pitch_thresh=0.45,           # 最大允许俯仰角（单位：弧度）
            roll_thresh=0.38            # 最大允许侧倾角（单位：弧度）
        )
    else:
        return None
    
    if valid_cross_points is not None:
        temp=global_envrionment_map[current_grid[0],current_grid[1]]
        best_point = None
        if temp==1:
            print("所在位置为水域，进行上岸点选择")
            obstacle_cost_dict = {}
            for pt in valid_cross_points:        # pt 是 (i,j) 格点
                C_obst, B_p, _, left_bins, right_bins = compute_obstacle_influence_cost(
                    current_grid=pt,
                    next_grid=initial_goal_grid,
                    obstacle_points=np.argwhere(global_envrionment_map == 0),   # 你的全局障碍物点列表
                    resolution=0.4,
                    min_x=0, min_y=0,
                    R=4,
                    alpha_deg=120,
                    bin_count=12,
                    mid_point=6
                )
                obstacle_cost_dict[(pt[0], pt[1])] = C_obst
            best_point =select_best_cross_point_batch(current_grid, initial_goal_grid, initial_start_grid, valid_cross_points,global_traversability_map,global_envrionment_map,water_edge_mask,risk_dict,obstacle_cost_dict=obstacle_cost_dict) 
            return best_point 
        else:
            print("所在位置为陆地，进行下水点选择")
            best_point =select_best_cross_point_batch(current_grid, initial_goal_grid,initial_start_grid, valid_cross_points,global_traversability_map,global_envrionment_map,water_edge_mask,risk_dict)
            return best_point
    else:
        return None











# SGP 模型定义
class SGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=torch.tensor([0.7, 0.7]))
        )
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel,
            inducing_points=inducing_points,
            likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def is_point_valid(point, global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map,local_range=5):
    result = run_sgp_mapping(point.x, point.y, local_range=local_range)
    if result is not None:
        xx_grid, yy_grid, mean_grid, slope_grid, var_grid, flatness_map, step_height_map, semantic_map, trav_map, grad_mean,rmse,r2 = result
        update_global_maps_from_sgp(xx_grid, yy_grid, mean_grid, slope_grid, var_grid, semantic_map, trav_map, grad_mean,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map)
        i_center, j_center = world_to_grid(point.x, point.y)

        if (0 <= i_center < global_envrionment_map.shape[0]) and (0 <= j_center < global_envrionment_map.shape[1]):
            env_value = global_envrionment_map[i_center, j_center]
            if env_value == 0:
                print(f"❌ 位置 ({point.x}, {point.y}) 是障碍物，无法作为起点或终点。")
                return False
            elif env_value == 255:
                print(f"⚠️ 位置 ({point.x}, {point.y}) 无法识别环境类型。")
                return False
            else:
                print(f"✅ 位置 ({point.x}, {point.y}) 合法，环境类型：{env_value}（1=水，2=陆）")
                return True
    return False






def draw_ellipse(ax, point1, point2, k=1.2, color='green', linestyle='--'):
    x1, y1 = point1
    x2, y2 = point2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    a = k * l / 2
    b = (np.sqrt(k**2 - 1) * l / 2)*1
    angle = np.arctan2(y2 - y1, x2 - x1)
    t = np.linspace(0, 2 * np.pi, 300)
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = R @ np.vstack((ellipse_x, ellipse_y))
    ax.plot(rotated[0, :] + cx, rotated[1, :] + cy, linestyle=linestyle, color=color, linewidth=2)


def visualize_path_local_traversability1(trav_map, path_indices,path_traditional,path_weighted_only, min_x, min_y, resolution=0.4,
                                         local_margin=5.0, save_path="path_local_traversability_heatmap.png",
                                         start_grid=None, goal_grid=None, current_grid=None, virtual_goal=None, virtual_path=None):

    if path_indices is None or len(path_indices) == 0:
        print("path路径为空，无法绘制")
        return

    path = np.array(path_indices)  # 栅格索引：(i, j)
    all_indices = list(path_indices)  # 保证是 Python 列表

    if start_grid is not None:
        all_indices.append(start_grid)
    if goal_grid is not None:
        all_indices.append(goal_grid)
    if current_grid is not None:
        all_indices.append(current_grid)

    # ✅ 再转成 NumPy 数组以便裁剪索引
    all_indices = np.array(all_indices)
    # ✅ 计算裁剪范围
    margin_cells = 2*int(local_margin / resolution)
    i_min = max(all_indices[:, 0].min() - margin_cells, 0)
    i_max = min(all_indices[:, 0].max() + margin_cells + 1, trav_map.shape[0])
    j_min = max(all_indices[:, 1].min() - margin_cells, 0)
    j_max = min(all_indices[:, 1].max() + margin_cells + 1, trav_map.shape[1])

    cropped_trav_map = trav_map[i_min:i_max, j_min:j_max]

    # ✅ 构建真实坐标轴
    x_ticks = np.linspace(min_x + i_min * resolution, min_x + (i_max - 1) * resolution, cropped_trav_map.shape[0])
    y_ticks = np.linspace(min_y + j_min * resolution, min_y + (j_max - 1) * resolution, cropped_trav_map.shape[1])
    xx, yy = np.meshgrid(x_ticks, y_ticks)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ✅ 构建 colormap
    base_cmap = cm.get_cmap('hot_r')
    new_cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))
    new_cmap.set_bad(color='lightgray')

    # ✅ 显示热力图（注意 .T）
    c = ax.imshow(
        np.ma.masked_invalid(cropped_trav_map.T),
        cmap=new_cmap,
        origin='lower',
        extent=[x_ticks.min(), x_ticks.max(), y_ticks.min(), y_ticks.max()],
        interpolation='none'
    )
    path1 = np.array(path_traditional)
    path_x1 = min_x + path1[:, 0] * resolution
    path_y1 = min_y + path1[:, 1] * resolution
    ax.plot(path_x1, path_y1, color='green', linewidth=3.0, label='Traditional D* Lite')

    path2 = np.array(path_weighted_only)
    path_x2 = min_x + path2[:, 0] * resolution
    path_y2 = min_y + path2[:, 1] * resolution
    ax.plot(path_x2, path_y2, color='purple', linewidth=4.0, label='Improved D* Lite')

    # ✅ 绘制路径
    path_x = min_x + path[:, 0] * resolution
    path_y = min_y + path[:, 1] * resolution
    ax.plot(path_x, path_y, color='blue', linewidth=3.0, label='Improved Cross-Domain D* Lite')
    



    # ✅ 起点、终点、当前位置（都按栅格坐标绘制）path_traditional,path_weighted_only,
    if start_grid is not None:
        sx, sy = min_x + start_grid[0] * resolution, min_y + start_grid[1] * resolution
        ax.plot(sx, sy, 'go', markersize=8, label='Start')

    if goal_grid is not None:
        gx, gy = min_x + goal_grid[0] * resolution, min_y + goal_grid[1] * resolution
        ax.plot(gx, gy, 'r*', markersize=12, label='Goal')

    if current_grid is not None:
        cx, cy = min_x + current_grid[0] * resolution, min_y + current_grid[1] * resolution
        ax.plot(cx, cy, 'mo', markersize=8, label='Current')

    if virtual_goal is not None:
        vx, vy = min_x + virtual_goal[0] * resolution, min_y + virtual_goal[1] * resolution
        ax.plot(vx, vy, 'm*', markersize=12, label='best_cross_point')        
    if virtual_path is not None and len(virtual_path) > 0:
        virtual_path = np.array(virtual_path)
        vp_x = min_x + virtual_path[:, 0] * resolution
        vp_y = min_y + virtual_path[:, 1] * resolution
        ax.plot(vp_x, vp_y, color='green', linestyle='--', linewidth=2.0, label='Virtual Path')
    # ✅ colorbar 和设置
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Traversability Cost (0=Good, 1=Bad)", fontsize=12)
    cbar.ax.tick_params(labelsize=12)            # colorbar 刻度放大

    ax.set_title("Local Traversability Heatmap around Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # —— 新增：放大坐标轴刻度字号 —— 
    ax.tick_params(axis='both', which='major', labelsize=12)

    # —— 新增：放大图例字体和标记尺寸 —— 
    leg = ax.legend(fontsize=14, markerscale=1.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()



def visualize_path_local_water_edge(
    water_edge_mask, path_indices, min_x, min_y, resolution=0.4,
    local_margin=5.0, save_path="path_local_water_edge.png",
    start_grid=None, goal_grid=None, current_grid=None,
    virtual_goal=None, virtual_path=None,
    candidate_cross_grids=None
):
    if path_indices is None or len(path_indices) == 0:
        print("路径为空，无法绘制水域边缘图")
        return

    path = np.array(path_indices)  # (i, j)
    all_indices = list(path_indices)

    if start_grid is not None:
        all_indices.append(start_grid)
    if goal_grid is not None:
        all_indices.append(goal_grid)
    if current_grid is not None:
        all_indices.append(current_grid)

    all_indices = np.array(all_indices)

    # ✅ 计算局部区域范围（以路径为中心，扩展 local_margin）
    margin_cells = 2 * int(local_margin / resolution)
    i_min = max(all_indices[:, 0].min() - margin_cells, 0)
    i_max = min(all_indices[:, 0].max() + margin_cells + 1, water_edge_mask.shape[0])
    j_min = max(all_indices[:, 1].min() - margin_cells, 0)
    j_max = min(all_indices[:, 1].max() + margin_cells + 1, water_edge_mask.shape[1])

    cropped_edge = water_edge_mask[i_min:i_max, j_min:j_max]

    # ✅ 计算坐标轴并修正 extent 对齐（关键步骤）
    x_ticks = np.linspace(min_x + i_min * resolution, min_x + (i_max - 1) * resolution, cropped_edge.shape[0])
    y_ticks = np.linspace(min_y + j_min * resolution, min_y + (j_max - 1) * resolution, cropped_edge.shape[1])

    extent = [
        min_x + i_min * resolution - resolution / 2,
        min_x + (i_max - 1) * resolution + resolution / 2,
        min_y + j_min * resolution - resolution / 2,
        min_y + (j_max - 1) * resolution + resolution / 2,
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    # ✅ 显示水域边缘图（红色）
    cmap = ListedColormap(['white', 'red'])  # 0为白色背景，1为水域边缘红色
    ax.imshow(
        cropped_edge.T,
        cmap=cmap,
        origin='lower',
        extent=extent,
        interpolation='none'
    )

    # ✅ 绘制路径
    path_x = min_x + path[:, 0] * resolution
    path_y = min_y + path[:, 1] * resolution
    ax.plot(path_x, path_y, color='blue', linewidth=3, label='Planned Path')

    # ✅ 绘制起点、终点、当前位置
    if start_grid is not None:
        sx, sy = min_x + start_grid[0] * resolution, min_y + start_grid[1] * resolution
        ax.plot(sx, sy, 'go', markersize=8, label='Start')

    if goal_grid is not None:
        gx, gy = min_x + goal_grid[0] * resolution, min_y + goal_grid[1] * resolution
        ax.plot(gx, gy, 'r*', markersize=12, label='Goal')

    if current_grid is not None:
        cx, cy = min_x + current_grid[0] * resolution, min_y + current_grid[1] * resolution
        ax.plot(cx, cy, 'mo', markersize=8, label='Current')

    # ✅ 绘制候选跨域点（绿色栅格块）
    if candidate_cross_grids is not None and len(candidate_cross_grids) > 0:
        for pt in candidate_cross_grids:
            i, j = pt
            if i_min <= i < i_max and j_min <= j < j_max:
                x = min_x + i * resolution
                y = min_y + j * resolution
                rect = patches.Rectangle(
                    (x - resolution / 2, y - resolution / 2),  # 左下角对齐
                    resolution,
                    resolution,
                    linewidth=1,
                    edgecolor='green',
                    facecolor='green',
                    alpha=0.6
                )
                ax.add_patch(rect)

    # ✅ 虚拟目标点及路径
    if virtual_goal is not None:
        vx, vy = min_x + virtual_goal[0] * resolution, min_y + virtual_goal[1] * resolution
        ax.plot(vx, vy, 'm*', markersize=12, label='best_cross_point')

    if virtual_path is not None and len(virtual_path) > 0:
        virtual_path = np.array(virtual_path)
        vp_x = min_x + virtual_path[:, 0] * resolution
        vp_y = min_y + virtual_path[:, 1] * resolution
        ax.plot(vp_x, vp_y, color='green', linestyle='--', linewidth=2.0, label='Virtual Path')

    ax.set_title("Local Water Edge Map around Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.set_title("Local Water Edge Map around Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # 增大坐标轴刻度字号
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 增大图例字体和图例标记尺寸
    leg = ax.legend(fontsize=14, markerscale=1.5)
    # 如果想让图例的边框更粗一点，也可以：
    # for line in leg.get_lines():
    #     line.set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()




def retrain_sgp_and_visualize_path(
    path_traditional,
    path_weighted_only,
    full_path_indices,
    resolution,
    min_x,
    min_y,
    start_grid=None,
    goal_grid=None,
    device="cpu"
):
    import numpy as np
    import torch
    import gpytorch
    from scipy.interpolate import griddata
    import plotly.graph_objects as go

    # —— 1. 构造 SGP 拟合数据 —— 
    pts = np.array([
        (min_x + i * resolution, min_y + j * resolution)
        for i, j in full_path_indices
    ])
    if pts.size == 0:
        print("⚠️ 路径为空，终止可视化")
        return
    path_x, path_y = pts[:,0], pts[:,1]

    # 提取路径附近点云
    local_range = 4
    mask = np.zeros(xyz_points.shape[0], dtype=bool)
    for px, py in zip(path_x, path_y):
        d2 = (xyz_points[:,0]-px)**2 + (xyz_points[:,1]-py)**2
        mask |= (d2 <= local_range**2)
    data = xyz_points[mask]
    if data.shape[0] < 10:
        print("⚠️ 点云数据不足，跳过 SGP 拟合")
        return

    # SGP 训练
    X = torch.tensor(data[:,:2], dtype=torch.float32)
    y = torch.tensor(data[:,2], dtype=torch.float32)
    M = min(1000, X.shape[0])
    inds = np.linspace(0, X.shape[0]-1, M, dtype=int)
    Z = X[inds]
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model = SGPModel(X, y, lik, Z).to(device)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(1):
        opt.zero_grad()
        loss = -mll(model(X), y)
        loss.backward(); opt.step()

    # 网格预测
    x0_min, x0_max = path_x.min(), path_x.max()
    y0_min, y0_max = path_y.min(), path_y.max()
    xs = np.linspace(x0_min-local_range, x0_max+local_range, 40)
    ys = np.linspace(y0_min-local_range, y0_max+local_range, 40)
    xx, yy = np.meshgrid(xs, ys)
    Xp = torch.tensor(np.stack([xx.ravel(), yy.ravel()],1),
                      dtype=torch.float32, device=device)
    model.eval(); lik.eval()
    with torch.no_grad():
        pr = lik(model(Xp))
    mean = pr.mean.cpu().numpy().reshape(xx.shape)

    # —— 2. 可视化 —— 
    fig = go.Figure()
    # 2.1 SGP mean surface
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=mean,
        colorscale='Viridis', opacity=0.8,
        name='SGP Mean'
    ))

    # 2.2 传统 D* Lite（绿色）
    if path_traditional:
        p1 = np.array(path_traditional)
        x1 = min_x + p1[:,0]*resolution
        y1 = min_y + p1[:,1]*resolution
        z1 = griddata((xx.ravel(),yy.ravel()), mean.ravel(),
                      (x1, y1), method='cubic') + 0.01
        fig.add_trace(go.Scatter3d(
            x=x1, y=y1, z=z1,
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=2),
            name='Traditional D* Lite'
        ))

    # 2.3 加权 D* Lite（紫色）
    if path_weighted_only:
        p2 = np.array(path_weighted_only)
        x2 = min_x + p2[:,0]*resolution
        y2 = min_y + p2[:,1]*resolution
        z2 = griddata((xx.ravel(),yy.ravel()), mean.ravel(),
                      (x2, y2), method='cubic') + 0.015
        fig.add_trace(go.Scatter3d(
            x=x2, y=y2, z=z2,
            mode='lines+markers',
            line=dict(color='purple', width=3),
            marker=dict(size=2),
            name='Weighted D* Lite'
        ))

    # 2.4 跨域 D* Lite（蓝色）
    z0 = griddata((xx.ravel(),yy.ravel()), mean.ravel(),
                  (path_x, path_y), method='cubic') + 0.005
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=z0,
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=2),
        name='Cross-Domain D* Lite'
    ))

    # 起点/终点
    if start_grid is not None:
        sx = min_x + start_grid[0]*resolution
        sy = min_y + start_grid[1]*resolution
        sz = griddata((xx.ravel(),yy.ravel()), mean.ravel(),
                      (sx, sy), method='cubic') + 0.02
        fig.add_trace(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode='markers', marker=dict(size=6, color='green'),
            name='Start'
        ))
    if goal_grid is not None:
        gx = min_x + goal_grid[0]*resolution
        gy = min_y + goal_grid[1]*resolution
        gz = griddata((xx.ravel(),yy.ravel()), mean.ravel(),
                      (gx, gy), method='cubic') + 0.02
        fig.add_trace(go.Scatter3d(
            x=[gx], y=[gy], z=[gz],
            mode='markers', marker=dict(size=6, color='red'),
            name='Goal'
        ))

    fig.update_layout(
        title="SGP 拟合与三条路径对比（不含平滑）",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=10,r=10,t=40,b=10)
    )
    fig.show()






def visualize_pointcloud_surface(full_path_indices, resolution, min_x, min_y):
    """
    在路径附近区域拟合地形曲面并三维可视化路径。
    参数：
        full_path_indices: 路径点的网格索引 (i,j)
        resolution: 地图分辨率（米/格）
        min_x, min_y: 原始地图的偏移坐标（左下角世界坐标）
    """
    # === 1. 将路径点转换为世界坐标 ===
    path_points = np.array([
        [min_x + i * resolution, min_y + j * resolution]
        for i, j in full_path_indices
    ])
    if path_points.size == 0:
        print("⚠️ 路径为空")
        return

    # === 2. 从原始 xyz_points 中提取路径附近的点 ===
    local_range = 5.0  # 单位：米
    mask = np.any([
        (xyz_points[:, 0] - px) ** 2 + (xyz_points[:, 1] - py) ** 2 <= local_range ** 2
        for px, py in path_points
    ], axis=0)
    subset_points = xyz_points[mask]
    if subset_points.shape[0] < 50:
        print("⚠️ 路径附近点云过少，无法拟合")
        return

    # === 3. 构建插值网格 ===
    x_min, x_max = subset_points[:, 0].min(), subset_points[:, 0].max()
    y_min, y_max = subset_points[:, 1].min(), subset_points[:, 1].max()
    x_lin = np.linspace(x_min, x_max, 100)
    y_lin = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_lin, y_lin)

    zz = griddata(
        (subset_points[:, 0], subset_points[:, 1]),
        subset_points[:, 2],
        (xx, yy),
        method='linear'
    )

    # 可选：对插值结果进行高斯滤波（让地形更平滑）
    zz = gaussian_filter(zz, sigma=1.0)

    # === 4. 对路径点插值 Z 值 ===
    path_z = griddata(
        (subset_points[:, 0], subset_points[:, 1]),
        subset_points[:, 2],
        (path_points[:, 0], path_points[:, 1]),
        method='linear'
    )

    # === 5. 可视化 ===
    fig = go.Figure()

    # 拟合曲面
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='Viridis',
        opacity=0.8,
        name="Fitted Surface"
    ))

    # 路径线
    fig.add_trace(go.Scatter3d(
        x=path_points[:, 0],
        y=path_points[:, 1],
        z=path_z,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=3, color='red'),
        name='Path'
    ))

    fig.update_layout(
        title="路径 + 点云拟合地形表面",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    fig.show()



import numpy as np
import math



def calculate_temp_target(
    temp_target_5,            # 二维列表（或 object array），每行至少 22 列
    c,                        # 当前行索引
    path_start,               # numpy array of shape (N,2)
    global_envrionment_map, 
    global_traversability_map,  # 环境地图：0 障碍，1=水域，2=陆地，255=未知
    global_mean_map,          # 高程地图 (同 shape)
    global_normal_map,        # 法向量地图，shape=(H,W,3)
    initial_start_grid,       # (i,j) 格点
    initial_goal_grid,        # (i,j) 格点        # 本函数只写入第 21 列
    time                      # 本函数写入第 22 列
):
    """
    将第 c 行的第 3~22 列写入 temp_target_5[c]：
      [ …, avg_pitch, avg_roll, avg_travel, avg_z_diff, ?, path_len, sum_pitch, sum_roll,
        up_pitch, up_roll, down_pitch, down_roll, roughness(=0), 
        z_diff_sum, z_diff_up, z_diff_down, path_array, num_transitions, time ]
    """
    # —— 1) 统计水/陆切换次数 —— 
    from math import isnan
    transitions = 0
    N = len(path_start)
    for k in range(N-1):
        e0 = global_envrionment_map[ tuple(path_start[k]) ]
        e1 = global_envrionment_map[ tuple(path_start[k+1]) ]
        # 只有 1⇄2 才算
        if (e0==1 and e1==2) or (e0==2 and e1==1):
            transitions += 1

    # —— 2) 计算路径长度（格点到格点欧氏距离之和） —— 
    path_len = 0.0
    for k in range(N-1):
        (i0,j0),(i1,j1) = path_start[k], path_start[k+1]
        path_len += math.hypot(i1-i0, j1-j0)

    # —— 3) 在“陆地”（env==2）上累加 z 差和偏航/俯仰 —— 
    z_diff_sum = 0.0
    pitch_sum = 0.0
    roll_sum  = 0.0
    travel_sum=0.0
    for k in range(N-1):
        i0,j0 = path_start[k]
        if global_envrionment_map[i0,j0] == 2:
            # 高差
            travel_sum=travel_sum+global_traversability_map[i0,j0]
            z0 = global_mean_map[i0,j0]
            z1 = global_mean_map[ tuple(path_start[k+1]) ]
            if not (isnan(z0) or isnan(z1)):
                z_diff_sum += abs(z1 - z0)
            # heading
            i1,j1 = path_start[k+1]
            heading = math.atan2(j1-j0, i1-i0)
            # 法向量
            nx, ny, nz = global_normal_map[i0,j0]
            if not (nx==0 and ny==0 and nz==0):
                # car_angle 返回 (yaw, pitch, roll)
                _, pitch, roll = car_angle((nx,ny,nz), heading)
                pitch_sum += abs(pitch)
                roll_sum  += abs(roll)

    # —— 4) 上岸/下水特征 —— 
    z_diff_up = z_diff_down = 0.0
    up_pitch = up_roll = down_pitch = down_roll = 0.0
    for k in range(1, N-2):
        i0,j0 = path_start[k]
        i1,j1 = path_start[k+1]
        # 上岸：陆地->水域
        if global_envrionment_map[i0,j0]==2 and global_envrionment_map[i1,j1]==1:
            z_diff_up = abs(global_mean_map[i1,j1] - global_mean_map[i0,j0])
            # 法向量取上岸点
            nx,ny,nz = global_normal_map[i0-1,j0-1]
            prev_i,prev_j = path_start[k-1]
            heading = math.atan2(j0-prev_j, i0-prev_i)
            if not (nx==0 and ny==0 and nz==0):
                _, up_pitch, up_roll = car_angle((nx,ny,nz), heading)
        # 下水：水域->陆地
        if global_envrionment_map[i0,j0]==1 and global_envrionment_map[i1,j1]==2:
            z_diff_down = abs(global_mean_map[i1,j1] - global_mean_map[i0,j0])
            # 法向量取下水点(下一格)
            nx,ny,nz = global_normal_map[i1,j1]
            heading = math.atan2(j0-j1, i0-i1)
            if not (nx==0 and ny==0 and nz==0):
                _, down_pitch, down_roll = car_angle((nx,ny,nz), heading)
    land_count = 0
    N = len(path_start)
    for k in range(N-1):
        i0,j0 = path_start[k]
        if global_envrionment_map[i0, j0] == 2:
            land_count += 1
    # —— 5) 写回 temp_target_5 —— 
    if land_count > 0:
        avg_pitch     = pitch_sum     / land_count
        avg_roll      = roll_sum      / land_count
        avg_travel    = travel_sum    / land_count
        avg_z_diff    = z_diff_sum    / land_count
    else:
        avg_pitch = avg_roll = avg_travel = avg_z_diff = 0.0
    
    row = temp_target_5[c]
    # Python 列号从0开始，MATLAB 的第3列对应 row[2]
    row[2]  = avg_pitch
    row[3]  = avg_roll
    row[4]  = avg_travel
    row[5]  = avg_z_diff
    row[7]  = path_len
    row[8]  = travel_sum
    row[9]  = pitch_sum
    row[10] = roll_sum
    row[11] = up_pitch
    row[12] = up_roll
    row[13] = down_pitch
    row[14] = down_roll
    row[15] = 0.0            # 原 roughness_sum，简化为 0
    row[16] = z_diff_sum
    row[17] = z_diff_up
    row[18] = z_diff_down
    row[19] = path_start.copy()
    row[20] = transitions
    row[21] = time

    return temp_target_5

