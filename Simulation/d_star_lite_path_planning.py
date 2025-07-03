import numpy as np
import sys
from generate_semantic_map import generate_semantic_map, visualize_semantic_map
from dstar_lite import DStarLite
from point_utils import Point
from utils import world_to_grid, grid_to_world
import time
import pandas as pd
# 从 informed_RRT.py 导入所需函数和类
from functions import (
    SGPModel,
    run_sgp_mapping,
    update_global_maps_from_sgp,
    is_point_valid,
    visualize_path_local_traversability1,
    is_path_fully_explored,
    extract_water_edge_from_semantic_map,
    visualize_path_local_water_edge,
    detect_structural_blockage_from_grid,
    extract_candidate_cross_points_in_radius,
    retrain_sgp_and_visualize_path,
    visualize_pointcloud_surface,
    select_virtual_goal_refined,
    try_virtual_goal_path,
    select_virtual_goal_adaptive,
    select_best_cross_domain_point,
    try_virtual_goal_path,
    car_angle,
    get_euler_from_normal,
    compute_cross_domain_risk,
    calculate_temp_target
)


rmse_list = []
r2_list   = []

def navigate_segment_with_cross_event(
      start_grid, goal_grid,
      use_weighted: bool,
      use_cross_domain: bool,
      use_huanjing: bool
):
    """
    从 start_grid 导航到 goal_grid；若途中触发跨域阻碍，则返回真正的跨域点。
    返回 (reached_goal: bool, path_list: List[格点], cross_pt: (i,j) or None)
    """
    planner = DStarLite(start_grid, goal_grid,
                        global_traversability_map,
                        global_envrionment_map,
                        global_normal_map,
                        use_weighted=use_weighted)
    planner.compute_shortest_path()

    current = start_grid
    path_list = [current]
    block_count = 0

    while current != goal_grid:
        # 1. 世界坐标
        x = min_x + current[0] * resolution
        y = min_y + current[1] * resolution

        # 2. 动态建图 + 更新全局地图
        result = run_sgp_mapping(x, y)
        xx, yy, mean_g, slope_g, var_g, flat_g, step_h, sem_map, trav_map, grad_m,rmse,r2 = result
        rmse_list.append(rmse)
        r2_list.append(r2)       
        update_global_maps_from_sgp(
            xx, yy, mean_g, slope_g, var_g,
            sem_map, trav_map, grad_m,
            global_mean_map, global_slope_map, global_roughness_map,
            global_envrionment_map, global_traversability_map,
            global_frozen_map, global_normal_map
        )

        # 3. 更新 D* Lite 起点 + 邻居 + 重算
        # 3. 更新 D* Lite 起点 + 邻居 + 重算
        old_start = planner.s_start
        planner.k_m += planner.heuristic(old_start, current)   # ← 补上这一行
        planner.s_start = current
        planner.update_vertex(current)
        for nbr in planner.get_neighbors(current):
            planner.update_vertex(nbr)
        planner.compute_shortest_path()

        # 4. 选最优 successor
        # 4. 选最优 successor（原来只看 g_val + c）
        env_now    = global_envrionment_map[current[0], current[1]]
        
        successors = planner.get_neighbors(current)
        next_grid, min_cost = None, float('inf')
        
        w_h = 0.02   # 启发式权重

        # 先在 g_val 有效的节点里找最优
        best_found = False
        for s in successors:
            if use_huanjing and global_envrionment_map[s] != env_now:
                continue
            g_val = planner.g.get(s, float('inf'))
            if g_val < float('inf'):
                c = planner.cost(current, s)
                h = planner.heuristic(s, planner.s_goal)
                score = g_val + c
                print(s, "g_val=", g_val, "score(using g)=", score)
                if score < min_cost:
                    min_cost, next_grid = score, s
                    best_found = True

        # 如果没有任何 g_val 有效的后继，再退到纯启发式选择
        if not best_found:
            
            min_cost = float('inf')
            for s in successors:
                if use_huanjing and global_envrionment_map[s] != env_now:
                    continue
                c = planner.cost(current, s)
                h = planner.heuristic(s, planner.s_goal)
                score = c + w_h * h
                print(s, "g_val=inf, score(using heuristic)=", score)
                if score < min_cost:
                    min_cost, next_grid = score, s



        print("next_grid=", next_grid)
        # 5. 路径失效检测
        i, j = next_grid
        if  global_envrionment_map[i, j] == 0:
            planner.update_vertex(current)
            planner.compute_shortest_path()
            continue

        # 6. 结构性阻碍检测
        if use_cross_domain:
            
            is_blocked, _, _, _, _ = detect_structural_blockage_from_grid(
                current_grid=current,
                next_grid=next_grid,
                edge_points=np.argwhere(
                    extract_water_edge_from_semantic_map(
                        global_envrionment_map,
                        global_traversability_map
                    )
                ),
                resolution=resolution,
                min_x=min_x, min_y=min_y
            )
            block_count = block_count + 1 if is_blocked else 0
            if block_count >= 2:
                # 初步选出候选跨域点
                print("大范围结构性阻碍")
                candidate_cross = select_best_cross_domain_point(
                    water_edge_mask=extract_water_edge_from_semantic_map(
                        global_envrionment_map,
                        global_traversability_map
                    ),
                    initial_goal_grid=initial_goal_grid,
                    initial_start_grid=initial_start_grid,
                    current_grid=current,
                    candidate_cross_grids=extract_candidate_cross_points_in_radius(
                        current_grid=current,
                        edge_points=np.argwhere(
                            extract_water_edge_from_semantic_map(
                                global_envrionment_map,
                                global_traversability_map
                            )
                        ),
                        global_envrionment_map=global_envrionment_map,
                        resolution=resolution, radius=4.0
                    ),
                    global_traversability_map=global_traversability_map,
                    global_envrionment_map=global_envrionment_map,
                    global_normal_map=global_normal_map,
                    global_mean_map=global_mean_map,
                    weight_dist=1.0, weight_rough=2.0,
                    resolution=resolution
                )
                print("candidate_cross")
                print(candidate_cross)
                # 在候选跨域点附近再选最优跨域点
                i0, j0 = candidate_cross
                current_env = global_envrionment_map[i0, j0]
                rows, cols = global_envrionment_map.shape
                best_cross, best_cost = None, float('inf')
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i0 + di, j0 + dj
                        if not (0 <= ni < rows and 0 <= nj < cols):
                            continue
                        neighbor_env = global_envrionment_map[ni, nj]
                        if current_env not in (1, 2) or neighbor_env not in (1, 2):
                            continue
                        if neighbor_env == current_env:
                            continue
                        normal_vec = global_normal_map[i0, j0]
                        direction_vec = np.array([ni - i0, nj - j0])
                        angle = np.arctan2(direction_vec[1], direction_vec[0])
                        _, pitch_raw, roll_raw = get_euler_from_normal(normal_vec, angle)
                        pitch, roll = abs(pitch_raw), abs(roll_raw)
                        elevation_diff = abs(global_mean_map[i0, j0] - global_mean_map[ni, nj])
                        cross_risk = compute_cross_domain_risk(pitch, roll, elevation_diff)
                        
                        if cross_risk < best_cost:
                            best_cost, best_cross = cross_risk, (ni, nj)
                cross_pt = best_cross
                # 模拟行进到真正的跨域点
                i0, j0 = current
                current_env = global_envrionment_map[i0, j0]
                if current_env==2:
                    reached, cross_path, _ = navigate_segment_with_cross_event(
                        current, candidate_cross,
                        use_weighted=True,
                        use_cross_domain=False,
                        use_huanjing=True
                    )
                    if reached:
                        if cross_path[0] != current:
                            cross_path = cross_path[::-1]
                        for grid in cross_path[1:]:
                            path_list.append(grid)
                    path_list.append(cross_pt)
                    return False, path_list, cross_pt
                elif current_env==1:
                    reached, cross_path, _ = navigate_segment_with_cross_event(
                        current, cross_pt,
                        use_weighted=True,
                        use_cross_domain=False,
                        use_huanjing=True
                    )
                    if reached:
                        if cross_path[0] != current:
                            cross_path = cross_path[::-1]
                        for grid in cross_path[1:]:
                            path_list.append(grid)
                    path_list.append(candidate_cross)
                    return False, path_list, candidate_cross
                else:
                    break
                # is_valid, cross_path = try_virtual_goal_path(


        # 7. 正常前进一步
        current = next_grid
        path_list.append(current)

    return True, path_list, None

# 第一步：加载本地点云
file_path = "20250330.xyz"
xyz_points = np.loadtxt(file_path)
min_x = np.min(xyz_points[:, 0])
min_y = np.min(xyz_points[:, 1])
max_x = np.max(xyz_points[:, 0])
max_y = np.max(xyz_points[:, 1])
resolution = 0.4
# 构建栅格函数（保持不变）
large_size = 0.8
small_size = 0.4
local_range = 5
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

# 6. 初始化全局地图（NaN 或零填充，待写入）
global_mean_map = np.full((num_x, num_y), np.nan)
global_slope_map = np.full((num_x, num_y), np.nan)        # 坡度地图
global_normal_map = np.zeros((num_x, num_y, 3))           # 法向量地图 (x, y, z)
global_roughness_map = np.full((num_x, num_y), np.nan)    # 粗糙度地图
global_traversability_map = np.full((num_x, num_y), np.nan)  # 通行性代价地图
global_envrionment_map = np.full((num_x, num_y), np.nan)   # 环境地图
global_frozen_map = np.zeros_like(global_envrionment_map, dtype=bool)
global_water_edge_map = np.zeros((num_x, num_y), dtype=bool)

# 定义起点和终点
# start_point = Point(250, 205)
# goal_point = Point(250, 195)

# start_point = Point(250, 120)
# goal_point = Point(260, 120)
start_point = Point(205, 160)
goal_point = Point(220, 140)

# start_point = Point(200, 150)
# goal_point = Point(240, 150)

# start_point = Point(170, 200)
# goal_point = Point(190, 185)

if not is_point_valid(start_point,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map) or not is_point_valid(goal_point,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map):
    raise ValueError("起点或终点无效，规划终止。")
    sys.exit(1)  # ⚠️ 终止整个程序运行
print("初始规划的路径1：")
# 6. 初始化全局地图（NaN 或零填充，待写入）

# global_mean_map = np.full((num_x, num_y), np.nan)
# global_slope_map = np.full((num_x, num_y), np.nan)        # 坡度地图
# global_normal_map = np.zeros((num_x, num_y, 3))           # 法向量地图 (x, y, z)
# global_roughness_map = np.full((num_x, num_y), np.nan)    # 粗糙度地图
# global_traversability_map = np.full((num_x, num_y), np.nan)  # 通行性代价地图
# global_envrionment_map = np.full((num_x, num_y), np.nan)   # 环境地图
# global_frozen_map = np.zeros_like(global_envrionment_map, dtype=bool)
# global_water_edge_map = np.zeros((num_x, num_y), dtype=bool)

# 起点附近局部建图（替代初始全局建图）
initial_xyz = xyz_points[np.linalg.norm(xyz_points[:, :2] - np.array([start_point.x, start_point.y]), axis=1) <= 5]
result =run_sgp_mapping(start_point.x, start_point.y)
xx_grid, yy_grid, mean_grid, slope_grid, var_grid, flatness_map, step_height_map, semantic_map, trav_map, grad_mean,rmse,r2 = result
update_global_maps_from_sgp(xx_grid, yy_grid, mean_grid, slope_grid, var_grid, semantic_map, trav_map, grad_mean,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map)
# 创建 D* Lite 规划器实例
initial_start_grid=world_to_grid(start_point.x, start_point.y)
initial_goal_grid=world_to_grid(goal_point.x, goal_point.y)

temp_target_5 = [[None]*22 for _ in range(3)]

start_time = time.time()
# 1. 纯传统 D* Lite（不加权、不做跨域）  
reached, path_traditional, _ = navigate_segment_with_cross_event(
    initial_start_grid, initial_goal_grid,
    use_weighted=False,
    use_cross_domain=False,
    use_huanjing=False
)
end_time = time.time()
elapsed_time = end_time - start_time

temp_target_5 =calculate_temp_target(
        temp_target_5, 0, path_traditional,
        global_envrionment_map,global_traversability_map, global_mean_map, global_normal_map,
        initial_start_grid, initial_goal_grid,
         elapsed_time)

print("🎯 path_traditional成功到达终点！任务完成。")
        # ✅ 输出最终路径的栅格坐标
print("🧩 path_traditional最终路径为（栅格坐标系）：")
for grid_point in path_traditional:
    print(grid_point)  

mean_rmse = np.mean(rmse_list)
std_rmse  = np.std(rmse_list)
mean_r2   = np.mean(r2_list)
std_r2    = np.std(r2_list)

print("========== SGP 建图总体性能 ==========")
print(f"平均 RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"平均 R²  = {mean_r2:.4f} ± {std_r2:.4f}")
# 2. 改进 D* Lite（加权代价，但不做跨域） 
#  
# 6. 初始化全局地图（NaN 或零填充，待写入）
global_mean_map = np.full((num_x, num_y), np.nan)
global_slope_map = np.full((num_x, num_y), np.nan)        # 坡度地图
global_normal_map = np.zeros((num_x, num_y, 3))           # 法向量地图 (x, y, z)
global_roughness_map = np.full((num_x, num_y), np.nan)    # 粗糙度地图
global_traversability_map = np.full((num_x, num_y), np.nan)  # 通行性代价地图
global_envrionment_map = np.full((num_x, num_y), np.nan)   # 环境地图
global_frozen_map = np.zeros_like(global_envrionment_map, dtype=bool)
global_water_edge_map = np.zeros((num_x, num_y), dtype=bool)

initial_xyz = xyz_points[np.linalg.norm(xyz_points[:, :2] - np.array([start_point.x, start_point.y]), axis=1) <= 5]
result =run_sgp_mapping(start_point.x, start_point.y)
xx_grid, yy_grid, mean_grid, slope_grid, var_grid, flatness_map, step_height_map, semantic_map, trav_map, grad_mean,rmse,r2 = result
update_global_maps_from_sgp(xx_grid, yy_grid, mean_grid, slope_grid, var_grid, semantic_map, trav_map, grad_mean,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map)

start_time = time.time()
reached, path_weighted_only, _ = navigate_segment_with_cross_event(
    initial_start_grid, initial_goal_grid,
    use_weighted=True,
    use_cross_domain=False,
    use_huanjing=False
)

end_time = time.time()
elapsed_time1 = end_time - start_time

temp_target_5 =calculate_temp_target(
        temp_target_5, 1, path_weighted_only,
        global_envrionment_map,global_traversability_map, global_mean_map, global_normal_map,
        initial_start_grid, initial_goal_grid,
         elapsed_time1)


print("🎯 path_weighted_only成功到达终点！任务完成。")
        # ✅ 输出最终路径的栅格坐标
print("🧩 path_weighted_only最终路径为（栅格坐标系）：")
for grid_point in path_weighted_only:
    print(grid_point)
# 6. 初始化全局地图（NaN 或零填充，待写入）

global_mean_map = np.full((num_x, num_y), np.nan)
global_slope_map = np.full((num_x, num_y), np.nan)        # 坡度地图
global_normal_map = np.zeros((num_x, num_y, 3))           # 法向量地图 (x, y, z)
global_roughness_map = np.full((num_x, num_y), np.nan)    # 粗糙度地图
global_traversability_map = np.full((num_x, num_y), np.nan)  # 通行性代价地图
global_envrionment_map = np.full((num_x, num_y), np.nan)   # 环境地图
global_frozen_map = np.zeros_like(global_envrionment_map, dtype=bool)
global_water_edge_map = np.zeros((num_x, num_y), dtype=bool)

# 起点附近局部建图（替代初始全局建图）
initial_xyz = xyz_points[np.linalg.norm(xyz_points[:, :2] - np.array([start_point.x, start_point.y]), axis=1) <= 5]
result =run_sgp_mapping(start_point.x, start_point.y)
xx_grid, yy_grid, mean_grid, slope_grid, var_grid, flatness_map, step_height_map, semantic_map, trav_map, grad_mea,rmse,r2 = result
update_global_maps_from_sgp(xx_grid, yy_grid, mean_grid, slope_grid, var_grid, semantic_map, trav_map, grad_mean,global_mean_map,global_slope_map,global_roughness_map,global_envrionment_map,global_traversability_map,global_frozen_map,global_normal_map)

# —— 主流程：支持多次跨域 —— 
full_path_indices = []
current = initial_start_grid
start_time = time.time()

while current != initial_goal_grid:
    reached, seg, cross_pt = navigate_segment_with_cross_event(current, initial_goal_grid,use_weighted=True,use_cross_domain=True,use_huanjing=True)
    # 合并本段（去掉 seg[0] 重复）
    full_path_indices += seg[0:]

    if reached:
        print("✅ 成功到达终点！")
        break

    # 跨域动作（下水或上岸）
    print(f"🚧 在格点 {cross_pt} 触发跨域，执行动作…")
    current = cross_pt
    print("上一段结束，new current =", cross_pt)
    print("下一段开始，start current =", current)
    # water_control.enter_water()  或 exit_water()


end_time = time.time()
elapsed_time2 = end_time - start_time

temp_target_5 =calculate_temp_target(
        temp_target_5, 2, full_path_indices,
        global_envrionment_map,global_traversability_map, global_mean_map, global_normal_map,
        initial_start_grid, initial_goal_grid,
         elapsed_time2)




# 假设 temp_target_5 是一个 list of list，形状为 (M, 22)
# 请根据你的实际列含义替换下面的列名
column_names = [
    "cx","cy",
    "start_i","start_j",
    "goal_i","goal_j",
    "unused7","path_len","unused9",
    "sum_pitch","sum_roll",
    "up_pitch","up_roll","down_pitch","down_roll",
    "roughness_sum","z_diff_sum","z_diff_up","z_diff_down",
    "raw_path","num_transitions","time"
]
df = pd.DataFrame(temp_target_5, columns=column_names)
df.to_csv("path_metrics.csv", index=False, encoding="utf-8-sig")





print("📍 完整路径格点序列：", full_path_indices)


for grid_point in path_traditional:
    if global_traversability_map[grid_point[0], grid_point[1]]>=1:
        global_traversability_map[grid_point[0], grid_point[1]]= 0.8

for grid_point in path_weighted_only:
    if global_traversability_map[grid_point[0], grid_point[1]]>=1:
        global_traversability_map[grid_point[0], grid_point[1]]= 0.8

for grid_point in full_path_indices:
    if global_traversability_map[grid_point[0], grid_point[1]]>=1:
        global_traversability_map[grid_point[0], grid_point[1]]= 0.8

water_edge_mask = extract_water_edge_from_semantic_map(
                        global_envrionment_map,
                        global_traversability_map
                    )
print("🎯 成功到达终点！任务完成。")
        # ✅ 输出最终路径的栅格坐标
print("🧩 最终路径为（栅格坐标系）：")
for grid_point in full_path_indices:
    print(grid_point)  
min_x = np.min(xyz_points[:, 0])
min_y = np.min(xyz_points[:, 1])
max_x = np.max(xyz_points[:, 0])
max_y = np.max(xyz_points[:, 1])



# print("water_edge_mask shape:", water_edge_mask.shape)
# print("water_edge_mask 有效像素数量：", np.sum(water_edge_mask))
start_grid = world_to_grid(start_point.x, start_point.y)
goal_grid = world_to_grid(goal_point.x, goal_point.y)
current_grid = full_path_indices[-1]

# visualize_path_local_water_edge(water_edge_mask, full_path_indices, min_x, min_y, resolution=0.4,start_grid=start_grid,goal_grid=goal_grid,current_grid=current_grid,virtual_goal=cross_pt 
#  )
# 起点终点是世界坐标，需要转栅格 ,virtual_path=path ,  virtual_goal=candidate_cross_grids


visualize_path_local_traversability1(
    trav_map=global_traversability_map,
    path_indices=full_path_indices,
    path_traditional=path_traditional,
    path_weighted_only=path_weighted_only,
    min_x=min_x,
    min_y=min_y,
    resolution=resolution,
    start_grid=start_grid,
    goal_grid=goal_grid,
    current_grid=current_grid,
    virtual_goal=cross_pt 

)

retrain_sgp_and_visualize_path(
    path_traditional=path_traditional,
    path_weighted_only=path_weighted_only,
    full_path_indices=full_path_indices,
    resolution=0.4,
    min_x=min_x,
    min_y=min_y,
    start_grid=initial_start_grid,
    goal_grid=initial_goal_grid
)


visualize_pointcloud_surface(full_path_indices=full_path_indices, resolution=0.4, min_x=min_x, min_y=min_y)