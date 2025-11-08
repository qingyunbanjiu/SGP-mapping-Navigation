import numpy as np
import plotly.graph_objects as go
import torch
import gpytorch
from time import time
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from flatness import compute_flatness
from compute_step_height import compute_step_height
from generate_semantic_map import generate_semantic_map
from generate_semantic_map import generate_semantic_map, visualize_semantic_map
from compute_traversability import compute_traversability_map, visualize_traversability_map
from scipy.interpolate import griddata

# 第一步：加载本地点云，筛选局部数据
file_path = "20250330.xyz"
xyz_points = np.loadtxt(file_path)

# x0, y0 = 250, 205
x0, y0 = 250, 205
local_range = 6
distance_sq = (xyz_points[:, 0] - x0) ** 2 + (xyz_points[:, 1] - y0) ** 2
mask = distance_sq <= local_range ** 2
local_points = xyz_points[mask]


vis_range = 12
vis_mask = (xyz_points[:, 0] - x0)**2 + (xyz_points[:, 1] - y0)**2 <= vis_range**2
vis_points = xyz_points[vis_mask]
# 如果点过多，进行下采样

# 第二步：对局部点云做稀疏高斯过程回归 (SGP)
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

# 数据准备
X_train_np = local_points[:, :2]
y_train_np = local_points[:, 2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)

# 诱导点选择
inducing_num = 300
idx = np.linspace(0, X_train.shape[0] - 1, inducing_num, dtype=int)
inducing_pts = X_train[idx, :]

# 模型训练
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = SGPModel(X_train, y_train, likelihood, inducing_pts).to(device)
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# 记录训练开始时间
start_time = time()

for i in range(1):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# 记录训练结束时间
end_time = time()

# 计算训练时间
training_time = end_time - start_time
print(f"模型训练时间: {training_time} 秒")

# 第三步：预测与可视化
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
slope_angle_deg = np.degrees(np.arctan(slope))

# 第四步：构建栅格
def build_grid(points, large_size=0.8, small_size=0.4):
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])

    # 大尺寸栅格
    large_grid_x = np.arange(min_x, max_x + large_size, large_size)
    large_grid_y = np.arange(min_y, max_y + large_size, large_size)
    large_grid = np.array(np.meshgrid(large_grid_x, large_grid_y)).T.reshape(-1, 2)

    # 小尺寸栅格
    small_grid_x = np.arange(min_x, max_x + small_size, small_size)
    small_grid_y = np.arange(min_y, max_y + small_size, small_size)
    small_grid = np.array(np.meshgrid(small_grid_x, small_grid_y)).T.reshape(-1, 2)

    return large_grid, small_grid

def get_grid_cell(point, grid, cell_size):
    x_index = int((point[0] - np.min(grid[:, 0])) // cell_size)
    y_index = int((point[1] - np.min(grid[:, 1])) // cell_size)
    return x_index, y_index

# 构建栅格
large_grid, small_grid = build_grid(xyz_points)

# 示例：判断某个点在栅格中的位置
test_point = [10, 10]
large_cell_index = get_grid_cell(test_point, large_grid, 0.8)
small_cell_index = get_grid_cell(test_point, small_grid, 0.4)
print(f"大尺寸栅格中的位置: {large_cell_index}")
print(f"小尺寸栅格中的位置: {small_cell_index}")

# 评估拟合效果
train_preds = likelihood(model(X_train)).mean.detach().cpu().numpy()
train_y_np = y_train.cpu().numpy()

mse = mean_squared_error(train_y_np, train_preds)
rmse = np.sqrt(mse)
r2 = r2_score(train_y_np, train_preds)

print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"决定系数 (R^2): {r2}")

# 统一0.4m格子的x/y
resolution = 0.4
grid_x = np.arange(xx.min(), xx.max() + 1e-6, resolution)
grid_y = np.arange(yy.min(), yy.max() + 1e-6, resolution)
xx_grid, yy_grid = np.meshgrid(grid_x, grid_y)

# 插值 mean
mean_grid = griddata(
    points=(xx.flatten(), yy.flatten()),
    values=mean.flatten(),
    xi=(xx_grid, yy_grid),
    method='nearest'
)

# 插值 slope
slope_grid = griddata(
    points=(xx.flatten(), yy.flatten()),
    values=slope.flatten(),
    xi=(xx_grid, yy_grid),
    method='nearest'
)

# var_grid 你已经插好了，也可以用相同方式统一一下
var_grid = griddata(
    points=(xx.flatten(), yy.flatten()),
    values=var.flatten(),
    xi=(xx_grid, yy_grid),
    method='nearest'
)

plt.figure(figsize=(10, 8))
plt.imshow(
    mean_grid.T,  # 注意 transpose，让X、Y方向对齐
    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
    origin='lower',
    cmap='viridis'
)
plt.colorbar(label="Mean Elevation (m)")
plt.title("SGP Mean Elevation (0.4m Resolution)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.savefig("mean_grid_map.png", dpi=600)
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(
    slope_grid.T,
    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
    origin='lower',
    cmap='Blues'
)
plt.colorbar(label="Slope Magnitude (|∇mean|)")
plt.title("SGP Slope Map (0.4m Resolution)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.savefig("slope_grid_map.png", dpi=600)
plt.show()

plt.figure(figsize=(10, 8))
plt.imshow(
    var_grid.T,
    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
    origin='lower',
    cmap='Reds'
)
plt.colorbar(label="Variance")
plt.title("SGP Variance Map (0.4m Resolution)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.tight_layout()
plt.savefig("variance_grid_map.png", dpi=600)
plt.show()

def interpolate_to_grid(data, xx_src, yy_src, xx_target, yy_target, method='linear'):
    """
    将连续值数据插值对齐到目标栅格（如 0.4m 分辨率）
    """
    points = np.column_stack((xx_src.flatten(), yy_src.flatten()))
    values = data.flatten()
    target_points = np.column_stack((xx_target.flatten(), yy_target.flatten()))
    interpolated = griddata(points, values, target_points, method=method)
    return interpolated.reshape(xx_target.shape)




def visualize_step_height_map(step_height_map, x0, y0, resolution=0.4, save_path="step_height_map_highres.png"):
    """
    带真实坐标轴的 Step Height Map 可视化
    step_height_map: 2D array
    x0, y0: 左下角起点坐标
    resolution: 每个栅格尺寸（米）
    """
    h, w = step_height_map.shape

    x_min = x0
    x_max = x0 + w * resolution
    y_min = y0
    y_max = y0 + h * resolution

    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.imshow(
        step_height_map.T, 
        cmap='plasma', 
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        interpolation='none'
    )
    fig.colorbar(c, ax=ax, label='Step Height (m)')
    ax.set_title("Step Height Map")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()





# 假设你已经有了一个高程网格（例如从 SGP 的 mean.reshape(xx.shape) 得到）
elevation_grid = mean  # 或者从栅格数据中提取的二维数组

# 调用函数计算 flatness
flatness_map = compute_flatness(
    elevation_grid=mean_grid,  # 就是插值后的高程
    resolution=0.4,            # 你的统一分辨率
    physical_window=1.2        # 物理范围，比如说以1.2米范围计算局部平坦度
)

fig, ax = plt.subplots(figsize=(10, 8))
c = ax.imshow(
    flatness_map.T, 
    extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], 
    cmap='plasma', 
    origin='lower'
)
fig.colorbar(c, ax=ax, label="Flatness (0~1)")
ax.set_title("Flatness Map (Physical Window = 1.2m)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
plt.tight_layout()
plt.savefig("flatness_map_aligned.png", dpi=600)
plt.show()



step_height_map = compute_step_height(mean_grid, window_size=3)
fig, ax = plt.subplots(figsize=(10, 8))
c = ax.imshow(
    step_height_map.T, 
    extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]], 
    cmap='plasma', 
    origin='lower'
)
fig.colorbar(c, ax=ax, label="Step Height (m)")
ax.set_title("Step Height Map (Physical Window = 1.2m)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
plt.tight_layout()
plt.savefig("step_height_map_aligned.png", dpi=600)
plt.show()









# 生成语义地图
semantic_map = generate_semantic_map(
    mean=mean_grid,
    slope=slope_grid,
    var=var_grid,
    step_map=step_height_map,
    flatness_map=flatness_map,
    resolution=0.4,  # 固定0.4
    slope_thresh=0.10,  # 水域判断的最大坡度
    elev_thresh=0.55,   # 水域判断的最大高程
    crit_slope=0.5238,  # 障碍物判断的坡度阈值（大概是30°）
    crit_step=0.35,     # Step Height 阈值
    crit_flat=0.5238    # Flatness阈值
)


visualize_semantic_map(semantic_map, xx_grid, yy_grid, save_path="semantic_map_aligned.png")

# 通行性地图计算
# ✅ 最终调用通行性地图生成函数（含强水域阻断）
# 计算通行性地图
trav_map = compute_traversability_map(
    step_map=step_height_map,
    flatness_map=flatness_map,
    slope_map=slope_grid,
    var_map=var_grid,
    semantic_map=semantic_map,
    w_step=0.3,
    w_flat=0.5,
    w_slope=0.2,
    crit_step=0.3,
    crit_flat=0.5236,
    crit_slope=0.5236,
    var_thresh=1.25
)

# 可视化通行性地图
visualize_traversability_map(trav_map, xx_grid, yy_grid, save_path="traversability_heatmap_final.png")






# 第四步：Plotly 展示
# 1️⃣ 原始点云 + SGP Mean
fig_mean = go.Figure()

# 替换原始点云为中等范围点云
fig_mean.add_trace(go.Scatter3d(
    x=vis_points[:, 0],
    y=vis_points[:, 1],
    z=vis_points[:, 2],
    mode='markers',
    marker=dict(size=0.8, color='blue'),
    name='Surrounding Point Cloud'
))


# SGP 均值面
fig_mean.add_trace(go.Surface(
    x=xx, y=yy, z=mean,
    colorscale='Viridis', opacity=0.7,
    name='SGP Mean', showscale=True
))

fig_mean.update_layout(
    title="SGP Mean + Local Point Cloud",
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
)
fig_mean.show()

# 2️⃣ Uncertainty (方差) 单独图层
fig_var = go.Figure()
fig_var.add_trace(go.Surface(
    x=xx, y=yy, z=var,  # 为了更容易分离观察
    surfacecolor=var,
    colorscale='Reds', opacity=0.9,
    name='Uncertainty', showscale=True
))
fig_var.update_layout(
    title="Uncertainty (Variance)",
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
)
fig_var.show()

# 3️⃣ Slope (梯度模) 单独图层
fig_slope = go.Figure()
fig_slope.add_trace(go.Surface(
    x=xx, y=yy, z=slope,  # 也偏移一下避免重叠
    surfacecolor=slope,
    colorscale='Blues', opacity=0.9,
    name='Slope', showscale=True
))
fig_slope.update_layout(
    title="Slope Magnitude (|∇mean|)",
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
)
fig_slope.show()    
fig_mean.write_image("sgp_mean_surface.svg", width=1600, height=1200)
fig_var.write_image("sgp_variance_surface.svg", width=1600, height=1200)
fig_slope.write_image("sgp_slope_surface.svg", width=1600, height=1200)

plt.figure(figsize=(8, 6))
plt.imshow(slope, extent=[x0 - local_range, x0 + local_range, y0 - local_range, y0 + local_range],
           origin='lower', cmap='Blues')
plt.colorbar(label='Slope Magnitude (|∇mean|)')
plt.title('2D Slope Map based on SGP Mean')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig("slope_2d_heatmap.svg", dpi=600)
plt.show()

# 定义栅格参数（确保与路径规划一致）
resolution = 0.4
grid_x = np.arange(xx.min(), xx.max() + 1e-6, resolution)
grid_y = np.arange(yy.min(), yy.max() + 1e-6, resolution)

# 初始化空的栅格（注意 shape 是 [len(grid_x)-1, len(grid_y)-1]）
variance_grid = np.full((len(grid_x)-1, len(grid_y)-1), np.nan)

# 将预测的 var 填入到最近的栅格
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x, y = xx[i, j], yy[i, j]
        v = var[i, j]
        # 找到对应的格子索引
        ix = int((x - grid_x[0]) // resolution)
        iy = int((y - grid_y[0]) // resolution)
        variance_grid[ix, iy] = v
        # if 0 <= ix < variance_grid.shape[0] and 0 <= iy < variance_grid.shape[1]:
        #     if np.isnan(variance_grid[ix, iy]):
        #         variance_grid[ix, iy] = v
        #     else:
        #         variance_grid[ix, iy] = (variance_grid[ix, iy] + v) / 2.0  # 简单平均

# 可视化
fig, ax = plt.subplots(figsize=(10, 8))
c = ax.imshow(
    variance_grid.T, 
    extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
    origin='lower', cmap='Reds'
)
fig.colorbar(c, ax=ax, label="Variance")
ax.set_title("2D Variance Map (Resolution = 0.4m)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.tight_layout()
plt.savefig("sgp_variance_grid_aligned.png", dpi=600)
plt.show()

