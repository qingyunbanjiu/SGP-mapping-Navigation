import numpy as np
import torch
import gpytorch
from time import time
from sklearn.metrics import mean_squared_error, r2_score

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

# 构建栅格函数
def build_grid(points, large_size=0.8, small_size=0.4):
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])
    large_grid_x = np.arange(min_x, max_x + large_size, large_size)
    large_grid_y = np.arange(min_y, max_y + large_size, large_size)
    large_grid = np.array(np.meshgrid(large_grid_x, large_grid_y)).T.reshape(-1, 2)
    small_grid_x = np.arange(min_x, max_x + small_size, small_size)
    small_grid_y = np.arange(min_y, max_y + small_size, small_size)
    small_grid = np.array(np.meshgrid(small_grid_x, small_grid_y)).T.reshape(-1, 2)
    return large_grid, small_grid

def get_grid_cell(point, grid, cell_size):
    x_index = int((point[0] - np.min(grid[:, 0])) // cell_size)
    y_index = int((point[1] - np.min(grid[:, 1])) // cell_size)
    return x_index, y_index

# 运行单步 SGP 建图流程
def run_sgp_mapping(x0, y0, xyz_points, local_range=3):
    mask = (
        (xyz_points[:, 0] >= x0 - local_range) & (xyz_points[:, 0] <= x0 + local_range) &
        (xyz_points[:, 1] >= y0 - local_range) & (xyz_points[:, 1] <= y0 + local_range)
    )
    local_points = xyz_points[mask]

    if len(local_points) < 50:
        return None, None, None, None, None

    X_train_np = local_points[:, :2]
    y_train_np = local_points[:, 2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)

    inducing_num = min(300, X_train.shape[0])
    idx = np.linspace(0, X_train.shape[0] - 1, inducing_num, dtype=int)
    inducing_pts = X_train[idx, :]

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SGPModel(X_train, y_train, likelihood, inducing_pts).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    start_time = time()
    for i in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    end_time = time()

    training_time = end_time - start_time

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

    train_preds = likelihood(model(X_train)).mean.detach().cpu().numpy()
    train_y_np = y_train.cpu().numpy()
    mse = mean_squared_error(train_y_np, train_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(train_y_np, train_preds)

    return xx, yy, mean, slope, var, training_time, rmse, r2

# 生成全局地图信息
def generate_global_map_info(xyz_points, path_points, local_range=3):
    global_xx_list = []
    global_yy_list = []
    global_mean_list = []
    global_slope_list = []
    global_var_list = []
    total_training_time = 0
    total_rmse = []
    total_r2 = []

    for x0, y0 in path_points:
        xx, yy, mean, slope, var, training_time, rmse, r2 = run_sgp_mapping(x0, y0, xyz_points, local_range)
        if xx is not None:
            global_xx_list.append(xx)
            global_yy_list.append(yy)
            global_mean_list.append(mean)
            global_slope_list.append(slope)
            global_var_list.append(var)
            total_training_time += training_time
            total_rmse.append(rmse)
            total_r2.append(r2)

    large_grid, small_grid = build_grid(xyz_points)

    return {
        "global_xx_list": global_xx_list,
        "global_yy_list": global_yy_list,
        "global_mean_list": global_mean_list,
        "global_slope_list": global_slope_list,
        "global_var_list": global_var_list,
        "total_training_time": total_training_time,
        "total_rmse": total_rmse,
        "total_r2": total_r2,
        "large_grid": large_grid,
        "small_grid": small_grid
    }
    