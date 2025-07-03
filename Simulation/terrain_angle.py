# 文件名：terrain_angle.py

import numpy as np
from scipy.spatial.transform import Rotation as R

def car_angle(normal_vec, yaw_angle_rad):
    """
    计算车辆姿态在给定地形法向量与当前航向角（yaw）下的欧拉角（roll, pitch, yaw）

    参数：
    - normal_vec: ndarray of shape (3,), 法向量 (x, y, z)
    - yaw_angle_rad: float，当前车体朝向角（单位：弧度）

    返回：
    - euler_angles: ndarray of shape (3,), 单位为弧度，顺序为 ZYX → [yaw, pitch, roll]
    """
    v = normal_vec / np.linalg.norm(normal_vec)
    x_1 = np.array([np.cos(yaw_angle_rad), np.sin(yaw_angle_rad), 0])
    y_1 = np.array([-np.sin(yaw_angle_rad), np.cos(yaw_angle_rad), 0])

    # 正负方向判断
    normal_dir = -1 if np.dot(x_1, v) > 0 else 1

    # 计算新的局部坐标系
    x_2 = np.cross(y_1, v)
    x_2 /= np.linalg.norm(x_2)
    y_2 = np.cross(v, x_2)

    # 构造旋转矩阵（列向量是 x_2, y_2, v）
    rot_mat = np.column_stack((x_2, y_2, v))

    # 转换为欧拉角（ZYX顺序，返回 yaw, pitch, roll）
    euler = R.from_matrix(rot_mat).as_euler('zyx')

    # 限制 pitch 和 roll 在 [-π/2, π/2]
    for i in [1, 2]:
        if abs(euler[i]) > np.pi / 2:
            euler[i] = np.sign(euler[i]) * (np.pi - abs(euler[i]))

    return normal_dir * euler
