import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
from icp_warm_up.utils import read_canonical_model, load_pc, visualize_icp_result
from scipy.spatial import cKDTree
from load_data import load_dataset
from pr2_utils import bresenham2D, test_bresenham2D, show_lidar, plot_map, lidar_scan_to_points, rotation2d, rotation_yaw, make_T3, make_T2, test_map, encoder_IMU_odometry, Kabsch, ICP, warmup_icp, build_trajectory, ICP_dataset, plot_trajectory, visualize_map

EPS = 1e-6

def intrinsic_matrix(f_su = 585.05, c_u = 242.94, c_v = 315.84):
    K = np.array([[f_su, 0, c_u],
                  [0, f_su, c_v],
                  [0, 0, 1]])
    
    return K

def disparity2depth(d):
    dd = -0.00304 * d + 3.31

    depth = np.zeros_like(dd)
    mask = dd > EPS
    depth[mask] = 1.03 / dd[mask]
    return depth, dd

def disparity2rgb(i, j, d):
    dd = -0.00304 * d + 3.31
    rgbi = ((526.37 * i) + 19276 - (7877.07 * dd)) / 585.051
    rgbj = (526.37 * j + 16662)/585.051

    rgbi = np.round(rgbi).astype(int)
    rgbj = np.round(rgbj).astype(int)
    return rgbi, rgbj


def disparity2pointcloud(d_image, rgb_image):
    # pinhole camera model
    # need to align every single pixel u,v in the depth image with teh rgbu, rgbv
    h_d, w_d = d_image.shape
    h_rgb, w_rgb, _ = rgb_image.shape

    K = intrinsic_matrix()
    i, j = np.meshgrid(np.arange(w_d), np.arange(h_d))                          # 2d pixel coords in depth image
    depth, dd = disparity2depth(d_image)

    z = depth
    x = (j - K[1, 2]) * depth / K[1, 1]
    y = (i - K[0, 2]) * depth / K[0, 0]
    points = np.stack((x, y, z), axis = -1)                                     # 3d points in camera frame
    mask_d = (z > 0.1) & (z < 30)                                               # filter out invalid points by depth

    rgbi, rgbj = disparity2rgb(i, j, d_image)
    mask_rgb = (rgbi >= 0) & (rgbi < w_rgb) & (rgbj >= 0) & (rgbj < h_rgb) & mask_d         # filter out invalid rgb 
    colours = np.zeros_like(points)
    colours = rgb_image[rgbj[mask_rgb], rgbi[mask_rgb]]
    return points[mask_rgb], colours

def rollpitchyaw2R(roll = 0.0, pitch = 0.36, yaw = 0.021):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx                                                            # extrinsic camera to body: z, y, x
    return R



def camera2body():
    t_camera_body = np.array([[0.18, 0.005, 0.36]])                           # translation from camera to body
    R_optical = rollpitchyaw2R()                                            # rotation from camera to body 
    R_o2b = np.array([[0, 0, 1],                                                # optical to body frame
                      [-1, 0, 0],
                      [0, -1, 0]])                                                   

    R_camera_body = R_optical @ R_o2b                                                     # camera to body frame

    
    T_cb = np.eye(4)                                                            # homogeneous transformation from camera to body
    T_cb[:3, :3] = R_camera_body
    T_cb[:3, 3] = t_camera_body
    print("T_cb: \n", T_cb)
    return T_cb



