import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import load_data as L
import pr2_utils as U
from mapping_utils import disparity2pointcloud, update_occupancy_grid, camera2body, body_pose, get_lidar_points_in_body


dir = "outputs"
os.makedirs(dir, exist_ok=True)

def pose_from_icp(dataset: int, lidar_data):
    x = np.load(f"outputs/x_icp_{dataset}.npy")                                 # pose sampled at lidar timestamps
    y = np.load(f"outputs/y_icp_{dataset}.npy")
    th = np.load(f"outputs/theta_icp_{dataset}.npy")
    t = lidar_data["lidar_stamps"]                                              # timestamp for the poses
    return t, x, y, th


def pose_from_odometry(dataset: int, lidar_data):
    # Raw odom at encoder timestamps, then resample to lidar stamps
    _, _, _, encoder_t, _, _, _, x, y, encoder_theta, _, _, _= U.encoder_IMU_odometry(dataset)

    t_lidar = lidar_data["lidar_stamps"]                                        # timestamp for the poses resamped to lidar
    x_l = np.interp(t_lidar, encoder_t, x)                                      # resample x,y,theta over lidar timestamps     
    y_l = np.interp(t_lidar, encoder_t, y)
    th_l = np.interp(t_lidar, encoder_t, encoder_theta)

    return t_lidar, x_l, y_l, th_l

def build_full_occupancy(occ_grid, lidar_ranges, angles, r_min, r_max, x_pose, y_pose, theta_pose, res, map_min, skip = 5):
    num_scans = lidar_ranges.shape[1]

    for i in range(0, num_scans, skip):                                         # skip some scans
        points_b = get_lidar_points_in_body(lidar_ranges[:, i], angles, r_min, r_max)

        R = U.rotation2d(theta_pose[i])
        t = np.array([x_pose[i], y_pose[i]])
        points_w = (R @ points_b.T).T + t

        occ_grid = update_occupancy_grid(occ_grid, points_w, t, res, map_min)

        if i % 1000 == 0:
            print(f"Occupancy Map: Processed scan {i}/{num_scans}")

    return occ_grid

def build_texture_map_generic(texture_grid, size, map_min, res, kinect_data, t_pose, x_pose, y_pose, theta_pose, rgb_dir, disp_dir, rgb_files, disp_files, skip = 5):

    kinect_stamps = kinect_data["disp_stamps"]
    rgb_stamps = kinect_data["rgb_stamps"]
    num_frames = min(len(disp_files), len(kinect_stamps))

    T_kb = camera2body()

    for i in range(0, num_frames, skip):
        ts = kinect_stamps[i]

        xr, yr, tr = body_pose(ts, t_pose, x_pose, y_pose, theta_pose)

        disp_img = cv2.imread(os.path.join(disp_dir, disp_files[i]), cv2.IMREAD_UNCHANGED)
        rgb_idx = np.argmin(np.abs(rgb_stamps - ts))
        rgb_img = cv2.imread(os.path.join(rgb_dir, rgb_files[rgb_idx]))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        points_c, colors = disparity2pointcloud(disp_img, rgb_img)

        points_c_hom = np.hstack((points_c, np.ones((points_c.shape[0], 1))))
        points_b = (T_kb @ points_c_hom.T).T[:, :3]

        R_wb = np.array([
            [np.cos(tr), -np.sin(tr), 0],
            [np.sin(tr),  np.cos(tr), 0],
            [0,           0,          1]
        ])
        t_wb = np.array([xr, yr, 0])
        points_w = (R_wb @ points_b.T).T + t_wb

        floor_mask = (points_w[:, 2] > -0.20) & (points_w[:, 2] < 0.20)         # points 20 cm above and below ground
        floor_points = points_w[floor_mask]
        floor_colors = colors[floor_mask]

        if len(floor_points) > 0:
            cells = np.floor((floor_points[:, :2] - map_min) / res).astype(int)
            valid = (
                (cells[:, 0] >= 0) & (cells[:, 0] < size[0]) &
                (cells[:, 1] >= 0) & (cells[:, 1] < size[1])
            )
            cells = cells[valid]
            floor_colors = floor_colors[valid]
            texture_grid[cells[:, 0], cells[:, 1]] = floor_colors

        if i % 1000 == 0:
            print(f"Kinect processed: {i}/{num_frames}")
    print("Finished texture map")

    return texture_grid

def run_map_with_pose(dataset: int, t_pose, x_pose, y_pose, theta_pose, map_min, map_max, res=0.1, skip = 5):
    encoder_data, lidar_data, imu_data, kinect_data = L.load_dataset(dataset)

    # map init
    map_min = np.array(map_min, dtype=float)
    map_max = np.array(map_max, dtype=float)
    size = np.ceil((map_max - map_min) / res).astype(int)

    occ_grid = np.zeros(size)
    texture_grid = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # lidar params
    lidar_ranges = lidar_data["lidar_ranges"]
    angles = (
        lidar_data["lidar_angle_min"]
        + np.arange(lidar_ranges.shape[0]) * lidar_data["lidar_angle_increment"]
    ).flatten()
    r_min = lidar_data["lidar_range_min"]
    r_max = lidar_data["lidar_range_max"]

    # occupancy
    occ_grid = build_full_occupancy(
        occ_grid, lidar_ranges, angles, r_min, r_max,
        x_pose, y_pose, theta_pose, res, map_min,
        skip=skip
    )

    # kinect files
    rgb_dir = f"../data/dataRGBD/RGB{dataset}"
    disp_dir = f"../data/dataRGBD/Disparity{dataset}"
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    disp_files = sorted([f for f in os.listdir(disp_dir) if f.endswith(".png")])

    texture_grid = build_texture_map_generic(texture_grid, size, map_min, res, kinect_data, t_pose, x_pose, y_pose, theta_pose, rgb_dir, disp_dir, rgb_files, disp_files, skip=skip)

    return occ_grid, texture_grid, size, map_min, map_max, res, (x_pose, y_pose, theta_pose)


def run_icp_map(dataset=20, res=0.05, skip = 5):
    encoder_data, lidar_data, imu_data, kinect_data = L.load_dataset(dataset)
    t_pose, x_pose, y_pose, theta_pose = pose_from_icp(dataset, lidar_data)
    return run_map_with_pose(dataset, t_pose, x_pose, y_pose, theta_pose, map_min=(-30, -30), map_max=(30, 30), res=res, skip = skip)


def run_odom_map(dataset=21, res=0.05, skip = 5):
    encoder_data, lidar_data, imu_data, kinect_data = L.load_dataset(dataset)

    t_pose, x_pose, y_pose, theta_pose = pose_from_odometry(dataset, lidar_data)

    return run_map_with_pose(dataset,t_pose, x_pose, y_pose, theta_pose, map_min=(-40, -40), map_max=(40, 40), res=res, skip=skip)


def occ_to_img(occ_grid):
    """Log-odds -> displayable grayscale image (occupied black, free white, unknown gray)."""
    img = np.zeros_like(occ_grid, dtype=np.uint8)
    img[occ_grid > 0] = 0
    img[occ_grid < 0] = 255
    img[occ_grid == 0] = 127
    return img

def plot_occupancy(occ_grid, map_min, map_max, x, y, ICP = True, dataset = 20):
    if ICP:
        title = f"Occupancy Map from ICP Pose (Dataset {dataset})"
        filename = f"occupancy_icp_{dataset}.png"
    else:
        title = f"Occupancy Map from Odometry Pose (Dataset {dataset})"
        filename = f"occupancy_odom_{dataset}.png"
    
    
    img = occ_to_img(occ_grid)
    plt.figure(figsize=(8,8))
    plt.imshow(img.T, origin="lower", cmap="gray", extent=[map_min[0], map_max[0], map_min[1], map_max[1]])
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.savefig(os.path.join(dir, filename))
    print(f"Saved {os.path.join(dir, filename)}")
    plt.show()

def plot_texture_overlay(occ_grid, texture_grid, map_min, map_max, x=None, y=None, ICP = True, dataset = 20):
    if ICP:
        title = f"ICP Texture and Occupancy Map (Dataset {dataset})"
        filename = f"full_texture_icp_{dataset}.png"
    else:
        title = f"Odometry Texture and Occupancy Map (Dataset {dataset})"
        filename = f"full_texture_odom_{dataset}.png"

    display = np.zeros((*occ_grid.shape, 3), dtype=np.uint8)                    # background from occupancy
    display[occ_grid < 0] = [200, 200, 200]                                     # free
    display[occ_grid > 0] = [0, 0, 0]                                           # occupied

    textured = np.any(texture_grid > 0, axis=2)
    display[textured] = texture_grid[textured]

    plt.figure(figsize=(8,8))
    plt.imshow(display.transpose(1,0,2), origin="lower", extent=[map_min[0], map_max[0], map_min[1], map_max[1]])
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.savefig(os.path.join(dir, filename))
    print(f"Saved {os.path.join(dir, filename)}")
    plt.show()