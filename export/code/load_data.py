import numpy as np


def load_dataset(dataset=20):
  encoder = {}
  lidar = {}
  imu = {}
  kinect = {}
  
  with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder["encoder_counts"] = data["counts"]                                  # 4 x n encoder counts
    encoder["encoder_stamps"] = data["time_stamps"]                             # encoder time stamps

  with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar["lidar_angle_min"] = data["angle_min"]                                # start angle of the scan [rad]
    lidar["lidar_angle_max"] = data["angle_max"]                                # end angle of the scan [rad]
    lidar["lidar_angle_increment"] = data["angle_increment"]                    # angular distance between measurements [rad]
    lidar["lidar_range_min"] = data["range_min"]                                # minimum range value [m]
    lidar["lidar_range_max"] = data["range_max"]                                # maximum range value [m]
    lidar["lidar_ranges"] = data["ranges"]                                      # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar["lidar_stamps"] = data["time_stamps"]                                 # acquisition times of the lidar scans
    
  with np.load("../data/Imu%d.npz"%dataset) as data:
    imu["imu_angular_velocity"] = data["angular_velocity"]                      # angular velocity in rad/sec
    imu["imu_linear_acceleration"] = data["linear_acceleration"]                # accelerations in gs (gravity acceleration scaling)
    imu["imu_stamps"] = data["time_stamps"]                                     # acquisition times of the imu measurements
  
  with np.load("../data/Kinect%d.npz"%dataset) as data:
    kinect["disp_stamps"] = data["disparity_time_stamps"]                       # acquisition times of the disparity images
    kinect["rgb_stamps"] = data["rgb_time_stamps"]                              # acquisition times of the rgb images

    return encoder, lidar, imu, kinect

