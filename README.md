# ECE276A PR2

This project implements a full Simultaneous Localization and Mapping (SLAM) with IMU/Encoder Odometry, LiDAR scan matching, Kinect RGB-D texture mapping, and GTSAM Pose Graph Optimization.

**execute everything in `code/main.ipynb`**

## Python Scripts Overview

*   **`load_data.py`**: load sensor data
*   **`pr2_utils.py`**: utilities for parsing dataset logs, computing the Kabsch, ICP algorithm
*   **`mapping_utils.py`**: utilities for disparity-to-depth conversion, 3D projections, coordinate frame transformations, and log-odds grid updates
*   **`main_mapping.py`**: uses the utilities in `mapping_utils.py` to generate occupancy/texture maps
*   **`slam_gtsam.py`**: for GTSAM Pose Graph optimization of ICP trajectories with fixed-interval and proximity-based loop closures
*   **`visualize_kinect.py`**: shows how an image gets processed (RGB, Disparity, Projection Alignment, and Height Filter Mask)