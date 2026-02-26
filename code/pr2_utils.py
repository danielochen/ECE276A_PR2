import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
from icp_warm_up.utils import read_canonical_model, load_pc, visualize_icp_result
from scipy.spatial import cKDTree
from load_data import load_dataset

def tic():
  return time.time()

def toc(tstart, name="Operation"):
  print('%s took: %s sec.\n' % (name,(time.time() - tstart)))


def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


def test_bresenham2D():
  import time
  sx = 0
  sy = 1
  print("Testing bresenham2D...")
  r1 = bresenham2D(sx, sy, 10, 5)
  r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10],[1,1,2,2,3,3,3,4,4,5,5]])
  r2 = bresenham2D(sx, sy, 9, 6)
  r2_ex = np.array([[0,1,2,3,4,5,6,7,8,9],[1,2,2,3,3,4,4,5,5,6]])
  if np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex),np.sum(r2 == r2_ex) == np.size(r2_ex)):
    print("...Test passed.")
  else:
    print("...Test failed.")

  # Timing for 1000 random rays
  num_rep = 1000
  start_time = time.time()
  for i in range(0,num_rep):
	  x,y = bresenham2D(sx, sy, 500, 200)
  # print("1000 raytraces: --- %s seconds ---" % (time.time() - start_time))


def show_lidar(ranges, angles, rmax = 30, title = "Lidar scan"):
  # angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  # ranges = np.load("test_ranges.npy")
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(rmax)
  # ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  # ax.set_title("Lidar scan data", va='bottom')
  ax.set_title(title, va='bottom')
  plt.show()


def plot_map(mapdata, cmap="binary"):
  plt.imshow(mapdata.T, origin="lower", cmap=cmap)


def lidar_scan_to_points(ranges, angles, range_min, range_max):
    ranges = np.asarray(ranges).reshape(-1)
    angles = np.asarray(angles).reshape(-1)
    valid = np.logical_and((ranges < range_max),(ranges > range_min))

    r = ranges[valid]
    a = angles[valid]

    x = r * np.cos(a)
    y = r * np.sin(a)

    return np.vstack((x, y))

def rotation2d(theta):
  # rotation matrix at angle theta
  R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])
  return R

def rotation_yaw(yaw):
  # rotations about z axis
  R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,  0, 1]])
  return R

def make_T3(R, p):
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = p.reshape(3,)
  return T

def make_T2(R, p):
  T = np.eye(3)
  T[:2, :2] = R
  T[:2, 2] = p.reshape(2,)
  return T

def test_map(dataset=20):
  t_start, t_end, encoder_counts, encoder_t, v, imu_wz, imu_t, x, y, encoder_theta, d_x, d_y, d_theta = encoder_IMU_odometry(dataset)
  encoder_data, lidar_data, imu_data, kinect_data = load_dataset(dataset)

  angle_min = lidar_data["lidar_angle_min"]                                     # start angle of the scan [rad]
  angle_max = lidar_data["lidar_angle_max"]                                     # end angle of the scan [rad]
  angle_increment = lidar_data["lidar_angle_increment"]                         # angular distance between measurements [rad]
  range_min = lidar_data["lidar_range_min"]                                     # minimum range value [m]
  range_max = lidar_data["lidar_range_max"]                                     # maximum range value [m]
  lidar_ranges = lidar_data["lidar_ranges"]                                     # range data [m] (Note: values < range_min or > range_max should be discarded)
  lidar_stamps = lidar_data["lidar_stamps"]                                     # acquisition times of the lidar scans

  # need to transform the lidar measurements bsaed off the 
  # print(lidar_ranges.shape[0])
  # Initialize a grid map
  MAP = {}
  MAP['res'] = np.array([0.05, 0.05])    # meters
  MAP['min'] = np.array([-20.0, -20.0])  # meters
  MAP['max'] = np.array([20.0, 20.0])    # meters
  MAP['size'] = np.ceil((MAP['max'] - MAP['min']) / MAP['res']).astype(int)
  isEven = MAP['size']%2==0
  MAP['size'][isEven] = MAP['size'][isEven]+1 # Make sure that the map has an odd size so that the origin is in the center cell
  MAP['map'] = np.zeros(MAP['size'])
  
  # Load Lidar scan
  # print("lidar_ranges shape:", lidar_ranges.shape)
  n_rays, n_scans = lidar_ranges.shape
  # angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  angles = (angle_min + np.arange(n_rays) * angle_increment).reshape(-1)
  all_points_w = []
  for i in range(n_scans):
    # ranges = np.load("test_ranges.npy")
    ranges = lidar_ranges[:, i]
    # valid1 = np.logical_and((ranges < 30),(ranges> 0.1))
    valid1 = np.logical_and((ranges < range_max),(ranges > range_min))
  
    t_scan = lidar_stamps[i]
    j = np.argmin(np.abs(encoder_t - t_scan))                                     # find nearest encoder pose to LiDAR scan
    x_r = x[j]
    y_r = y[j]
    theta_r = encoder_theta[j]
  
    # Lidar points in the sensor/body frame in range 
    points_b = np.column_stack((ranges[valid1]*np.cos(angles[valid1]), ranges[valid1]*np.sin(angles[valid1])))

    # convert to world frame
    R = rotation2d(theta_r)
  
    points_w = (R @ points_b.T).T + np.array([x_r, y_r])                            # world frame
    
    all_points_w.append(points_w)
    # Convert from meters to cells
    cells = np.floor((points_w - MAP['min']) / MAP['res']).astype(int)
    
    # Insert valid points in the map
    valid2 = np.all((cells >= 0) & (cells < MAP['size']),axis=1)
    MAP['map'][tuple(cells[valid2].T)] = 1

  # Plot the Lidar points
  all_points_w = np.vstack(all_points_w)
  fig1 = plt.figure()
  # plt.plot(all_points_w[:,0],all_points_w[:,1],'.k')
  plt.scatter(all_points_w[:,0], all_points_w[:,1], 1, 'k')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(f'Lidar scan of dataset {dataset}')
  plt.axis('equal')
  plt.grid(True)
  
  # Plot the grid map
  fig2 = plt.figure()
  plot_map(MAP['map'],cmap='binary')
  plt.title(f'Grid map of dataset {dataset}')
  plt.grid(True)
  plt.axis('equal')
  
  plt.show()
  return MAP



def encoder_IMU_odometry(dataset = 20, d_tick = 0.0022, plot = False):
    encoder_data, lidar_data, imu_data, kinect_data = load_dataset(dataset)
    encoder_counts = encoder_data["encoder_counts"]
    encoder_stamps = encoder_data["encoder_stamps"] # 40 Hz 
    
    imu_angular_velocity = imu_data["imu_angular_velocity"] 
    imu_stamps = imu_data["imu_stamps"] # 100 Hz 
    # print("encoder stamps size", len(encoder_stamps)) 
    # print("imu stamps size", len(imu_stamps)) 
    
    # only use time range when both sensors are on 
    t_start = max(encoder_stamps[0], imu_stamps[0]) 
    t_end = min(encoder_stamps[-1], imu_stamps[-1]) 
    # print("\n\ntime range", t_start, "to ", t_end) 


    imu_range = (imu_stamps >= t_start) & (imu_stamps <= t_end)
    
    imu_angular_velocity_yaw = imu_angular_velocity[2, imu_range]               # imu yaw only synced max start, min end
    imu_stamps_yaw = imu_stamps[imu_range]                                      # sync up imu with average imu over longer encoder time steps since encoder is slower/less samples 

    encoder_range = (encoder_stamps >= t_start) & (encoder_stamps <= t_end)     # encoder to max start, min end 
    encoder_stamps_sync = encoder_stamps[encoder_range]
    encoder_counts_sync = encoder_counts[:, encoder_range]

    encoder_dt = np.diff(encoder_stamps_sync) 
    
    # encoder displacements 
    d_R = (encoder_counts_sync[0,:] + encoder_counts_sync[2,:]) * d_tick/2                # (FR+RR)/2 * 0.0022m 
    d_L = (encoder_counts_sync[1,:]+encoder_counts_sync[3,:]) * d_tick/2                  # (FL+RL)/2 * 0.0022m 
    d_center = (d_R + d_L) / 2                                                  # distance traveled by robot center 
    
    v = d_center[1:]/encoder_dt # velocity at robot center 
    
    imu_dt = np.diff(imu_stamps_yaw) 
    imu_theta_yaw = np.zeros(imu_angular_velocity_yaw.shape) 
    imu_theta_yaw[1:] = np.cumsum((imu_angular_velocity_yaw[0:-1]+imu_angular_velocity_yaw[1:])/2 * imu_dt) # interporlate imu data instead of finding nearest points to encoder timestamp 
    
    encoder_theta = np.interp(encoder_stamps_sync, imu_stamps_yaw, imu_theta_yaw) # interpolate imu angular distance over encoder timestamp 
    d_theta = np.diff(encoder_theta) 
    omega = d_theta / encoder_dt 
    
    N = len(encoder_dt) 
    x = np.zeros(N+1) 
    y = np.zeros(N+1) 
    # theta = np.zeros(N+1) # use abosulte, dead reckon the changes in orientation every step 
    
    # theta[1:] = np.cumsum(omega * encoder_dt) 
    # for i in range(N):
    #     x[i+1] = x[i] + d_center[i] * np.cos(encoder_theta[i]) 
    #     y[i+1] = y[i] + d_center[i] * np.sin(encoder_theta[i]) 
    
    d_x = d_center[1:] * np.cos(encoder_theta[:-1]) # d_center = v * dt 
    d_y = d_center[1:] * np.sin(encoder_theta[:-1]) 
    
    x[1:] = np.cumsum(d_x) 
    y[1:] = np.cumsum(d_y)

    plt.figure()
    plt.plot(x, y)
    plt.axis("equal")
    plt.title(f"Odometry Trajectory (Encoder + IMU) of dataset {dataset}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
    plt.show()

    return t_start, t_end, encoder_counts_sync, encoder_stamps_sync, v, imu_angular_velocity_yaw, imu_stamps_yaw, x, y, encoder_theta, d_x, d_y, d_theta

# Kabsch Algorithm from HW2
def Kabsch(Z, M):
  d, N = Z.shape

  # mean of dataset
  Z_mean = Z.mean(axis=1, keepdims=True)
  M_mean = M.mean(axis=1, keepdims=True)

  # centroid cloud
  Z_delta = Z - Z_mean                                                          # δzi - z_mean
  M_delta = M - M_mean                                                          # δmi - m_mean

  # want to find R and p so that Rz + p = m
  # same as solving for the R and p that minimizes the sum ||(Rz_i + p) − m_i||
  # minimize with setting gradient with respect to p
  Q_k = Z_delta @ M_delta.T
 
  # solve with Kabsch Algorithm, SVD: let Q = UΣV.T
  U, Sigma, V_t = np.linalg.svd(Q_k)

  # print("\nU\n", U)
  # print("\nSigma\n", Sigma)
  # print("\nV transpose\n", V_t)

  R = V_t.T @ U.T                                                               # rotation matrix

  if np.linalg.det(R) < 0:                                                      # check if determinant is negative reflection case
    V_t[-1, :] *= -1
    R = V_t.T @ U.T

  p = M_mean - R @ Z_mean                                                       # bias

  return R, p

# ICP
def ICP(Z, M, R_init, p_init, iterations = 50, tolerance = 1e-6, d_max = 2.0):
  # t_T_t+1​ = M_T_Z, source to target or Z to M
  dim1, N_Z = Z.shape                                                           # source cloud
  dim2, N_M = M.shape                                                           # target cloud

  R = R_init                                                                    # set R, p to initial
  p = p_init

  mse_prev = np.inf                                                             # MSE init
  kdtree = cKDTree(M.T)                                                         # target cloud KD-tree

  for i in range(iterations):
    Z_correspond = R @ Z + p                                                    # source transform                             

    # nearest neighbours in 2 or 3D using KD tree
    d, i_kd = kdtree.query(Z_correspond.T)
    valid_matches = d < d_max                                                   # filter out outliers that are too far apart, thanks Hesam
    
    Z_correspond = Z_correspond[:, valid_matches]
    M_correspond = M[:, i_kd[valid_matches]]
    

    d_R, d_p = Kabsch(Z_correspond, M_correspond)                               # iterate Kabsch step

    R = d_R @ R                                                                 # update transforms with R, p
    p = d_R @ p + d_p

    mse = np.mean(d[valid_matches]**2)                                          # MSE over valid matches
    if abs(mse_prev - mse) < tolerance:                                         # exit early if reached
        break
    mse_prev = mse

  return R, p, mse


# 2A
def warmup_icp(model_name, pc_id, yaw_steps=36, iterations=1000, tolerance=1e-6):
    # t_T_t+1​ = M_T_Z, source to target or Z to M
    # target = canonical model, source = measured pc
    Z = load_pc(model_name, pc_id).T                                            # source 
    M  = read_canonical_model(model_name).T                                     # target

    mse_best = np.inf
    R_best = None
    p_best = None
    yaw_best = None

    yaws = np.linspace(-np.pi, np.pi, yaw_steps, endpoint=False)                # 36 values between -pi and pi

    for yaw in yaws:
        R_init = rotation_yaw(yaw)
        Z_mean = Z.mean(axis=1, keepdims=True)
        M_mean = M.mean(axis=1, keepdims=True)
        p_init = M_mean - Z_mean
        # print("z mean ", Z_mean)
        # print("m mean ", M_mean)
        # print("p init ", p_init)

        R_icp, p_icp, mse_icp = ICP(Z, M, R_init, p_init, iterations, tolerance)

        if mse_icp < mse_best:
          mse_best = mse_icp
          R_best = R_icp
          p_best = p_icp
          yaw_best = yaw

    pose_best = make_T3(R_best, p_best)
    return Z.T, M.T, pose_best, mse_best, R_best, p_best, yaw_best





#2B

def build_trajectory(R_list, p_list):
    # input R_list and p_list contain transformations i_T_i+1
    T = np.eye(3)
    traj = [T.copy()]

    for R, p in zip(R_list, p_list):
        T = T @ make_T2(R, p)
        traj.append(T.copy())
    traj = np.array(traj)

    return traj, traj[:,0,2], traj[:,1,2]

def ICP_dataset(dataset=20):
  # t_T_t+1​ = M_T_Z, source to target or Z to M
  encoder_data, lidar_data, imu_data, kinect_data = load_dataset(dataset)
  t_start, t_end, encoder_counts, encoder_t, v, imu_wz, imu_t, x, y, encoder_theta, d_x, d_y, d_theta = encoder_IMU_odometry(dataset)
  

  lidar_ranges = lidar_data["lidar_ranges"]
  lidar_stamps = lidar_data["lidar_stamps"]
  angle_min = lidar_data["lidar_angle_min"]
  angle_increment = lidar_data["lidar_angle_increment"]
  range_min = max(lidar_data["lidar_range_min"], 0.1)                           # TA advice: filter out < 0.1m
  range_max = min(lidar_data["lidar_range_max"], 30.0) # TA advice: filter out > 30.0m

  # need to transform the lidar measurements bsaed off the 
  print(lidar_ranges.shape[0])

  # Load Lidar scan
  print("lidar_ranges shape:", lidar_ranges.shape)
  n_rays, n_scans = lidar_ranges.shape
  # angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  angles = (angle_min + np.arange(n_rays) * angle_increment).reshape(-1)

  R_list = []
  p_list = []
  mse_list = []

  T_total = np.eye(3)                                                            # initialize T pose
  x_icp = [0.0]
  y_icp = [0.0]
  theta_icp = [0.0]

  for i in range(n_scans - 1):  
    t0 = lidar_stamps[i]                                                        # sync timestamps with encoder
    t1 = lidar_stamps[i+1]
    j0 = int(np.argmin(np.abs(encoder_t - t0)))
    j1 = int(np.argmin(np.abs(encoder_t - t1)))

    dx_w = x[j1] - x[j0]                                                        # world frame odometry from encoder                                  
    dy_w = y[j1] - y[j0]
    dtheta = encoder_theta[j1] - encoder_theta[j0]

    R_0 = rotation2d(encoder_theta[j0])                                         # robot orientation at t = 0

    dp_local = R_0.T @ np.array([[dx_w], [dy_w]])                               # world to robot/body fram at t =0
    dx_b = dp_local[0, 0]
    dy_b = dp_local[1, 0]

 
    
    # Build source and target point clouds
    # t_T_t+1​ = M_T_Z, source to target or Z to M, i+1 to i
    Z = lidar_scan_to_points(lidar_ranges[:, i+1], angles, range_min, range_max)

    M = lidar_scan_to_points(lidar_ranges[:, i], angles, range_min, range_max) 

    # R_init = rotation2d(d_theta[i])
    R_init = rotation2d(dtheta)

    # p_init = np.array([[d_x[i]],[d_y[i]]])
    p_init = np.array([[dx_b], [dy_b]])

    R_icp, p_icp, mse_icp = ICP(Z, M, R_init, p_init, iterations=1000, tolerance=1e-6)               # run ICP, t_T_t+1 = M_T_Z

    # append list of R, p between scans

    R_list.append(R_icp)
    p_list.append(p_icp)
    mse_list.append(mse_icp)

    T_rel = make_T2(R_icp, p_icp)
    T_total = T_total @ T_rel
    
    x_icp.append(T_total[0, 2])
    y_icp.append(T_total[1, 2])
    theta_icp.append(np.arctan2(T_total[1, 0], T_total[0, 0]))

    # if i % 200 == 0:
    # plot_icp_step(Z, M, R_icp, p_icp, i) #TEST
    #   print(i, "MSE:", mse_icp) # TEST

  traj, x_icp, y_icp = build_trajectory(R_list, p_list)
  print("dataset: ", dataset, "mean mse: ", np.mean(mse_list), "max mse: ", np.max(mse_list))
  visualize_map(lidar_ranges, angles, traj, range_min, range_max)

  x_icp, y_icp, theta_icp = np.array(x_icp), np.array(y_icp), np.array(theta_icp)

  return R_list, p_list, mse_list, x_icp, y_icp, theta_icp






def plot_icp_step(Z, M, R, p, step):
    plt.figure(figsize=(6,6))

    # target scan (t+1)
    plt.scatter(M[0], M[1], s=1, c='black', label="target")

    # original source
    plt.scatter(Z[0], Z[1], s=1, c='red', alpha=0.3, label="source")

    # aligned source
    Z_aligned = R @ Z + p
    plt.scatter(Z_aligned[0], Z_aligned[1], s=1,
                c='blue', label="aligned")

    plt.axis('equal')
    plt.grid(True)
    plt.title(f"ICP alignment step {step}")
    plt.legend()
    plt.show()

def plot_trajectory(x, y, x_icp, y_icp, dataset):
  plt.figure()
  plt.plot(x, y, label="Odometry", alpha=0.5)
  plt.plot(x_icp, y_icp, label="ICP trajectory")
  plt.axis('equal')
  plt.grid(True)
  plt.legend()
  plt.title(f"Trajectory comparison of dataset {dataset}")
  plt.show()

def visualize_map(lidar_ranges, angles, traj, range_min, range_max):
  plt.figure(figsize=(7,7))

  for i, T in enumerate(traj):

      pts = lidar_scan_to_points(
          lidar_ranges[:,i],
          angles,
          range_min,
          range_max
      )

      R = T[:2,:2]
      t = T[:2,2:3]

      pts_w = R @ pts + t

      plt.scatter(pts_w[0], pts_w[1], s=0.2, c='k')

  plt.axis('equal')
  plt.grid(True)
  plt.title("ICP Map")
  plt.show()


if __name__ == '__main__':
  show_lidar()
  test_bresenham2D()
  test_map()
  
  
  

