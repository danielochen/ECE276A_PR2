
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
import icp_warm_up.utils as U1
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

def plot_icp_alignment(P, Q, R, p, title="ICP alignment"):
    plt.figure(figsize=(6,6))
    plt.scatter(Q[0], Q[1], s=1, label="Q (target)", alpha=0.8)

    plt.scatter(P[0], P[1], s=1, label="P (source)", alpha=0.3)

    P_aligned = R @ P + p
    plt.scatter(P_aligned[0], P_aligned[1], s=1, label="R@P+p (aligned)", alpha=0.8)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()

def rotation2d(theta):
  # rotation matrix at angle theta
  R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]])
  return R

def Rz(yaw):
  # rotations about z axis
  R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw),  np.cos(yaw), 0],
                [0,  0, 1]])
  return R

def make_T(R, p):
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = p.reshape(3,)
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
  print(lidar_ranges.shape[0])
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
  print("lidar_ranges shape:", lidar_ranges.shape)
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
    print("encoder stamps size", len(encoder_stamps)) 
    print("imu stamps size", len(imu_stamps)) 
    
    # only use time range when both sensors are on 
    t_start = max(encoder_stamps[0], imu_stamps[0]) 
    t_end = min(encoder_stamps[-1], imu_stamps[-1]) 
    print("\n\ntime range", t_start, "to ", t_end) 


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
def Kabsch(P, Q):
  d, N = P.shape

  # mean of dataset
  P_mean = P.mean(axis=1, keepdims=True)
  Q_mean = Q.mean(axis=1, keepdims=True)

  # centroid cloud
  P_delta = P - P_mean                                                          # δzi - z_mean
  Q_delta = Q - Q_mean                                                          # δmi - m_mean

  # want to find R and p so that Rz + p = m
  # same as solving for the R and p that minimizes the sum ||(Rz_i + p) − m_i||
  # minimize with setting gradient with respect to p
  Q = P_delta @ Q_delta.T
 
  # solve with Kabsch Algorithm, SVD: let Q = UΣV.T
  U, Sigma, V_t = np.linalg.svd(Q)

  # print("\nU\n", U)
  # print("\nSigma\n", Sigma)
  # print("\nV transpose\n", V_t)

  R = V_t.T @ U.T                                                               # rotation matrix

  if np.linalg.det(R) < 0:                                                      # check if determinant is negative reflection case
    V_t[-1, :] *= -1
    R = V_t.T @ U.T

  p = Q_mean - R @ P_mean                                                       # bias

  return R, p

# ICP
def ICP(P, Q, R_init, p_init, iterations = 1000, tolerance = 1e-6):
  dim1, N_P = P.shape                                                           # source cloud
  dim2, N_Q = Q.shape                                                           # target cloud

  R = R_init                                                                    # set R, p to initial
  p = p_init

  mse_prev = np.inf                                                             # MSE init
  kdtree = cKDTree(Q.T)                                                         # target cloud KD-tree

  for i in range(iterations):
    P_correspond = R @ P + p                                                              # source transform                             

    # nearest neighbours in 2 or 3D using KD tree
    d, i_kd = kdtree.query(P_correspond.T)
    Q_correspond = Q[:, i_kd]

    d_R, d_p = Kabsch(P_correspond, Q_correspond)                               # iterate Kabsch step

    R = d_R @ R                                                                 # update transforms with R, p
    p = d_R @ p + d_p

    mse = np.mean(d**2)
    if abs(mse_prev - mse) < tolerance:                                         # exit early if reached
        break
    mse_prev = mse

  return R, p, mse


# 2A
def warmup_icp(model_name, pc_id, yaw_steps=36, iters=50, tol=1e-6):
    # target = canonical model, source = measured pc
    target = U1.read_canonical_model(model_name)
    source = U1.load_pc(model_name, pc_id)

    # convert to (3,N)
    Q = target.T
    P = source.T

    best = {"mse": np.inf, "R": None, "p": None, "yaw": None}

    yaws = np.linspace(-np.pi, np.pi, yaw_steps, endpoint=False)

    for yaw in yaws:
        R_init = Rz(yaw)
        p_init = np.zeros((3,1))

        R_icp, p_icp, mse = ICP(P, Q, R_init, p_init, iterations=iters, tolerance=tol)

        if mse < best["mse"]:
            best.update({"mse": mse, "R": R_icp, "p": p_icp, "yaw": yaw})

    T_best = make_T(best["R"], best["p"])
    return source, target, T_best, best


#2B
def ICP_dataset(dataset=20, scan_idx=100):
  encoder_data, lidar_data, imu_data, kinect_data = load_dataset(dataset)
  t_start, t_end, encoder_counts, encoder_t, v, imu_wz, imu_t, x, y, encoder_theta, d_x, d_y, d_theta = encoder_IMU_odometry(dataset)

  lidar_ranges = lidar_data["lidar_ranges"]
  lidar_stamps = lidar_data["lidar_stamps"]
  angle_min = lidar_data["lidar_angle_min"]
  angle_increment = lidar_data["lidar_angle_increment"]
  range_min = lidar_data["lidar_range_min"]
  range_max = lidar_data["lidar_range_max"]

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

  for i in range(n_scans - 1):  
    t0 = lidar_stamps[i]                                                        # sync timestamps with encoder
    t1 = lidar_stamps[i+1]
    j0 = int(np.argmin(np.abs(encoder_t - t0)))
    j1 = int(np.argmin(np.abs(encoder_t - t1)))

    dx = x[j1] - x[j0]                                            
    dy = y[j1] - y[j0]
    dtheta = encoder_theta[j1] - encoder_theta[j0]

    # Build source and target point clouds
    Q = lidar_scan_to_points(lidar_ranges[:, i], angles, range_min, range_max)

    P = lidar_scan_to_points(lidar_ranges[:, i+1], angles, range_min, range_max) 

    # R_init = rotation2d(d_theta[i])
    R_init = rotation2d(dtheta[i])

    # p_init = np.array([[d_x[i]],[d_y[i]]])
    p_init = np.array([[dx[i]],[dy[i]]])

    R_icp, p_icp, mse_icp = ICP(P, Q, R_init, p_init)                            # run ICP

    R_list.append(R_icp)
    p_list.append(p_icp)
    mse_list.append(mse_icp)


  return R_list, p_list, mse_list


if __name__ == '__main__':
  show_lidar()
  test_bresenham2D()
  test_map()
  
  
  

