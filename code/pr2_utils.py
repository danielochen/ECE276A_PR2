
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
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


def show_lidar():
  angles = np.arange(-135,135.25,0.25)*np.pi/180.0
  ranges = np.load("test_ranges.npy")
  plt.figure()
  ax = plt.subplot(111, projection='polar')
  ax.plot(angles, ranges)
  ax.set_rmax(10)
  ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
  ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
  ax.grid(True)
  ax.set_title("Lidar scan data", va='bottom')
  plt.show()


def plot_map(mapdata, cmap="binary"):
  plt.imshow(mapdata.T, origin="lower", cmap=cmap)

def test_map(dataset=20):
  t_start, t_end, encoder_counts, encoder_t, v, imu_wz, imu_t, x, y, encoder_theta = encoder_IMU_odometry(dataset)
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
    R = np.array([[np.cos(theta_r), -np.sin(theta_r)], 
                  [np.sin(theta_r),  np.cos(theta_r)]])
    
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
  # plt.plot(points_w[:,0],points_w[:,1],'.k')
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
    if plot:
        plt.figure()
        plt.plot(encoder_stamps_sync, encoder_theta)
        plt.title("Heading vs Time (Integrated IMU)")
        plt.xlabel("time (s)")
        plt.ylabel("theta (rad)")
        plt.grid(True)
        plt.show()


        print("dt min:", encoder_dt.min(), "dt median:", np.median(encoder_dt))
        print("v max:", v.max())
        print("where v > 4 m/s:", np.sum(v > 4))

        t_mid = 0.5 * (encoder_stamps_sync[:-1] + encoder_stamps_sync[1:])

        plt.figure()
        plt.plot(t_mid, v)
        plt.title("Linear velocity v(t) from encoders")
        plt.xlabel("time (s)")
        plt.ylabel("v (m/s)")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(t_mid, omega)
        plt.title("Yaw rate ω(t) aligned to encoder intervals")
        plt.xlabel("time (s)")
        plt.ylabel("ω (rad/s)")
        plt.grid(True)
        plt.show()

        dt = imu_dt
        plt.figure()
        plt.plot(dt)
        plt.title("Encoder dt per step")
        plt.xlabel("step k")
        plt.ylabel("dt (s)")
        plt.grid(True)
        plt.show()

        print("dt mean:", dt.mean(), "min:", dt.min(), "max:", dt.max())
    return t_start, t_end, encoder_counts_sync, encoder_stamps_sync, v, imu_angular_velocity_yaw, imu_stamps_yaw, x, y, encoder_theta
if __name__ == '__main__':
  show_lidar()
  test_bresenham2D()
  test_map()
  
  
  

