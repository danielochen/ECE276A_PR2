import os
import scipy.io as sio
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

def read_canonical_model(model_name):
  '''
  Read canonical model from .mat file
  model_name: str, 'drill' or 'liq_container'
  return: numpy array, (N, 3)
  '''
  # model_fname = os.path.join('./data', model_name, 'model.mat')
  model_fname = os.path.join(DATA_DIR, model_name, "model.mat")
  model = sio.loadmat(model_fname)

  cano_pc = model['Mdata'].T / 1000.0 # convert to meter

  return cano_pc


def load_pc(model_name, id):
  '''
  Load point cloud from .npy file
  model_name: str, 'drill' or 'liq_container'
  id: int, point cloud id
  return: numpy array, (N, 3)
  '''
  # pc_fname = os.path.join('./data', model_name, '%d.npy' % id)
  pc_fname = os.path.join(DATA_DIR, model_name, f"{id}.npy")
  pc = np.load(pc_fname)

  return pc


def visualize_icp_result(source_pc, target_pc, pose, model_name  = "drill", pc_id = 0, show = False, save = True):
  '''
  Visualize the result of ICP
  source_pc: numpy array, (N, 3)
  target_pc: numpy array, (N, 3)
  pose: SE(4) numpy array, (4, 4)
  '''

  source_pcd = o3d.geometry.PointCloud()
  source_pcd.points = o3d.utility.Vector3dVector(source_pc.reshape(-1, 3))
  source_pcd.paint_uniform_color([0, 0, 1])

  target_pcd = o3d.geometry.PointCloud()
  target_pcd.points = o3d.utility.Vector3dVector(target_pc.reshape(-1, 3))
  target_pcd.paint_uniform_color([1, 0, 0])

  source_pcd.transform(pose)

  if save:
    o3d.io.write_point_cloud(f"{model_name}_{pc_id}_source.ply", source_pcd)
    o3d.io.write_point_cloud(f"{model_name}_{pc_id}_target.ply", target_pcd)

  
  if show:
    o3d.visualization.draw_geometries([source_pcd, target_pcd])


def show_icp_warmup(model_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    pc_id=(0, 1, 2, 3)
    title = f"Point Clouds for {model_name}"
    plt.suptitle(title)
    for ax, id in zip(axes, pc_id):
        fname = f"{model_name}_{id}.png"
        path = os.path.join("../images", fname)
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(fname)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


  
  


