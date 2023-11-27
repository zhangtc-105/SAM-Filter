import numpy as np
import open3d as o3d
from PIL import Image
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import os
def read_ply(filename):
    print("Load %s ply point cloud, print it, and render it", str(filename))
    cloud = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([cloud])
    return np.asarray(cloud.points)

def visualize_depth():
    depth_image = np.load("depth.npy")
    print(depth_image.shape)
    depth_image = depth_image * 255 /1000
    depth_image = depth_image.astype(np.uint8)
    print(depth_image.shape)
    img = Image.fromarray(depth_image)
    img.save("depth.png")
def reconstruction(depth_file_path, indices_file_path, output_3d_file_path, output_3d_ply_path):
    indices = np.load(indices_file_path)
    indices = indices.T
    print(indices.shape)
    depth_image = np.load(depth_file_path)


    height, width = depth_image.shape
    fx = 386.495  # focal length x
    fy = 386.495  # focal length y
    cx = 320.917  # optical center x
    cy = 244.474  # optical center y
    
    X = (indices[:,0] - cx) * depth_image[indices[:,0], indices[:,1]] / fx
    Y = (indices[:,1] - cy) * depth_image[indices[:,0], indices[:,1]] / fy
    Z = depth_image[indices[:,0], indices[:,1]]
    points = np.stack([X, Y, Z], axis=-1).reshape(-1,3)
    filtered_points = gmm(points)
    print(filtered_points.shape)
    # Storing the 3D reconstruction points
    np.save(output_3d_file_path, filtered_points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Save Point Cloud as .ply file
    o3d.io.write_point_cloud(output_3d_ply_path, pcd)
    
    # Visualization
    
    return points


def gmm(points, num_points=80):
    z_score = stats.zscore(points)
    filtered_entries = (np.abs(z_score) < 3).all(axis=1)
    filtered_points = points[filtered_entries]
    gmm = GaussianMixture(n_components=num_points, covariance_type='full').fit(filtered_points)
    
    return gmm.means_


def main():
    output_dir = "output"
    depth_dir = "depth_arrays"
    reconstruction_dir = "reconstruction"
    for file_name in os.listdir(output_dir):
        if file_name.endswith("mask_indices.npy"):
            identifier = file_name.split('mask_indices')[0][4:]  # Extracting number from "rgb_{number}mask_indices.npy"

            # Construct the filenames
            indices_file_path = os.path.join(output_dir, f'rgb_{identifier}mask_indices.npy')
            depth_file_path = os.path.join(depth_dir, f'depth_{identifier}.npy')
            
            if os.path.exists(indices_file_path) and os.path.exists(depth_file_path):

            # Reconstruct the 3D points
                if not os.path.exists(reconstruction_dir):
                    os.makedirs(reconstruction_dir)
                output_3d_file_path = os.path.join(reconstruction_dir, f'reconstruction_{identifier}.npy')
                output_3d_ply_path = os.path.join(reconstruction_dir, f'reconstruction_{identifier}.ply')
                reconstruction(depth_file_path, indices_file_path, output_3d_file_path, output_3d_ply_path)


    # indices = np.load("mask_indices.npy")
    # indices = indices.T
    # indices = np.hstack((indices, np.zeros((indices.shape[0], 1))))
    # gmm = GaussianMixture(n_components=70)
    # gmm.fit(indices)
    # points = gmm.means_
    # points = points.astype(np.int32)
    # np.save("gmm.npy", points)

if __name__ == "__main__":
    main()
    output_dir = "reconstruction"
    for file_name in os.listdir(output_dir):
        if file_name.endswith("ply"):
            path = os.path.join(output_dir, file_name)
            read_ply(path)