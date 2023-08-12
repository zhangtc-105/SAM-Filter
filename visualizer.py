import numpy as np
import open3d as o3d
from PIL import Image
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors

def read_ply(filename):
    
    cloud = o3d.io.read_point_cloud(filename)

    distances = np.linalg.norm(np.asarray(cloud.points), axis=1)
    radius = 3
    indices = np.where(distances<radius)[0]
    cloud_filtered = cloud.select_by_index(indices)

    o3d.visualization.draw_geometries([cloud_filtered])
    return np.asarray(cloud.points)

def visualize_depth():
    depth_image = np.load("depth.npy")
    print(depth_image.shape)
    depth_image = depth_image * 255 /1000
    depth_image = depth_image.astype(np.uint8)
    print(depth_image.shape)
    img = Image.fromarray(depth_image)
    img.save("depth.png")
def read_npy():
    indices = np.load("mask_indices.npy")
    indices = indices.T
    depth_image = np.load("depth.npy")
    print(depth_image.shape)

    height, width = depth_image.shape
    fx = 386.495    # focal length x
    fy = 386.495    # focal length y
    cx = 320.917    # optical center x
    cy = 244.474    # optical center y
    depth_scale = 100
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    print(indices.shape)
    X = (indices[:,0] - cx) * depth_image[indices[:,0],indices[:,1]]/ fx
    Y = (indices[:,1] - cy) * depth_image[indices[:,0],indices[:,1]]/ fy
    Z = depth_image[indices[:,0],indices[:,1]]
    print(X.shape, Y.shape, Z.shape)
    points = np.stack([X, Y, Z], axis=-1).reshape(-1,3)
    filtered_points = gmm(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("3d_reconstruction.pcd", pcd)
    return points

def gmm(points, num_points=80):
    z_score = stats.zscore(points)
    filtered_entries = (np.abs(z_score) < 3).all(axis=1)
    filtered_points = points[filtered_entries]
    gmm = GaussianMixture(n_components=num_points, covariance_type='full').fit(filtered_points)
    np.save("gmm.npy",np.array(gmm.means_))
    print(gmm.means_)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90, 0)
    ax.scatter(gmm.means_[:,0],gmm.means_[:,1], gmm.means_[:,2], c='b')
    plt.axis('off')
    plt.savefig("gmm.png")
    
    return gmm.means_

def compute_knn(points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    return distances[:,1:], indices[:,1:]

def main():
    # filename = "out.ply"
    # points = read_ply(filename)
    visualize_depth()
    points = read_npy()
    print(points.shape)

if __name__ == "__main__":
    main()