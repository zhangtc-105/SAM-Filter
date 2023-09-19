import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import cv2

def edge_fn(distances, sigma=1.0):
    # distances is a matrix of distances between points
    return np.exp(-distances**2/(2*sigma**2))

def get_adjacency_matrix(data):
    data = data-data.mean(axis=0)
    sum_square = np.sum(data**2, axis=1)
    distance_matrix = np.sqrt(sum_square[:,None] + sum_square[None,:] - 2*data.dot(data.T))
    
    # find sigma using mean of second nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    sigma = np.mean(distances[:,1])
    print(sigma)
    graph = edge_fn(distance_matrix, sigma)
    np.fill_diagonal(graph, 0)

    return graph

def graph_diffuse(graph, data, steps=10, src=5, dst=46):
    # src is the index of the point we want to diffuse from
    # dst is the index of the point we want to diffuse to
    # steps is the number of steps we want to diffuse
    print("start graph diffusion")
    heat_distribution = np.zeros(graph.shape[0])
    heat_distribution[src] = 1.0

    degree_matrix = np.diag(np.sum(graph, axis=1))
    adj_matrix = graph
    normalized_adj_matrix = np.linalg.inv(degree_matrix).dot(adj_matrix)

    for i in range(steps):
        heat_distribution = normalized_adj_matrix.dot(heat_distribution)
        heat_distribution[src]=1.0
        heat_distribution[dst]=0.0

    rank = np.argsort(heat_distribution)
    return rank
    idx_to_rank = np.zeros(rank.shape[0], dtype=np.int32)
    for i in range(rank.shape[0]):
        idx_to_rank[rank[i]]=i
    print(idx_to_rank)
    return idx_to_rank
    # lims = [np.min(data), np.max(data)]
    # std_data = data - data.mean(axis=0)
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(azim=0, elev=90)
    # ax.set_xlim(np.min(std_data[:, 0])-10, np.max(std_data[:, 0])+10)
    # ax.set_ylim(np.min(std_data[:, 1])-10, np.max(std_data[:, 1])+10)
    # ax.set_zlim(np.min(std_data[:, 2])-10, np.max(std_data[:, 2])+10)
    # ax.set_axis_off()
    # scatter = ax.scatter(std_data[:,0], std_data[:,1], std_data[:,2], c=idx_to_rank, cmap='viridis')
    # # for i in range(len(data)):
    # #     ax.text(std_data[i, 0], std_data[i, 1], std_data[i, 2], f"{idx_to_rank[i]}")
    # cbar_ax = fig.add_axes([0.8, 0.3, 0.05, 0.5])
    
    # cbar = plt.colorbar(scatter,shrink=0.3,cax=cbar_ax)
    # cbar.ax.tick_params(labelsize=20)
    # plt.savefig("diffusion.png")
    
def main():
    points = np.load("gmm.npy")
    exclude_indices = [28,38,33,21,55,21,57,69,65]
    mask = np.ones(points.shape[0], dtype=bool)
    mask[exclude_indices] = False
    points = points[mask]
    fig = cv2.imread("Images/IMG_E4472.JPG")
    height, width, _ = fig.shape
    height, width = height//4, width//4
    plt.rcParams['axes.axisbelow'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].scatter(points[:,1], -points[:,0], c='r', s=10)
    ax[0].set_title("Before indexing")
    ax[0].set_xlim(0, width)
    ax[0].set_ylim(-height, 0)
    
    for i in range(points.shape[0]):
        ax[0].text(points[i, 1], -points[i,0], f"{i}")
    
    g = get_adjacency_matrix(points)
    rank = graph_diffuse(g, points, steps=9999, src=8, dst=41)
    ordered_points = points[rank]
    ax[1].scatter(ordered_points[:,1], -ordered_points[:,0], c='b', s=10)
    ax[1].set_title("After indexing")
    ax[1].set_xlim(0, width)
    ax[1].set_ylim(-height, 0)
    for i in range(ordered_points.shape[0]):
        ax[1].text(ordered_points[i, 1], -ordered_points[i,0], f"{i}")
    ax[2].scatter(ordered_points[:,1], -ordered_points[:,0], color=(0,0,0), s=10)
    ax[2].plot(ordered_points[:,1], -ordered_points[:,0], color=(0,0,0), linewidth=0.5)
    ax[2].set_title("Adding Edges")
    ax[2].set_xlim(0, width)
    ax[2].set_ylim(-height, 0)

    plt.tight_layout()
    plt.savefig("indexing.png")
    

    # plt.figure()
    # plt.matshow(g)
    # plt.savefig("adjacency.png")
    # print(g.shape)
    

if __name__ == '__main__':
    main()