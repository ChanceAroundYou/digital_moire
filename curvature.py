import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree 

def compute_average_curvature(ply_file_path):
    # 读取PLY文件
    mesh = o3d.io.read_triangle_mesh(ply_file_path) 
    vertices = np.asarray(mesh.vertices) 
    triangles = np.asarray(mesh.triangles)
    # 构建KD树用于邻居搜索
    tree = cKDTree(vertices)
    average_curvatures = []
    for i in range(len(vertices)):
        # 搜索每个顶点的邻居
        _, neighbor_indices = tree.query(vertices[i], k=10) # 取10个最近邻居
        # 计算每个顶点的平均曲率 
        # 这里简单使用Laplace-Beltrami算子的离散近似
        laplacian = np.zeros(3) 
        for j in neighbor_indices: 
            if j != i: 
                laplacian += vertices[j] - vertices[i]
        laplacian /= len(neighbor_indices) - 1 
        normal = np.cross(
                vertices[triangles[0][0]] - vertices[triangles[0][1]],
                vertices[triangles[0][0]] - vertices[triangles[0][2]])
        normal /= np.linalg.norm(normal) 
        curvature = np.dot(laplacian, normal)
        average_curvatures.append(curvature)
    avg_cur = np.mean((average_curvatures))
    avg_cur =(np.abs(avg_cur)+0.005)*10
    return avg_cur
 
if __name__ == "__main__":
    ply_file_path = "data/mesh/severe_AIS.ply" 
    average_curvatures = compute_average_curvature(ply_file_path) 
    print('Severe AIS case:')
    print(f'Asymmetric index: {average_curvatures}')
    
    ply_file_path = 'data/mesh/AIS.ply'
    average_curvatures = compute_average_curvature(ply_file_path) 
    print('AIS case:')
    print(f'Asymmetric index: {average_curvatures}')
    
    ply_file_path = 'data/mesh/normal.ply'
    average_curvatures = compute_average_curvature(ply_file_path) 
    print('Normal case:')
    print(f'Asymmetric index: {average_curvatures}')
