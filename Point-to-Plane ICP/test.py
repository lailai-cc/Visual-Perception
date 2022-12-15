import numpy as np
import time
import icp
from plyfile import *
import open3d as o3d
import copy

def read_ply_xyz(filename):
    with open(filename,'rb') as f:
        plydata = PlyData.read(f)
        n = plydata['vertex'].count
        vertics = np.zeros(shape=[n,3],dtype=np.float32)
        vertics[:, 0]=plydata['vertex'].data['x']
        vertics[:, 1] = plydata['vertex'].data['y']
        vertics[:, 2] = plydata['vertex'].data['z']
    return n,vertics

def pre_handle(filename):
    pcd = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pcd], width=800, height=600)  # 点云可视化

    # 体素下采样
    processed_data = o3d.geometry.voxel_down_sample(pcd,5)
    #o3d.visualization.draw_geometries([processed_data], width=800, height=600)  # 点云可视化

    # outlier removal
    processed_data, outlier_index = o3d.geometry.radius_outlier_removal( processed_data ,nb_points=10,radius=10)
    #估计法向量
    o3d.geometry.estimate_normals( processed_data , search_param=o3d.geometry.KDTreeSearchParamKNN(20))
    o3d.visualization.draw_geometries([processed_data ],width=800,height=600) #点云可视化

    name=filename[:-4] #提取除了.ply的文件路径
    newname=name+'_pre.ply'
    o3d.io.write_point_cloud(newname,processed_data)   #保存为ply文件
    return newname

def visualize(p1,p2):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p1)
    pcd.paint_uniform_color([1, 0.706, 0])  # source 为黄色
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(p2)
    pcd2.paint_uniform_color([0, 0.651, 0.929])  # target 为蓝色
    o3d.visualization.draw_geometries([pcd,pcd2],width=800,height=600) #点云可视化
    #o3d.io.write_point_cloud(filename,pcd)   #保存为ply文件

def transform(p):
    cloud= o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p)
    # 平移
    translation= copy.deepcopy(cloud).translate((20,-10, 5))
    #translation = copy.deepcopy(cloud).translate((0.02, 0.03, 0.1))
    #旋转
    translation.rotate(np.array([[0.1],[0.1],[1]]))
    B =  np.asarray(translation.points)
    return B

# Constants
num_tests = 1                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .5                            # max translation of the test set
rotation = .5                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def test_best_fit(filename=None):

    # Generate a random dataset
    if(filename==None):
        N=10                                # number of random points in the dataset
        A = np.random.rand(N, dim)
    else:
        N,A = read_ply_xyz(filename) #读入点云数据
    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation #生成三个维度的随机偏移量
        B += t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * noise_sigma
        visualize(A, B)
        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform_point(B, A)
        total_time += time.time() - start


        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        # Transform C
        C = np.dot(T, C.T).T
        visualize(A, C[:,0:3], )

        #np.allclose()是比较两个array是不是每一个元素都相等，如果不满足或超出阈值就报错
        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma) # T should transform B (or C) to A
        assert np.allclose(-t1, t, atol=6*noise_sigma)      # t and t1 should be inverses
        assert np.allclose(R1.T, R, atol=6*noise_sigma)     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return


def test_icp(filename=None,random=True):

    # 如果filename为None，就随机生成点云集
    if (filename == None):
        N = 10  # number of random points in the dataset
        A = np.random.rand(N, dim)
    else:#如果不为None,则读取点云文件
        N,A = read_ply_xyz(filename)  # 读入点云数据

    total_time_point = 0
    total_time_plane = 0

    for i in range(num_tests):
        if(random):
            B = np.copy(A)

            # Translate
            t = np.random.rand(dim) * translation
            B += t

            # Rotate
            R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
            B = np.dot(R, B.T).T

            # Add noise
            B += np.random.randn(N, dim) * noise_sigma
        else:
            B=transform(A)
        # Shuffle to disrupt correspondence
        np.random.shuffle(B)

       #Point to Point
        visualize(A,B)
        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp_point(B, A, tolerance=0.000001)
        total_time_point += time.time() - start
        print('icp point-ti-point distences:',np.mean(distances))

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)
        print(T)
        # Transform C
        C = np.dot(T, C.T).T

        visualize(A, C[:,0:3])  #可视化配准结果

        # assert np.mean(distances) < 6*noise_sigma                   # mean error should be small
        # assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)     # T and R should be inverses
        # assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)        # T and t should be inverses

        # Point to Plane
        visualize(A, B)
        # Run ICP
        start = time.time()
        T, distances, iterations = icp.icp_plane(B, A, tolerance=0.000001)
        total_time_plane += time.time() - start
        print('icp point-ti-plane distences:',np.mean(distances))
        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:, 0:3] = np.copy(B)
        print(T)
        # Transform C
        C = np.dot(T, C.T).T

        visualize(A, C[:, 0:3])  #可视化配准结果

        # assert np.mean(distances) < 6 * noise_sigma  # mean error should be small
        # assert np.allclose(T[0:3, 0:3].T, R, atol=6 * noise_sigma)  # T and R should be inverses
        # assert np.allclose(-T[0:3, 3], t, atol=6 * noise_sigma)  # T and t should be inverses

    print('icp point-ti-point time: {:.3}'.format(total_time_point/num_tests))
    print('icp point-ti-plane time: {:.3}'.format(total_time_plane / num_tests))
    return


if __name__ == "__main__":
    newname=pre_handle('../Mug.ply')
    test_icp(newname,False)