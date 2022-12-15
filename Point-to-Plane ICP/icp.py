import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from test import visualize

def best_fit_transform_point(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    # 找到A和B的质心centroid_A centroid_B，再由每个点减去中心值进行归一化
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    #SVD分解
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0: #计算行列式
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp_point(A, B, init_pose=None, max_iterations=30, tolerance=0.000001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform_point(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform_point(A, src[:m,:].T)
    print("iter:",i)
    return T, distances, i


def best_fit_transform_plane(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding source points
      B: Nxm numpy array of corresponding target points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[0]

    #获取法向量
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(B)
    o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(10))
    target_normal = np.asarray(pcd.normals)

    #构造A,b矩阵
    b_vector = np.zeros((m,1))
    A_vector = np.ones((m, 6))

    for i in range(m):
        A_vector[i][0] = target_normal [i][2]*A[i][1] - target_normal [i][1]*A[i][2]
        A_vector[i][1] = target_normal [i][0]*A[i][2] - target_normal [i][2]*A[i][0]
        A_vector[i][2] = target_normal [i][1]*A[i][0] - target_normal [i][0]*A[i][1]
        A_vector[i][3] = target_normal [i][0]
        A_vector[i][4] = target_normal [i][1]
        A_vector[i][5] = target_normal [i][2]

        diff = B[i] - A[i]  #Qi-Pi
        b_vector[i]=np.dot(diff.T,target_normal[i])

    #解出x
    x = np.linalg.solve(np.dot(A_vector.T, A_vector), np.dot(A_vector.T, b_vector))
    #print(x)
    t=np.zeros(3)
    t[0]=x[3]
    t[1]=x[4]
    t[2]=x[5]

    R = np.ones((3,3))
    R[0,1] = -x[2]
    R[0,2] = x[1]
    R[1,0] = x[2]
    R[1,2] = -x[0]
    R[2,0] = -x[1]
    R[2,1] = x[0]
    #构造T矩阵
    T=np.zeros((4,4))
    T[:3,:3] = np.copy(R)
    T[:3,3] = np.copy(t)
    T[3,3]=1

    return T, R, t

def icp_plane(A, B, init_pose=None, max_iterations=30, tolerance=0.000001):
    '''
    The Iterative Closest Point-to-Plane method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform_plane(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            print(np.abs(prev_error - mean_error))
            break
        prev_error = mean_error
    print('iter:',i)
    # calculate final transformation
    T,_,_ = best_fit_transform_plane(A, src[:m,:].T)
    #visualize(A,src[:m,:].T)

    return T, distances, i
