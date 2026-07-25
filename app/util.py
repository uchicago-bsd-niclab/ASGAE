#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tog
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from app.DataLoaders import MyData
import vtk
import SimpleITK as sitk
from os import path
import open3d as o3d
from scipy.spatial import cKDTree
import copy

# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
def array2polydata(points):
    """
    Convierte un array de puntos (N, 3) en un objeto vtkPolyData.

    Parámetros:
    - points: np.ndarray de tamaño (N, 3) con coordenadas (x, y, z).

    Retorna:
    - polydata: objeto vtkPolyData con los puntos.
    """
    # Verificar que la entrada sea un array numpy
    points = np.asarray(points)

    # Crear un vtkPoints y agregar los puntos
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p[0], p[1], p[2])

    # Crear un vtkPolyData y asignar los puntos
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    return polydata

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def euler2quat(e, order=3):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def ApplyTransformPC(pc, mask, rttn, trns, scl=None):
    
    outPC = torch.zeros(pc.shape).to(pc)
    for x in range(pc.shape[0]):
        mask_src=pc[x,mask[x]].to(pc)
        # outPC[x,mask[x]]=torch.matmul((mask_src-torch.mean(mask_src,axis=0)),rttn[x].transpose(1,0))+trns[x]+torch.mean(mask_src,axis=0)
        outPC[x,mask[x]]=torch.matmul((mask_src),rttn[x].transpose(1,0))+trns[x]
    return outPC

def npmat2euler(mats, seq='xyz'):
    eulers = []
    # for i in range(mats.shape[0]):
    for i in range(len(mats)):
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')
def error_rotmat_angles(mat_pred, mat_gt, seq='xyz'):
    # mat_diff = np.matmul(mat_pred, mat_gt.T)
    return npmat2euler(np.matmul(mat_pred, mat_gt.transpose((0,2,1))))

def error_euler_angles(mat_pred,eulers_gt, seq='xyz'):
    mat_diff = []
    for i in range(mat_pred.shape[0]):
        r_pred =  mat_pred[i]
        r_gt = Rotation.from_euler(seq,eulers_gt[i],degrees=True).as_matrix() 
        mat_diff.append(r_pred.dot(r_gt.T))

    return npmat2euler(mat_diff)

def fit_in_m1_to_1(points):
    '''
    Input: Nx3 
    Output: Nx3 
    fits the point cloud in [(-1,-1,-1) to (1,1,1)]
    '''
    points = points - np.mean(points,axis=0)
    dist_from_orig = np.linalg.norm(points,axis=1)
    points = points/np.max(dist_from_orig)
    return points

def get_transformations(igt):
	R_ba = igt[:, 0:3, 0:3]								# Ps = R_ba * Pt
	translation_ba = igt[:, 0:3, 3].unsqueeze(2)		# Ps = Pt + t_ba
	R_ab = R_ba.permute(0, 2, 1)						# Pt = R_ab * Ps
	translation_ab = -torch.bmm(R_ab, translation_ba)	# Pt = Ps + t_ab
	return R_ab, translation_ab, R_ba, translation_ba

def batched_pairwise_dist(a, b):
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P

def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    P = batched_pairwise_dist(a, b)
    return torch.mean(torch.min(P, 2)[0].float(), dim=-1), torch.mean(torch.min(P, 1)[0].float(), dim=-1), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def euler2rot(x):

    B = x.size(0)
    
    sinX = torch.sin(x[:,0])
    sinY = torch.sin(x[:,1])
    sinZ = torch.sin(x[:,2])

    cosX = torch.cos(x[:,0])
    cosY = torch.cos(x[:,1])
    cosZ = torch.cos(x[:,2])

    Rx = torch.zeros((B, 3, 3))
    Rx[:,0, 0] = 1.0
    Rx[:,1, 1] = cosX
    Rx[:,1, 2] = -sinX
    Rx[:,2, 1] = sinX
    Rx[:,2, 2] = cosX

    Ry = torch.zeros((B,3, 3))
    Ry[:,0, 0] = cosY
    Ry[:,0, 2] = sinY
    Ry[:,1, 1] = 1.0
    Ry[:,2, 0] = -sinY
    Ry[:,2, 2] = cosY

    Rz = torch.zeros((B,3, 3))
    Rz[:,0, 0] = cosZ
    Rz[:,0, 1] = -sinZ
    Rz[:,1, 0] = sinZ
    Rz[:,1, 1] = cosZ
    Rz[:,2, 2] = 1.0

    R = torch.matmul(torch.matmul(Rz, Ry), Rx)
    return R

def plotPCbatch(pcArray1, pcArray2, pcArray3, show = True, save = False, name=None, fig_count=4 , sizex = 5, sizey=10):
    
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    pc3 = pcArray3[0:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*3):

        ax = fig.add_subplot(3,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        elif i>=fig_count and i<fig_count*2:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='r', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc1[i-2*fig_count,:,0], pc1[i-2*fig_count,:,2], pc1[i-2*fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)
            ax.scatter(pc3[i-2*fig_count,:,0], pc3[i-2*fig_count,:,2], pc3[i-2*fig_count,:,1], c='r', marker='.', alpha=0.8, s=8)


        ax.set_xlim3d(-0.6, 0.6)
        ax.set_ylim3d(-0.6, 0.6)
        ax.set_zlim3d(-0.6, 0.6)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig
def ApplyTransform(data, transform):
    # Creating a copy of the input meshes
    a = vtk.vtkPolyData()
    a.DeepCopy(data)
    data = a

    for p in range(data.GetNumberOfPoints()):
        coords = np.array(data.GetPoint(p))
        newCoords = transform.TransformPoint(coords.astype(np.float64))
        data.GetPoints().SetPoint(p, newCoords[0], newCoords[1], newCoords[2])
    # Recalculating the normals and saving
    filter = vtk.vtkPolyDataNormals()
    filter.SetInputData(data)
    filter.ComputeCellNormalsOff()
    filter.ComputePointNormalsOff()
    filter.NonManifoldTraversalOff()
    filter.AutoOrientNormalsOn()
    filter.ConsistencyOn()
    filter.Update()
    data = filter.GetOutput()
    return data
def ReadPolyData(filename):
    if filename.endswith('.vtk'):
        reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def WritePolyData(data, filename):
    if filename.endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    # Saving landmarks
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Update()
    return

def ApplyTransform2Mesh(meshName,R, t, s=[]):
    basedirs = 'D:\OneDrive - The University of Colorado Denver\FinalModelingData'
    mesh = ReadPolyData(path.join(basedirs,str(meshName),'ExternalHeadSurface-updated.vtp'))
    rigid_euler = sitk.AffineTransform(3)
    rigid_euler.SetTranslation(t[:,0].T.astype(float))
    rigid_euler.SetMatrix(R.ravel().astype(float))
    if s:
        rigid_euler.SetScale(s)
    mesh = ApplyTransform(mesh, rigid_euler)

    WritePolyData(mesh, path.join('meshResults',meshName)+'.vtp')
    
def AddTransformGraph(dataset, angle = 0, translation = 1):
    rad = np.pi / 180 * angle
    R = euler2rot(torch.Tensor(np.random.uniform(-rad,rad,3*len(dataset))*np.pi).reshape(len(dataset),3))
    trns = torch.Tensor(np.random.uniform(-translation,translation,3*len(dataset)).reshape(len(dataset),3))
    Data = []
    for i in range(len(dataset)):
        pos2 = torch.matmul((dataset[i].pos-torch.mean(dataset[i].pos,axis=0)), R[i].transpose(1,0))+trns[i]+torch.mean(dataset[i].pos,axis=0)
        data = MyData(pos = torch.tensor(dataset[i].pos), edge_index = torch.tensor(dataset[i].edge_index), num_nodes = len(dataset[i].pos), R=R[i], trns = trns[i], GT=torch.tensor(pos2), batch = torch.zeros(len(dataset[i].pos), dtype = torch.int64))
        # dataR = MyData(x = x, pos = torch.tensor(pos2), edge_index = torch.tensor(edge_indices, dtype = torch.long), node_weight = node_weights, num_nodes = len(pos), R=R, trns = trns, scl= scl)
        Data.append(data)
    return Data

def applyTransformGraphPos(pos, batch, scale=True):
    # Apply a transformation to the data
    x, mask = tog.utils.to_dense_batch(pos, batch)
    rad = 1/9
    R = euler2rot(torch.Tensor(np.random.uniform(-rad,rad,3*x.size(0))*np.pi).reshape(x.size(0),3)).to(x)
    trns = torch.Tensor(np.random.uniform(-50,50,3*x.size(0)).reshape(x.size(0),3)).to(x)
    s = torch.Tensor(np.random.uniform(0.5,1.5,x.size(0)).reshape(x.size(0),1)).to(x)
    
    one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(R.shape[0], 1, 1).to(x)    # (Bx1x4)
    
    if scale:
        tmatrix = torch.cat([s.view(-1,1,1)*R, trns.unsqueeze(-1)], dim=2)                        # (Bx3x4)
        tmatrix = torch.cat([tmatrix, one_], dim=1)   
        # pos2 = torch.matmul((x-torch.mean(x,axis=1).unsqueeze(1)), R)+trns.unsqueeze(1)+torch.mean(x,axis=1).unsqueeze(1)
        pos2 = apply_transform(x, tmatrix)
        pos2[~mask] = 0.0
    else:
        tmatrix = torch.cat([R, trns.unsqueeze(-1)], dim=2)                        # (Bx3x4)
        tmatrix = torch.cat([tmatrix, one_], dim=1)   
        # pos2 = torch.matmul((x-torch.mean(x,axis=1).unsqueeze(1)), R)+torch.mean(x,axis=1).unsqueeze(1)
        pos2 = apply_transform(x, tmatrix)
        pos2[~mask] = 0.0
    pos2 = from_dense_batch(pos2, mask)[0]
    
      
    return pos2, tmatrix.to(torch.float32)

def inverse_transform(transform, scale=True):
    """
    Aplica la transformación inversa de una matriz homogénea 4x4 a una nube de puntos.

    Args:
        coord (torch.Tensor): [B, N, 3] Nube de puntos.
        transform (torch.Tensor): [B, 4, 4] Matriz de transformación homogénea.

    Returns:
        torch.Tensor: [B, N, 3] Nube de puntos transformada inversamente.
    """
    B = transform.shape[0]

    # Extraer rotación y traslación
    R_s = transform[:, :3, :3]  # [B, 3, 3]
    t = transform[:, :3, 3:]  # [B, 3, 1]

    # Calcular inversa
    if scale:
        scale = torch.linalg.norm(R_s, dim=1).mean(dim=1, keepdim=True)  # [B, 1]
        scale = scale.clamp(min=1e-8)  # evitar división por cero
        R = R_s / scale.view(B, 1, 1)
    else:
        scale = torch.ones(B, 1, device=transform.device, dtype=transform.dtype)
        R = R_s
    
    R_inv = R.transpose(1, 2)/scale.view(B,1,1)  # R^T
    t_inv = -torch.bmm(R_inv, t)  # -R^T * t

    # Construir matriz homogénea inversa
    inv_transform = torch.eye(4, device=transform.device, dtype=transform.dtype).unsqueeze(0).repeat(B, 1, 1)
    inv_transform[:, :3, :3] = R_inv
    inv_transform[:, :3, 3] = t_inv.squeeze(-1)

    # Aplicar inversa
    return inv_transform

def ApplyTransform2Mesh(meshName,R, t, s=[], args=None):
    basedirs = 'D:\OneDrive - The University of Colorado Denver\FinalModelingData'
    mesh = ReadPolyData(path.join(basedirs,str(meshName),'ExternalHeadSurface-updated.vtp'))
    rigid_euler = sitk.AffineTransform(3)
    rigid_euler.SetTranslation(t.T.astype(float))
    rigid_euler.SetMatrix(R.ravel().astype(float))
    if s:
        rigid_euler.SetScale(s)
    mesh = ApplyTransform(mesh, rigid_euler)
    if args:
        WritePolyData(mesh, path.join('meshResults',meshName)+str(args.model)+'.vtp')
    else:
        WritePolyData(mesh, path.join('meshResults',meshName)+'.vtp')

def from_dense_batch(dense_bath, mask):
    """
    Converts a dense batch of data and its corresponding mask into a compact representation.

    Args:
        dense_bath (torch.Tensor): A dense batch tensor of shape (B, N, F), where
            B is the batch size, N is the number of nodes per batch, and F is the feature dimension.
        mask (torch.Tensor): A boolean or binary mask tensor of shape (B, N) indicating valid nodes.

    Returns:
        data_x (torch.Tensor): A tensor containing only the valid (unmasked) node features,
            of shape (total_num_nodes, F), where total_num_nodes is the sum of valid nodes across the batch.
        data_batch (torch.Tensor): A tensor of shape (total_num_nodes,) indicating the batch index
            for each node in data_x.
    """
    # dense batch, B, N, F
    # mask, B, N
    B, N, F = dense_bath.size()
    flatten_dense_batch = dense_bath.reshape(B*N,F)
    flatten_mask = mask.view(-1)
    data_x = flatten_dense_batch[flatten_mask, :]
    num_nodes = torch.sum(mask, dim=1)  # B, like 3,4,3
    pr_value = torch.cumsum(num_nodes, dim=0)  # B, like 3,7,10
    indicator_vector = torch.zeros(torch.sum(num_nodes, dim=0))
    indicator_vector[pr_value[:-1]] = 1  # num_of_nodes, 0,0,0,1,0,0,0,1,0,0,1
    data_batch = torch.cumsum(indicator_vector, dim=0)  # num_of_nodes, 0,0,0,1,1,1,1,1,2,2,2
    return data_x, data_batch

def apply_transform(coord, transform):
    """
    Aplica una transformación rígida 4x4 a una nube de puntos.

    Args:
        coord (torch.Tensor): [B, N, 3] Nube de puntos (batch).
        transform (torch.Tensor): [B, 4, 4] Matriz de transformación homogénea por batch.

    Returns:
        torch.Tensor: [B, N, 3] Nube transformada.
    """
    B, N, _ = coord.shape
    coord_h = torch.cat([coord, torch.ones(B, N, 1, device=coord.device)], dim=2)  # [B, N, 4]
    coord_transf = torch.bmm(coord_h, transform.transpose(1, 2))  # [B, N, 4]
    return coord_transf[:, :, :3]  

def plot_PC(pc_trg, mask_trg, output=None, mask_src = None):
    if output!=None and mask_src != None:
        for i in range(len(pc_trg)):
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pc_trg[i,mask_trg[i]].detach().cpu().numpy())
            pcd1.paint_uniform_color([1, 0.5, 0])
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(output[i,mask_src[i]].detach().cpu().numpy())
            pcd2.paint_uniform_color([0, 0.5, 1])
            
            o3d.visualization.draw_geometries([pcd1, pcd2])
    else:
        for i in range(len(pc_trg)):
            create_pcd_obj(pc_trg[i,mask_trg[i]].detach().cpu().numpy(), [1, 0.5, 0])

def create_pcd_obj(np_array,col=[1,0,0]):
	'''
	input: nx3 array
	output: pcd object 
			can be displayed using o3d.visualization.draw_geometries([pcd1,pcd2])
	'''

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np_array[:,0:3])
	pcd.paint_uniform_color(col)
	return pcd

def transformPC2Mesh(PC, graph):
    # Creating a mesh from the pc input
    graph.edge_index = tog.utils.remove_self_loops(graph.edge_index)[0]
    points = vtk.vtkPoints()
    for i, (x, y, z) in enumerate(PC):
        points.InsertNextPoint(x, y, z)

    edges = graph.edge_index
    edge_dict = {k: [] for k in range(graph.num_nodes)}
    #build the whole edge list as a dict!
    for k,v in zip(edges[0],edges[1]):
        edge_dict[k.item()].append(v.item())
    # now sort through and generate the faces
    faces = []
    for k,v in edge_dict.items():
        for j in v:
            [faces.append((k, j, x)) for x in edge_dict[j] if k in edge_dict[x]]

    #now construct the cells and the points
    cellArray = vtk.vtkCellArray()
    for face in faces:
        cellArray.InsertNextCell(3)
        cellArray.InsertCellPoint(face[0])
        cellArray.InsertCellPoint(face[1])
        cellArray.InsertCellPoint(face[2])

    # Create a vtkPolyData object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cellArray)

    filter = vtk.vtkGeometryFilter()
    filter.SetInputData(polydata)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkTriangleFilter()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()

    filter = vtk.vtkCleanPolyData()
    filter.SetInputData(mesh)
    filter.Update()
    mesh = filter.GetOutput()
    return mesh

def GenerateMesh(data):
    
    if hasattr(data, 'pos') and hasattr(data, 'x') and data.x.size(1) > 3:
        data.edge_index = tog.utils.remove_self_loops(data.edge_index)[0]
        pos= data.pos
        edges = data.edge_index
        normals =  data.x[:,:3]
        textures = data.x[:,3:]/data.x[:,3:].max()#).type(torch.int)
        edge_dict = {k: [] for k in range(data.num_nodes)}
        #build the whole edge list as a dict!
        for k,v in zip(edges[0],edges[1]):
            edge_dict[k.item()].append(v.item())
        # now sort through and generate the faces
        faces = []
        for k,v in edge_dict.items():
            for j in v:
                [faces.append((k, j, x)) for x in edge_dict[j] if k in edge_dict[x]]

        #now construct the cells and the points
        cellArray = vtk.vtkCellArray()
        for face in faces:
            cellArray.InsertNextCell(3)
            cellArray.InsertCellPoint(face[0])
            cellArray.InsertCellPoint(face[1])
            cellArray.InsertCellPoint(face[2])

        if textures.size(1)>0:
            textureArray = vtk.vtkFloatArray()
            textureArray.SetName('Texture')
            textureArray.SetNumberOfComponents(3)
            for i in range(len(textures)):
                textureArray.InsertNextTuple3(textures[i,0],textures[i,1],textures[i,2])
        if normals.size(1)>0:
            norms= vtk.vtkFloatArray()
            norms.SetName('Normals')
            norms.SetNumberOfComponents(3)
            for i in range(normals.shape[0]):
                norms.InsertNextTuple3(normals[i,0], normals[i,1], normals[i,2])


        points = vtk.vtkPoints()
        for point in pos:
            points.InsertNextPoint(point[0], point[1], point[2])
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetPolys(cellArray)
        if textures.size(1)>0:
            polyData.GetPointData().AddArray(textureArray)
        if normals.size(1)>0:
            polyData.GetPointData().AddArray(norms)
            
    else:
        data.edge_index = tog.utils.remove_self_loops(data.edge_index)[0]
        pos= data.x
        edges = data.edge_index
        edge_dict = {k: [] for k in range(data.num_nodes)}
        #build the whole edge list as a dict!
        for k,v in zip(edges[0],edges[1]):
            edge_dict[k.item()].append(v.item())
        # now sort through and generate the faces
        faces = []
        for k,v in edge_dict.items():
            for j in v:
                [faces.append((k, j, x)) for x in edge_dict[j] if k in edge_dict[x]]

        #now construct the cells and the points
        cellArray = vtk.vtkCellArray()
        for face in faces:
            cellArray.InsertNextCell(3)
            cellArray.InsertCellPoint(face[0])
            cellArray.InsertCellPoint(face[1])
            cellArray.InsertCellPoint(face[2])

        points = vtk.vtkPoints()
        for point in pos:
            points.InsertNextPoint(point[0], point[1], point[2])
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetPolys(cellArray)

    return polyData

def ComputePointToSurfaceError(polydata, points):
    # Create a vtkImplicitPolyDataDistance object
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(polydata)
    
    distances = []
    for point in points:
        distance = implicit_distance.EvaluateFunction(point)
        distances.append(distance)
    
    return np.abs(distances)

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')
        self.mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
        self.std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std
        return data

def compute_pointwise_distances(mesh1, mesh2):
    points1 = mesh1.GetPoints()
    points2 = mesh2.GetPoints()
    distances = np.zeros(points1.GetNumberOfPoints())
    if points1.GetNumberOfPoints() == points2.GetNumberOfPoints():
        for i in range(points1.GetNumberOfPoints()):
            p1 = np.array(points1.GetPoint(i))
            p2 = np.array(points2.GetPoint(i))
            distances[i] = np.linalg.norm(p1 - p2)
    else:
        tree = cKDTree(np.array([mesh2.GetPoint(i) for i in range(mesh2.GetNumberOfPoints())]))
        for i in range(points1.GetNumberOfPoints()):
            p1 = np.array(points1.GetPoint(i))
            _, idx = tree.query(p1)
            p2 = np.array(points2.GetPoint(idx))
            distances[i] = np.linalg.norm(p1 - p2)

    return distances

def add_distance_scalar_to_mesh(mesh, distances, textureDist=None,name="TextureError"):
    vtk_distances = vtk.vtkFloatArray()
    vtk_distances.SetName("DistanceError")
    vtk_distances.SetNumberOfValues(len(distances))
    
    for i, d in enumerate(distances):
        vtk_distances.SetValue(i, d)

    mesh.GetPointData().AddArray(vtk_distances)
    
    if textureDist is not None:
        vtk_texture = vtk.vtkFloatArray()
        vtk_texture.SetName(name)
        # vtk_texture.SetNumberOfComponents(3)
        vtk_texture.SetNumberOfValues(len(textureDist))
        
        for i, d in enumerate(textureDist):
            vtk_texture.SetValue(i,d)
        mesh.GetPointData().AddArray(vtk_texture)
        
    return mesh

def project_texture_error(reference_polydata, predicted_polydata, texture_array_name="Texture"):
    # Asegúrate de que el array de textura está en la malla de referencia
    if reference_polydata.GetPointData().GetArray('Texture') is None:
        raise ValueError(f"Array '{texture_array_name}' not found in reference mesh.")

    # Crear el filtro de proyección
    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(reference_polydata)  # Malla con el valor que queremos proyectar
    probe.SetInputData(predicted_polydata)   # Malla donde queremos proyectar esos valores
    probe.Update()

    # Obtener los datos proyectados
    output = probe.GetOutput()
    projected_array = output.GetPointData().GetArray(texture_array_name)
    
    if projected_array is None:
        raise RuntimeError("No projection was made, possibly due to unmatched geometry.")

    # Convertir a NumPy para análisis si se desea
    import numpy as np
    projected_errors = np.array([projected_array.GetTuple1(i) for i in range(projected_array.GetNumberOfTuples())])

    return projected_errors  # Podrías también hacer np.mean(...) aquí si necesitas un resumen



def compute_chamfer_distances_mesh(mesh1, mesh2, textureDist=False):
    """
    Computes the symmetric Chamfer distance between mesh1 and mesh2.
    Returns:
        distances1: distances from mesh1 points to closest mesh2 points
        distances2: distances from mesh2 points to closest mesh1 points
    """
    points1 = np.array([mesh1.GetPoint(i) for i in range(mesh1.GetNumberOfPoints())])
    points2 = np.array([mesh2.GetPoint(i) for i in range(mesh2.GetNumberOfPoints())])

    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)

    d1, _ = tree2.query(points1)
    d2, _ = tree1.query(points2)
    dt1, dt2 = None, None
    
    if textureDist:
        # If texture distance is required, compute the texture error
        texture1 = mesh1.GetPointData().GetArray("Texture")
        texture2 = mesh2.GetPointData().GetArray("Texture")
        
        if texture1 is None or texture2 is None:
            raise ValueError("Texture arrays not found in one of the meshes.")
        
        texture1 = np.array([texture1.GetTuple(i) for i in range(texture1.GetNumberOfTuples())])
        texture2 = np.array([texture2.GetTuple(i) for i in range(texture2.GetNumberOfTuples())])
        
        tree11 = cKDTree(texture1)
        tree22 = cKDTree(texture2)

        dt1, _ = tree22.query(texture1)
        dt2, _ = tree11.query(texture2)

    return d1, d2, dt1, dt2

def batch_correlation(A, B, batch, eps=1e-8):
    """
    Calcula la correlación de Pearson entre dos tensores A y B de tamaño [bs, N, 512].

    Args:
        A (torch.Tensor): Tensor de tamaño [bs, N, 512]
        B (torch.Tensor): Tensor de tamaño [bs, N, 512]
        eps (float): Para evitar división por cero

    Returns:
        torch.Tensor: Correlación promedio por batch, tamaño [bs]
    """
    # Centrado de cada vector en la dimensión N
    A_mean = tog.utils.scatter(A, batch, dim=0, reduce='mean')[batch]  # [bs, 1, 512]
    B_mean = tog.utils.scatter(B, batch, dim=0, reduce='mean')[batch]  # [bs, 1, 512]
    
    A_centered = A - A_mean
    B_centered = B - B_mean

    # Numerador: suma de productos punto entre vectores centrados
    numerator = tog.utils.scatter((A_centered * B_centered), batch, dim=0, reduce='sum')  # [bs, 512]
    # numerator = tog.utils.scatter((A_centered * B_centered).sum(dim=1), batch, dim=0, reduce='sum')

    # Denominador: norma de cada vector (en dimensión N)
    A_norm = tog.utils.scatter(A_centered.pow(2), batch, dim=0, reduce='sum').sqrt()
    B_norm = tog.utils.scatter(B_centered.pow(2), batch, dim=0, reduce='sum').sqrt()

    denominator = (A_norm * B_norm) + eps  # evitar división por cero

    correlation = numerator / denominator  # [bs, 512]

    # Promedio sobre dimensión de características
    return 1-correlation.mean()


def point_to_surface_metric(target, source_points, fileName='Errors', saveData=True, texture=None, returnMesh=False):
        """
        Estimates the point-to-surface metric between transferred source and target meshes.
        Uses the transferred graphs from the transfer function.
        Returns the average minimum distance from each point in the transferred source to the surface of the transferred target.
        """
        
        target_copy = copy.deepcopy(target)
        
        # source_mesh = GenerateMesh(source_copy)
        target_mesh = GenerateMesh(target_copy)

        # Calculate the point-to-surface distance

            # distSrc = ComputePointToSurfaceError(source_mesh, source_points)
        distTrg = ComputePointToSurfaceError(target_mesh, source_points)
        if texture is not None:
            mesh_with_DisTex1 = add_distance_scalar_to_mesh(target_mesh, distTrg, texture, name='TextureError')
            WritePolyData(mesh_with_DisTex1, fileName+'_distances_source_mesh.vtp')
            
        if saveData:
            WritePolyData(target_mesh, fileName+'_src_mesh.vtp')
        
        if returnMesh:
            return np.mean(distTrg), target_mesh
        else:
            return np.mean(distTrg)
    
def metrics_for_ASGAE(graph_GT, estimated_points, texture=None):
    """
    Computes metrics for ASGAE model evaluation.
    Args:
        graph (torch_geometric.data.Data): Input graph with ground truth points in graph.y.
        estimated_points (torch.Tensor): Estimated points of shape [N, 3].
    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, and Chamfer distance.
    """
    if 'pos' in graph_GT.keys():
        gt_points = graph_GT.pos.cpu().numpy()
        gt_texture = graph_GT.x[:,3:].cpu().numpy()
    else:
        gt_points = graph_GT.x.cpu().numpy()
    est_points = estimated_points.cpu().numpy()

    # Mean Squared Error
    if gt_points.shape[0] == estimated_points.shape[0]:
        distances = np.mean(np.linalg.norm(gt_points - est_points, axis=1))

        # Mean Absolute Error
        mae = np.mean(np.abs(gt_points - est_points))
    else:
        distances = 0
        mae = 0

    # Chamfer Distance
    tree_gt = cKDTree(gt_points)
    tree_est = cKDTree(est_points)

    d1, _ = tree_est.query(gt_points)
    d2, _ = tree_gt.query(est_points)

    chamfer_dist = np.mean(d1**2) + np.mean(d2**2)
    
    p2s,MeshGT = point_to_surface_metric(graph_GT, est_points, saveData=False, returnMesh=True)
    if texture is not None:
        p2s_texture = ComputePointToSurfaceTextureError(MeshGT, est_points, texture.cpu().numpy(), metric="L1")
        distances_Texture = np.mean(np.linalg.norm(gt_texture - texture.cpu().numpy(), axis=1))
        # p2s = np.mean(p2s_texture)
        return {
            'Distance': distances,
            'Distance_Texture': distances_Texture,
            'MAE': mae,
            'Chamfer': chamfer_dist,
            'P2S':p2s,
            'P2S_Texture': p2s_texture['summary']['MSE']
        }
    else:
        return {
            'Distance': distances,
            'MAE': mae,
            'Chamfer': chamfer_dist,
            'P2S':p2s
        }

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def barycentric_weights(p, a, b, c):
    """Calcula pesos baricéntricos de un punto p respecto a triángulo (a,b,c)."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        # Triángulo degenerado: asigna todo el peso al vértice más cercano
        dists = [np.linalg.norm(p - a), np.linalg.norm(p - b), np.linalg.norm(p - c)]
        w = np.zeros(3)
        w[np.argmin(dists)] = 1.0
        return w
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

def ComputePointToSurfaceTextureError(mesh, query_points, est_textures, metric="RMSE"):
    """
    Calcula el error entre texturas predichas (est_textures) y la textura
    almacenada en el array 'Texture' de PointData del mesh.

    mesh          : vtkPolyData con PointData["Texture"]
    query_points  : (N,3) ndarray
    est_textures  : (N,) o (N,C) ndarray (predicciones)
    metric        : 'L1' | 'L2' | 'MAE' | 'RMSE'
    """
    query_points = np.asarray(query_points, dtype=float)

    # Aseguramos que la malla esté triangulada
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(mesh)
    tri.Update()
    mesh = tri.GetOutput()

    # Extraemos la textura de los vértices
    tex_vtk = mesh.GetPointData().GetArray("Texture")
    if tex_vtk is None:
        raise ValueError("El mesh no contiene un array 'Texture' en PointData.")
    tex = vtk_to_numpy(tex_vtk).astype(float)
    ncomp = tex_vtk.GetNumberOfComponents()

    # Locator para búsqueda de punto más cercano
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    vtk_points = mesh.GetPoints()
    N = query_points.shape[0]
    surf_tex = np.zeros((N, ncomp))

    cp = [0.0, 0.0, 0.0]
    cid = vtk.reference(0)
    sid = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i, p in enumerate(query_points):
        locator.FindClosestPoint(p, cp, cid, sid, dist2)
        cell = mesh.GetCell(int(cid))
        ids = cell.GetPointIds()

        # Coordenadas de los vértices
        v0 = np.array(vtk_points.GetPoint(ids.GetId(0)))
        v1 = np.array(vtk_points.GetPoint(ids.GetId(1)))
        v2 = np.array(vtk_points.GetPoint(ids.GetId(2)))

        # Pesos baricéntricos
        w = barycentric_weights(np.array(cp), v0, v1, v2)

        # Interpolamos la textura
        t0 = tex[ids.GetId(0)]
        t1 = tex[ids.GetId(1)]
        t2 = tex[ids.GetId(2)]
        surf_tex[i, :] = w[0]*t0 + w[1]*t1 + w[2]*t2

    # Normalizamos est_textures a 2D
    est = np.asarray(est_textures, dtype=float)
    if est.ndim == 1 and ncomp == 1:
        est = est[:, None]
    elif est.ndim == 1 and ncomp > 1:
        raise ValueError(f"Las predicciones son escalares pero la textura de la superficie tiene {ncomp} componentes.")
    elif est.ndim == 2 and est.shape[1] != ncomp:
        raise ValueError(f"Número de componentes incompatible: {est.shape[1]} vs {ncomp}.")

    diff = est - surf_tex
    if ncomp == 1:
        per_point_L1 = np.abs(diff[:,0])
        per_point_L2 = np.sqrt(diff[:,0]**2)
    else:
        per_point_L1 = np.linalg.norm(diff, ord=1, axis=1)
        per_point_L2 = np.linalg.norm(diff, ord=2, axis=1)

    # Selección de métrica
    if metric.upper() in ("L1", "MAE"):
        per_point_error = per_point_L1
    elif metric.upper() in ("L2", "MSE"):
        per_point_error = per_point_L2
    else:
        raise ValueError("Métrica no soportada.")

    mae = float(np.mean(per_point_L1))
    mse = float(np.mean(per_point_L2))
    summary = {"MAE": mae, "MSE": mse}

    return {
        "per_point_error": per_point_error,
        "summary": summary,
        "surf_values": surf_tex
    }


import os
import json
import numpy as np
import csv
from datetime import datetime

def save_raw_metric_lists(args, out_prefix="metrics", keys=None):
    """
    Guarda las listas crudas de args en:
      - {out_prefix}.npz  (dict de arrays)
      - {out_prefix}.json (dict de listas)
      - {out_prefix}_long.csv (dos columnas: metric,value)

    Soporta longitudes distintas entre listas.
    """
    # Define aquí las claves que quieres extraer
    # keys = [
    #     "mse_coord_epoch",
    #     "mae_coord_epoch",
    #     "mse_color_epoch",
    #     "mae_color_epoch",
    #     "distP2SAB",
    #     "distP2SBA",
    #     "distP2STexAB",
    #     "distP2STexBA",
    # ]

    # Recolecta listas desde args (si falta alguna, la ignora con aviso)
    data = {}
    for k in keys:
        if hasattr(args, k):
            arr = np.asarray(getattr(args, k))
            # Fuerza a 1D si es posible (por si viniera con shape (N,1))
            if arr.ndim > 1 and arr.shape[1] == 1:
                arr = arr.ravel()
            data[k] = arr.astype(float)  # uniformiza a float
        else:
            print(f"[warn] args no tiene el atributo '{k}', se omitirá.")

    if not data:
        raise ValueError("No se encontró ninguna de las listas especificadas en 'args'.")

    # 1) Guardar NPZ (sin pérdida, recomendado para NumPy)
    npz_path = f"{out_prefix}.npz"
    np.savez_compressed(npz_path, **data)

    # 2) Guardar JSON (legible)
    json_path = f"{out_prefix}.json"
    with open(json_path, "w") as f:
        json.dump({k: v.tolist() for k, v in data.items()}, f, indent=2)

    # 3) Guardar CSV en formato 'long' (metric,value)
    csv_path = f"{out_prefix}_long.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in data.items():
            for val in v:
                writer.writerow([k, float(val)])

    # (Opcional) Log cortito
    total_values = sum(len(v) for v in data.values())
    print(f"Guardado:")
    print(f" - NPZ : {npz_path}")
    print(f" - JSON: {json_path}")
    print(f" - CSV : {csv_path}  (total filas={total_values})")

# ==== Ejemplo de uso ====
# save_raw_metric_lists(args, out_prefix="exp42_metrics")
def SaveLandmarks(landmarks, path, returnPoints=False):
    # Crear estructura de puntos
    landmarksPoints = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for p in range(landmarks.shape[0]):
        points.InsertNextPoint(landmarks[p, :3])
    landmarksPoints.SetPoints(points)

    # Si hay textura, agregarla antes de escribir
    if landmarks.shape[1] == 6:
        textures = landmarks[:, 3:6]
        textureArray = vtk.vtkFloatArray()
        textureArray.SetName('Texture')
        textureArray.SetNumberOfComponents(3)
        for i in range(len(textures)):
            textureArray.InsertNextTuple3(*textures[i])
        landmarksPoints.GetPointData().AddArray(textureArray)

    # Escribir archivo
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(landmarksPoints)
    writer.Write()  # Usar Write() en lugar de Update()

    if returnPoints:
        return landmarksPoints
    

def ensure_triangles(poly):
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    return tri.GetOutput()

def project_points_to_surface_closest(polydata_surface, points_np):
    """
    polydata_surface: vtkPolyData (superficie)
    points_np: np.ndarray de shape (N, 3) con puntos a proyectar
    return: np.ndarray (N, 3) con puntos proyectados en la superficie
    """
    surface = ensure_triangles(polydata_surface)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    projected = np.empty_like(points_np, dtype=float)

    # Buffers para la búsqueda
    closest = [0.0, 0.0, 0.0]
    cellId = vtk.reference(0)
    subId = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i, p in enumerate(points_np):
        locator.FindClosestPoint(p, closest, cellId, subId, dist2)
        projected[i, :] = closest

    return projected
