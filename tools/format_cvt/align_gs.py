from re import L
import torch
from pytorch3d.ops import knn_points
from tqdm import tqdm
# from LHM import LHM
# from LHM import LHM
from cvt_LHM_mesh_output import save_points_to_ply,save_ply
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

import math
import numpy as np

def align_pointclouds_knn(source_points, target_points, with_scale=True, K=1):
    """
    Aligns source to target by first matching each source point to its nearest target neighbor.
    source_points: [N1, 3]
    target_points: [N2, 3]
    """
    knn = knn_points(source_points.unsqueeze(0), target_points.unsqueeze(0), K=K)
    matched_targets = target_points[knn.idx[0, :, 0]]  # shape: [N1, 3]
    aligned, t = optimize_translation_only(source_points, matched_targets, num_iters=100, lr=1e-2, verbose=True)
    return aligned, t



def rigid_align_umeyama(source, target, with_scale=True):
    """
    Computes the rigid transformation that aligns source to target.
    source: [N, 3] point cloud
    target: [N, 3] point cloud
    Returns:
        aligned_source: [N, 3]
        R: [3, 3] rotation matrix
        t: [3,] translation
        s: float (scale)
    """
    # angle_x = math.pi
    # axis_x = torch.tensor([1.0, 0.0, 0.0])
    # rotvec_x = angle_x * axis_x  # (3,)

    # 转换为旋转矩阵
    R = torch.eye(3)
    # R[2,2]=-1.0 # (1, 3, 3)
    
    src_mean = source.mean(dim=0, keepdim=True)
    tgt_mean = target.mean(dim=0, keepdim=True)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    # 计算协方差矩阵
    cov = src_centered.T @ tgt_centered / source.shape[0]
    U, S, Vt = torch.linalg.svd(cov)
    # R = Vt.T @ U.T
    

    # 保证右手系（det(R)=1）
    # if torch.det(R) < 0:
    #     Vt[-1, :] *= -1
    #     R = Vt.T @ U.T

    if with_scale:
        var_src = (src_centered ** 2).sum() / source.shape[0]
        scale = (S.sum()) / var_src
    else:
        scale = 1.0

    t = tgt_mean[0] - scale * R @ src_mean[0]

    aligned = scale * (R @ source.T).T + t
    return aligned, R, t, scale

from pytorch3d.loss import chamfer_distance

def optimize_translation_only(source_points, target_points, num_iters=100, lr=1e-2, verbose=False):
    """
    Only optimize a translation T to align source_points to target_points.
    source_points: [N, 3]
    target_points: [M, 3]
    Returns:
        best_T: [3,]
        aligned_source: source_points + best_T
    """
    source_points = source_points.clone().detach().cuda()
    target_points = target_points.clone().detach().cuda()
    
    T = torch.zeros(1, 3, requires_grad=True, device=source_points.device)

    optimizer = torch.optim.Adam([T], lr=lr)

    for i in range(num_iters):
        optimizer.zero_grad()
        moved_source = source_points + T  # (N, 3)
        loss, _ = chamfer_distance(moved_source.unsqueeze(0), target_points.unsqueeze(0))
        loss.backward()
        optimizer.step()
        if verbose and i % 10 == 0:
            print(f"[Iter {i}] Loss: {loss.item():.6f}")

    best_T = T.detach().squeeze()
    aligned_source = source_points + best_T
    return aligned_source.detach().squeeze().cpu(), best_T.detach().squeeze().cpu()

#load source point cloud
angle_x = math.pi
axis_x = torch.tensor([1.0, 0.0, 0.0])
rotvec_x = angle_x * axis_x  # (3,)
R = axis_angle_to_matrix(rotvec_x).squeeze(0) # (3, 3)
print("R:", R)
Exavt_data= torch.load('/home/wenbo/ExAvatar/fitting/data/Custom/data/Adrian_1/LHM_file/full_neutral_pose.pth')
# "position","full_body_pose","betas"
Exavt_positions = Exavt_data['position'].cpu().squeeze() # [N, 3]

#load target point cloud
LHM_data = torch.load('/data1/users/wenbo/LHM/exps/Gaussians/video_human_benchmark/human-lrm-1B/Adrian_1_014_LHM_w_Exavt_neutral_pose_gs.pth')

LHM_pose = LHM_data['offset_xyz'].cpu().squeeze() # [N, 3]
LHM_pose = torch.matmul(LHM_pose, R.T) + torch.tensor([0,-0.25,5])# [N, 3]
all_points = torch.cat([Exavt_positions, LHM_pose], dim=0) # [N, 3]

#create different color for Exavt and LHM, save as single np array
Exavt_color = torch.zeros_like(Exavt_positions) # [N, 3]
Exavt_color[:, 0] = 255
LHM_color = torch.zeros_like(LHM_pose) # [N, 3]
LHM_color[:, 1] = 255
all_color = torch.cat([Exavt_color, LHM_color], dim=0) # [N, 3] 
#save as ply
save_points_to_ply(all_points.numpy(), '/home/wenbo/ExAvatar/fitting/data/Custom/data/Adrian_1/LHM_file/Exavt_neutral_pose&raw_LHM_result.ply')
aligned, t=align_pointclouds_knn(LHM_pose,Exavt_positions, with_scale=False)
# aligned = aligned+t

all_points = torch.cat([aligned, Exavt_positions], dim=0) # [N, 3]
#save as ply
save_points_to_ply(aligned.numpy(), '/home/wenbo/ExAvatar/fitting/data/Custom/data/Adrian_1/LHM_file/aligned_LHM_result.ply', 
                   color=LHM_data["shs"].cpu().numpy().squeeze()*255)
save_points_to_ply(all_points.numpy(), '/home/wenbo/ExAvatar/fitting/data/Custom/data/Adrian_1/LHM_file/Exavt_neutral_pose&aligned_LHM_result.ply',all_color)
print("aligned shape:", aligned.shape)
print("R:", R)
print("t:", t)

knn = knn_points(Exavt_positions.unsqueeze(0), aligned.unsqueeze(0), K=1)
Exavt_matched_list = knn.idx[0, :, 0]
Exavt_matched_list = Exavt_matched_list.cpu()
print("Exavt_matched_list:", Exavt_matched_list.shape)
# write back to LHM
LHM_data['position'] = aligned.cpu()
LHM_data['offset_xyz'] = aligned.cpu()
LHM_data["Exavt_matched_list"]= Exavt_matched_list
# LHM_data["rotation"] = LHM_data["rotation"]@R
LHM_data["s2t_rotation"] = R
print("LHM_data.keys():", LHM_data.keys())
torch.save(LHM_data, '/home/wenbo/ExAvatar/fitting/data/Custom/data/Adrian_1/LHM_file/Final_aligned_LHM_result_Exavt_neutral_pose_gs.pth')
# print("scale:", scale)
