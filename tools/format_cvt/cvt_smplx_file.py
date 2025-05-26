from email.mime import image
from math import comb
import os
import os.path as osp
import json
import re
import cv2
from glob import glob
import numpy as np
import argparse
import pickle as pkl
import smplx
from sympy import im
from tqdm import tqdm
import torch
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle

# from engine.pose_estimation.pose_utils import camera


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--output_path', type=str, dest='output_path')
    # parser.add_argument('--focal', type=float, default=2000)
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."

    return args

def load_gt_camera_from_metadata(gt_cam_data_path):
    if "metadata" not in gt_cam_data_path:
        gt_cam_data_path = osp.join(gt_cam_data_path, "metadata")
    # assert osp.exists(gt_cam_data_path), "gt_cam_data_path does not exist."
    if not osp.exists(gt_cam_data_path):
        return None,None
    print("Loading camera data from metadata...")
        
    #FOR LOADING CAMERA DATA SPECIFIED FOR RECORD3D APP
    with open(gt_cam_data_path, "r") as f:
        metadata = json.load(f)
    gt_camera_intrinsic = np.eye(3)
    #check if perFrameIntrinsicCoeffs key exists
    # if "perFrameIntrinsicCoeffs" in metadata:
    #    #gt_camera_intrinsic[0,0]= np.array(metadata["perFrameIntrinsicCoeffs"])[0][0]
    #     gt_camera_intrinsic[1,1]= np.array(metadata["perFrameIntrinsicCoeffs"])[0][1]
    #     gt_camera_intrinsic[0,2]= np.array(metadata["perFrameIntrinsicCoeffs"])[0][2]
    #     gt_camera_intrinsic[1,2]= np.array(metadata["perFrameIntrinsicCoeffs"])[0][3]
    # elif "K" in metadata:
    gt_camera_intrinsic = np.array(metadata["K"]).reshape(3,3).T
    image_size_wh = np.array([metadata["w"],metadata["h"]])
    print(gt_camera_intrinsic)
    return gt_camera_intrinsic,image_size_wh

def make_virtual_cam(root_path,save_json=False,focal=2000):
    print('Making virtual camera parameters..., focal:',focal)
    img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
    img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]
    if save_json:
        save_root_path = osp.join(root_path, 'cam_params')
        os.makedirs(save_root_path, exist_ok=True)
        frame_idx_list = [int(x.split('/')[-1][:-4]) for x in img_path_list]
        for frame_idx in frame_idx_list:
            with open(osp.join(save_root_path, str(frame_idx) + '.json'), 'w') as f:
                json.dump({'R': np.eye(3).astype(np.float32).tolist(), 't': np.zeros((3), dtype=np.float32).tolist(), 'focal': (focal,focal), 'princpt': (img_width/2, img_height/2)}, f)
    else:
        virtual_cam = {'R': np.eye(3).astype(np.float32), 't': np.zeros((3), dtype=np.float32), 'focal': (focal,focal), 'princpt': (img_width/2, img_height/2)}
        return virtual_cam

"""this is just a rough code for combining smpl and mano, not used in the final code
    the expr params in the smoothed smplx is still missing, and we have to use the one obtained from the original smoothing(fitting) code
"""
def combine_smpl_mano(smpl_path,mano_path,smoothed_smplx_path,output_path=None):
    with open(smpl_path,'rb') as f:
        smpl = pkl.load(f)
    with open(mano_path,'rb') as f:
        mano = pkl.load(f)
    with open(smoothed_smplx_path,'r') as f:
        smoothed_smplx = json.load(f)
    
    #create a new dict to store the combined params
    
    combined_params = {}
    key_list = ["root_pose","body_pose","jaw_pose","leye_pose","reye_pose","lhand_pose","rhand_pose","expr","trans"]
    combined_params["root_pose"] = matrix_to_axis_angle(smpl["smpl_params"]["global_orient"]).detach().cpu().squeeze().numpy()
    assert combined_params["root_pose"].shape == (3,)
    combined_params["body_pose"] = matrix_to_axis_angle(smpl["smpl_params"]["body_pose"][:,:-2,:,:]).detach().cpu().squeeze().numpy()
    assert combined_params["body_pose"].shape == (21,3)
    combined_params["trans"] = smpl["cam_t"].detach().cpu().squeeze().numpy()
    assert combined_params["trans"].shape == (3,)
    combined_params["jaw_pose"] = smoothed_smplx["jaw_pose"]
    combined_params["leye_pose"] = smoothed_smplx["leye_pose"]
    combined_params["reye_pose"] = smoothed_smplx["reye_pose"]
    combined_params["expr"] = smoothed_smplx["expr"]
    hand_lr = mano["right"] # 0 for left, 1 for right
    combined_params["lhand_pose"] = np.zeros((15,3))
    combined_params["rhand_pose"] = np.zeros((15,3))
    for i in range(len(hand_lr)):
        if hand_lr[i] == 0:
            combined_params["lhand_pose"] = matrix_to_axis_angle(mano["pred_mano_params"][i]["hand_pose"]).detach().cpu().squeeze().numpy()
            assert combined_params["lhand_pose"].shape == (15,3)
        elif hand_lr[i] == 1:
            combined_params["rhand_pose"] = matrix_to_axis_angle(mano["pred_mano_params"][i]["hand_pose"]).detach().cpu().squeeze().numpy()
            assert combined_params["rhand_pose"].shape == (15,3)
    if len(hand_lr) != 2:
        print("mano_path",mano_path)

    # combined_params["lhand_pose"] = mano["pred_mano_params"]
    # combined_params["rhand_pose"] = mano["pred_mano_params"]
    for key in combined_params.keys():
        # print(key,type(combined_params[key]))
        if type(combined_params[key]) == list:
            pass
        else:
            combined_params[key] = combined_params[key].tolist()
    #write to new json file
    with open(output_path,'w') as f:
        json.dump(combined_params,f)


def format_smplx_file(root_path,output_path):
    #use GT focal length
    K,image_size_wh= load_gt_camera_from_metadata(osp.join(root_path, 'metadata'))
    if K is None:
        #specified for cxk short video
        print("GT camera data not found, using predefined camera parameters...")
        K = np.array([[1432.3338623046875,0,720],[0,1432.3338623046875,960],[0,0,1]])
        image_size_wh = [720,960]
    focal= K[0,0]
    print('focal:',focal)
    os.makedirs(osp.join(output_path), exist_ok=True)
    #load smplx_smooth results and camera_hmr results
    camera_hmr_path = "/data1/users/wenbo/CameraHMR/CameraHMR_demo/demo_output/Adrian_1/smpl_init" #%4d.pkl 0001
    smplx_smooth_path = "/data1/users/wenbo/ExAvatar_RELEASE/avatar/data/Custom/data/Adrian_1/smplx_optimized/smplx_params_smoothed" #%d.json 0 

    target_json_keys=['betas', 'root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'trans', 'focal', 'princpt', 'img_size_wh', 'pad_ratio']
    # betas (10,)root_pose (3,) body_pose (21, 3) jaw_pose (3,)leye_pose (3,)reye_pose (3,)lhand_pose (15, 3) rhand_pose (15, 3)trans (3,)focal (2,) princpt (2,) img_size_wh (2,) pad_ratio () 0.2
    # camhmr_hamba_smplx_params:[ 'root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'trans',"expr"(50)]
    # smpl_init :['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'trans',"shape"(10),"expr"(10)]
    # smplx_smooth: ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'trans',"expr"(50)]
    # camera_hmr: vertices','smpl_params','cam',"cam_t",'focal_length';
    #                       smpl_params:'global_orient', 'body_pose', 'betas'

    #load camera_hmr results
    camera_hmr_files = glob(osp.join(camera_hmr_path, '*.pkl'))
    # print("camera_hmr_files:",camera_hmr_files)
    camera_hmr_files=sorted(camera_hmr_files,key=lambda x:int(x.split('/')[-1][:-4]))
    total_camhmr_file_num = len(camera_hmr_files)
    smplx_smooth_files = glob(osp.join(smplx_smooth_path, '*.json'))
    smplx_smooth_files=sorted(smplx_smooth_files,key=lambda x:int(x.split('/')[-1][:-5]))
    print("total total_camhmr_file_num:",total_camhmr_file_num," total_smplx_smooth_file_num:",len(smplx_smooth_files))
    # assert len(camera_hmr_files) == len(smplx_smooth_files)
    for i in tqdm(range(1,total_camhmr_file_num)):
        camera_hmr_file_path = os.path.join(camera_hmr_path, "%d.pkl" % i)
        smplx_smooth_file_path = os.path.join(smplx_smooth_path, "%d.json" % i)

        with open(camera_hmr_file_path,'rb') as f:
            camera_hmr = pkl.load(f)
        with open(smplx_smooth_file_path,'r') as f:
            smplx_smooth = json.load(f)
        
        #create a new dict to store the combined params
        combined_params = {}
        combined_params["betas"] = camera_hmr["smpl_params"]["betas"].detach().cpu().squeeze().numpy().tolist()
        combined_params["root_pose"] = matrix_to_axis_angle(camera_hmr["smpl_params"]["global_orient"]).detach().cpu().squeeze().numpy().tolist()
        combined_params["body_pose"] = matrix_to_axis_angle(camera_hmr["smpl_params"]["body_pose"][:,:-2,:,:]).detach().cpu().squeeze().numpy().tolist()
        combined_params["trans"] = camera_hmr["cam_t"].detach().cpu().squeeze().numpy().tolist()
        combined_params["jaw_pose"] = smplx_smooth["jaw_pose"]
        combined_params["leye_pose"] = smplx_smooth["leye_pose"]
        combined_params["reye_pose"] = smplx_smooth["reye_pose"]
        combined_params["lhand_pose"] = smplx_smooth["lhand_pose"]
        combined_params["rhand_pose"] = smplx_smooth["rhand_pose"]
        
        combined_params["focal"] = [focal,focal]
        #specialized for cxk video
        combined_params["princpt"] = [K[0,2],K[1,2]]
        combined_params["img_size_wh"] = image_size_wh
        # combined_params["pad_ratio"] = [0.2] #just a placeholder
        # for key in target_json_keys:
        #     combined_params[key] = [str(num) for num in combined_params[key]]
        combined_params["pad_ratio"] = 0.2
        #write to new json file
        with open(osp.join(output_path, "%05d.json" % i),'w') as f:
            json.dump(combined_params,f)
        
def format_smplx(file_path,output_path):
    neutral_pose_data=torch.load(file_path)
    pose=neutral_pose_data['pose'].cpu().numpy().squeeze()
    shape=neutral_pose_data['betas'].cpu().numpy().squeeze()
    print("pose:",pose.shape," shape:",shape.shape)
    for i in tqdm(range(10)):
        #create a new dict to store the combined params
        combined_params = {}
        combined_params["betas"] = [0.32866472005844116, -0.3559509217739105, 1.568978190422058, 1.1016368865966797, 0.32729071378707886, 0.2667834162712097, 0.30214911699295044, -0.5776857733726501, -0.19059178233146667, 0.3939511179924011]
        combined_params["root_pose"] = pose[0].tolist()
        combined_params["body_pose"] = pose[1:22].tolist()
        combined_params["trans"] = np.array([0,0.5,5]).tolist()
        combined_params["jaw_pose"] = np.zeros(3).tolist()
        combined_params["leye_pose"] = np.zeros(3).tolist()
        combined_params["reye_pose"] = np.zeros(3).tolist()
        combined_params["lhand_pose"] = np.zeros((15,3)).tolist()
        combined_params["rhand_pose"] = np.zeros((15,3)).tolist()
        
        combined_params["focal"] = [2000,2000]
        #specialized for cxk video
        combined_params["princpt"] = [640.0, 360.0]
        combined_params["img_size_wh"] = [1280, 720]
        # combined_params["pad_ratio"] = [0.2] #just a placeholder
        # for key in target_json_keys:
        #     combined_params[key] = [str(num) for num in combined_params[key]]
        combined_params["pad_ratio"] = 0.2
        #write to new json file
        with open(osp.join(output_path, "%05d.json" % i),'w') as f:
            json.dump(combined_params,f)

def format_neutral_pose_rot(file_path,output_path):
    #input an already formated smplx file, then only rotate the root pose along z axis,in 5s 150frames
    import torch
    import math
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    # Number of frames to rotate
    num_frames = 150

    # Step 1: 固定的绕 X 轴旋转 180°
    angle_x = math.pi
    axis_x = torch.tensor([1.0, 0.0, 0.0])
    rotvec_x = angle_x * axis_x  # (3,)

    # 转换为旋转矩阵
    R_x = axis_angle_to_matrix(rotvec_x[None, :])  # (1, 3, 3)

    # Step 2: 构造一圈绕 Z 轴的旋转角度
    angle_axis_list = []

    for i in range(num_frames):
        angle_y = 2 * math.pi * i / num_frames
        axis_y = torch.tensor([0.0, 1.0, 0.0])
        rotvec_y = angle_y * axis_y  # (3,)

        # 转换为旋转矩阵
        R_y = axis_angle_to_matrix(rotvec_y[None, :])  # (1, 3, 3)

        # 组合旋转：先绕 X，再绕 Z → 注意顺序为 R = R_z @ R_x
        R_total = torch.bmm(R_y, R_x)  # (1, 3, 3)

        # 转回 angle-axis
        aa_total = matrix_to_axis_angle(R_total)[0]  # (3,)
        angle_axis_list.append(aa_total)

    # 最终得到一个 [num_frames, 3] 的 angle-axis 列表
    # angle_axis_tensor = torch.stack(angle_axis_list, dim=0)
    angle_axis_list = np.array(angle_axis_list)
    with open(file_path,'r') as f:
        template_json = json.load(open(file_path))
    for i in tqdm(range(150)):
        rot_param = template_json.copy()
        rot_param["root_pose"] = angle_axis_list[i].tolist()
        with open(osp.join(output_path, "%05d.json" % i),'w') as f:
            json.dump(rot_param,f)
    print("rotated neutral pose json files saved to ",output_path)
    

def main():
    # args = parse_args()
    # root_path = args.root_path
    # output_path = args.output_path
    # focal = args.focal
    # print('root_path:',root_path,' focal:',focal)
    # make_virtual_cam(root_path,save_json=True,focal=focal)
    from tqdm import tqdm
    # for i in tqdm(range(1,695)):
    #     combine_smpl_mano('/home/wenbo/CameraHMR/CameraHMR_demo/demo_output/Adrian_1/smpl_init/{}.pkl'.format(i),
    #                     '/home/wenbo/test_videos/Adrian_1/mano_params/{}_mano.pkl'.format(i),
    #                     '/home/wenbo/test_videos/Adrian_1/smplx_optimized/smplx_params_smoothed/{}.json'.format(i),
    #                     '/home/wenbo/test_videos/Adrian_1/camhmr_hamba_smplx_params/{}.json'.format(i))
    
    
    """run the ExAVT/fitting/tool/run.py (DECA+HAND4WHOLE) to get smplx init
    and run the CameraHMR inference to get camera intrinsic(if unknown) and body/root/betas 
    """
    # format_smplx_file("/data1/users/wenbo/CameraHMR/CameraHMR_demo/demo_output/Adrian_1","/data1/users/wenbo/LHM/train_data/motion_video/Adrian_1/smplx_params")
    # format_smplx("/data1/users/wenbo/ExAvatar_RELEASE/avatar/data/Custom/data/Adrian_1/LHM_file/full_neutral_pose.pth",
                #  "/data1/users/wenbo/LHM/train_data/motion_video/neutral_pose_10/smplx_params")
    format_neutral_pose_rot("/data1/users/wenbo/LHM/train_data/motion_video/neutral_pose_10/smplx_params/00000.json",
                 "/data1/users/wenbo/LHM/train_data/motion_video/neutral_pose_150/smplx_params")
    
if __name__ == '__main__':
    main()
    