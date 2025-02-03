import os
import os.path as osp

import numpy as np
import trimesh
import cv2

from scipy.spatial.distance import cdist


from .grasp_utils import regularize_pc_point_count

from .pc_utils import (
    backproject_camera,
    load_depth_img,
)


def get_fetch_gripper_mesh():
    return trimesh.load("./data/fetch_gripper_base_pose.obj")


def get_gripper_pts_with_RT(gripper_mesh, RT_gripper, count=512):
    """
    Returns a sample of gripper points on its mesh, when the gripper pose is RT_gripper
    """
    gripper_pts = np.array(
        trimesh.sample.sample_surface_even(
            mesh=gripper_mesh.copy().apply_transform(RT_gripper),
            count=512,
            seed=42,
        )[0]
    )
    return gripper_pts


def get_camera_K():
    k = [
        527.8869068647631,
        0.0,
        321.7148665756361,
        0.0,
        524.7942507494529,
        230.2819198622499,
        0.0,
        0.0,
        1.0,
    ]
    intrinsics = np.array(k).reshape(3, 3)
    return intrinsics


def get_npz_data(tasks_dir, task_id, frame_id):
    """
    Gives some sample data logged from demonstrations

    - npz_data (dict): hamer and grasp transfer related data
    - obj_pc (np.array): (N,3) object point cloud from the first frame id
    - RT_camera (np.array): 4x4 TF for camera pose in robot base
    - RT_gripper (np.array): 4x4 TF for gripper pose at the frame id

    """
    ########### Folder Structure ###########
    input_dir = osp.join(tasks_dir, task_id)
    pose_log_dir = osp.join(input_dir, "pose")
    hamer_root_dir = osp.join(input_dir, "out", "hamer")
    hamer_npz_dir = osp.join(hamer_root_dir, "model")
    depth_img_dir = osp.join(input_dir, "depth")
    samv2_dir = osp.join(input_dir, "out", "samv2")
    # NOTE: Assuming only 1 folder within the samv2 directory
    masks_dir = osp.join(samv2_dir, os.listdir(samv2_dir)[0], "obj_masks")

    ############ Npz Files Listing ###########
    npz_files = [
        f
        for f in os.listdir(hamer_npz_dir)
        if osp.isfile(osp.join(hamer_npz_dir, f)) and f.lower().endswith((".npz"))
    ]
    if not npz_files:
        raise ValueError(f"No npz files found in {hamer_npz_dir}!....")

    npz_files = sorted(npz_files)

    ############ Find matching fname with frame id ###########
    npzfname = ""
    for f in npz_files:
        fid = osp.splitext(f)[0]
        if fid == frame_id:
            npzfname = f
            print("Found the npzfname!:", f)
            break

    ############ Load Hamer and Grasp Transfer Data ###########
    npz_fpath = osp.join(hamer_npz_dir, npzfname)
    npz_data = dict(
        np.load(npz_fpath, allow_pickle=True)
    )  # load the npz as dict to be able to update later

    ############ Load Camera Pose Data ###########
    pose_data = dict(np.load(osp.join(pose_log_dir, npzfname)))
    RT_camera = pose_data["RT_camera"]

    ########### Load Obj PC ###########
    # Keeping the first frame id as 1 instead of 0
    first_frame_id = "000001"
    depth_img_f = osp.join(depth_img_dir, f"{first_frame_id}.png")
    depth_im = load_depth_img(depth_img_f)
    mask_f = osp.join(masks_dir, f"{first_frame_id}.png")
    mask_im = cv2.imread(mask_f, 0)
    intrinsics = get_camera_K()
    obj_pc_first_view = backproject_camera(depth_im, intrinsics, target_mask=mask_im)

    ########### Load RT_Gripper ###########
    RT_gripper_list = npz_data["target_transfer_pose"]

    # NOTE: JUST USING THE FIRST ELEMENT, CAN LEFT/RIGHT IDX ALSO for correctness
    RT_gripper = RT_gripper_list[0]

    ########### Dict for all relevant data ###########
    logged_data = {
        "npz_data": npz_data,
        "RT_camera": RT_camera,
        "obj_pc_first_view": obj_pc_first_view,
        "RT_old": RT_gripper,  # OLD (potentially bad grasp pose)
    }

    return logged_data


def translate_grasp_along_palm_normal(RT_gripper, delta=0.05, forward_axis=0, sign=1):
    """
    - Constructs a "bad" grasp for FETCH GRIPPER
    - Does so by pushing a current grasp 5cm ahead
    - Since its Fetch, we assume palm forward axis is +x
    - Can also be used to construct a standoff grasp

    delta (float): can be positive (go along palm normal) or negative (go in reverse direction)
    forward_axis: {0 (x), 1 (y), 2 (z)} for palm normal
    sign: 1 or -1 -- sign for axis i.e +-x, +-y, +-z

    Returns:
     - RT_bad (np.array) 4x4 TF
    """
    RT_bad = RT_gripper.copy()
    if sign < 0:
        delta *= -1
    palm_normal = RT_gripper[:3, forward_axis]
    t_old = RT_gripper[:3, 3]
    t_new = t_old + delta * palm_normal
    RT_bad[:3, 3] = t_new
    return RT_bad


def determine_local_objpc_region(obj_pc, gripper_pts, dist_threshold=0.08, count=1000):
    obj_hand_dist = np.min(cdist(obj_pc, gripper_pts), axis=1)
    idxs_close = obj_hand_dist < dist_threshold
    obj_pc_subset = obj_pc[idxs_close]
    # DO FPS on selected object points
    obj_pc_subset = regularize_pc_point_count(
        obj_pc_subset, count, use_farthest_point=True
    )[0]
    return obj_pc_subset
