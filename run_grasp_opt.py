import os
import os.path as osp
import argparse

import numpy as np
import torch
import trimesh
import cv2

from scipy.spatial.distance import cdist
import plotly.graph_objects as go


from model.hand_opt import AdamGraspCmap
from utils.grasp_utils import get_handmodel, convert_9dGrasp_to_RT

from utils.pc_utils import (
    apply_extrinsics,
    estimate_normals_with_open3d,
    transform_to_camera_frame,
    compute_contact_map,
    compute_contact_map_aligned,
)

from utils.fig_utils import (
    plot_point_cloud,
    plot_point_cloud_cmap,
    plot_trimesh_mesh,
    viz_obj_pc_with_grasps,
)

from utils import test_data_utils

# from listener import ImageListener


def get_q(RT, target_model):
    """
    Constructs a grasp q tensor (9-D) for an equivalent 4x4 RT pose
    """
    tra = RT[:3, 3]
    rot = RT[:3, :3]
    q = torch.zeros(9)
    q[:3] = torch.tensor(tra)
    q[3:] = torch.tensor(rot.T.reshape(-1)[:6])
    q = q.to(target_model.device)
    if q.shape[0] != 9 + len(target_model.dynamic_joints):
        # We optimized only for pose, so need to provide dummy joints
        q = torch.cat(
            (
                q,
                (
                    target_model.dynamic_joints_q_upper[0]
                    - target_model.dynamic_joints_q_mid[0]
                ),
            ),
            dim=0,
        )
    return q


def optimize_grasp(
    obj_pc: np.array,
    RT_camera: np.array,
    RT_current: np.array,
    device: str,
    target_gripper: str = "fetch_gripper",
    threshold_dist_local: float = 0.1,
    energy_func: str = "align_dist",
    sharp_factor: float = 10,
    weight_collision: float = 1,
    weight_contact: float = 0.2,
    optimize_only_translation: bool = False,
    num_opt_iters: int = 100,
    num_parallel_opt: int = 4,
) -> np.array:

    # Collect params and flags
    THRESHOLD_DIST_LOCAL = threshold_dist_local
    ENERGY_FUNC = energy_func
    SHARP_FACTOR = sharp_factor
    WT_COLLISION = weight_collision
    WT_CONTACT = weight_contact
    OPT_ONLY_TRANS = optimize_only_translation
    NUM_ITERS = num_opt_iters
    assert ENERGY_FUNC in {"align_dist", "euclidean_dist"}

    # obtain fetch gripper mesh
    fetch_gripper_mesh = test_data_utils.get_fetch_gripper_mesh()

    target_model = get_handmodel(
        target_gripper,
        1,
        device,
        json_path="urdf_assets_meta.json",
        datadir="./grippers/",
    )

    ############### Local Region of ObjPC ###############
    # NOTE
    # Determine local region for the object pc using distance to fetch gripper mesh points
    gripper_pts = test_data_utils.get_gripper_pts_with_RT(
        fetch_gripper_mesh, RT_current
    )
    obj_pc_subset = test_data_utils.determine_local_objpc_region(
        obj_pc, gripper_pts, dist_threshold=THRESHOLD_DIST_LOCAL, count=1000
    )

    ########## Object PC Normals & ContactMap ##########
    objpcd_with_normals_in_world = estimate_normals_with_open3d(
        apply_extrinsics(obj_pc_subset, RT_camera), RT_camera[:3, 3]
    )
    # Get in camera frame
    objpcd_with_normals_in_camera = transform_to_camera_frame(
        objpcd_with_normals_in_world, RT_camera
    )
    objpc_pts = np.array(obj_pc_subset)
    objpc_nrm = np.asarray(objpcd_with_normals_in_camera.normals)

    ## The standoff for the "colliding" grasp will the Source Grasp in GraspOpt
    RT_standoff = test_data_utils.translate_grasp_along_palm_normal(
        RT_current, delta=-0.1
    )
    q_standoff_grasp = get_q(RT_standoff, target_model)
    q_current_grasp = get_q(RT_current, target_model)
    source_q = q_current_grasp  # NOTE: using the bad grasp as initial grasp
    # source_q = q_standoff_grasp

    # NOTE: Use gripper surface points for the contact map goal
    gripper_surf_pts_for_cmap = (
        target_model.get_surface_points(q_current_grasp.unsqueeze(0))[0].cpu().numpy()
    )

    if ENERGY_FUNC == "align_dist":
        contact_map = compute_contact_map_aligned(
            gripper_surf_pts_for_cmap, objpc_pts, objpc_nrm, SHARP_FACTOR
        )
    else:
        contact_map = compute_contact_map(
            gripper_surf_pts_for_cmap, objpc_pts, SHARP_FACTOR
        )

    # removed viz

    # ############# VIZ: Local Obj PC region and Gripper Pts + Contact Map #############
    # vis_data = []
    # vis_data += [plot_point_cloud_cmap(objpc_pts, color_levels=contact_map, size=3)]
    # vis_data += [
    #     plot_trimesh_mesh(
    #         fetch_gripper_mesh.copy().apply_transform(RT_current),
    #         color="gray",
    #         opacity=0.8,
    #     )
    # ]

    # fig = go.Figure(data=vis_data)
    # fig.show()
    #############################################################################

    ######### Grasp Opt Init #########

    # # Construct the contact map goal:
    # # Goal = [obj pc points (N,3), obj pc normals (N,3), contact map (N, 1)], Shape = (N, 7)

    # Augmented Obj Pt Cloud
    # NOTE: Initialize a thin shell around the object point cloud for collision
    # consideration
    # NOTE: 0.005 i.e 5mm used since using larger values effectively means that
    # we are scaling up the object --> could create issues in the optimization
    augmented_obj_pts = objpc_pts + 0.005 * objpc_nrm
    cmap_goal = np.concatenate(
        [augmented_obj_pts, objpc_nrm, contact_map.reshape(-1, 1)], axis=1
    )

    # cmap_goal = np.concatenate(
    #     [objpc_pts, objpc_nrm, contact_map.reshape(-1, 1)], axis=1
    # )
    cmap_tensor = torch.tensor(cmap_goal)

    grasp_transfer_opt = AdamGraspCmap(
        target_robot_name=target_gripper,
        contact_weight=WT_CONTACT,
        collision_weight=WT_COLLISION,
        opt_only_trans=OPT_ONLY_TRANS,
        sharp_factor=SHARP_FACTOR,
        source_grasp=source_q,
        num_particles=num_parallel_opt,
        learning_rate=1e-3,
        max_iter=NUM_ITERS,
        device=device,
        energy_func_name=ENERGY_FUNC,
    )

    q_traj, energy, _ = grasp_transfer_opt.run_adam(
        contact_map_goal=cmap_tensor, source_grasp=source_q, running_name="test"
    )
    min_energy_index = energy.min(dim=0)[1]
    best_q = q_traj[min_energy_index.item(), -1]
    if best_q.shape[0] != 9 + len(target_model.dynamic_joints):
        # We optimized only for pose, so need to provide dummy joints
        best_q = torch.cat(
            (
                best_q,
                (
                    target_model.dynamic_joints_q_upper[0]
                    - target_model.dynamic_joints_q_mid[0]
                ),
            ),
            dim=0,
        )
    RT_optimized_grasp = convert_9dGrasp_to_RT(best_q)

    ############### GrasOpt Result Viz ################
    # VIZ: Local Obj PC region and Gripper Pts + Contact Map
    vis_data = []
    vis_data += [plot_point_cloud_cmap(objpc_pts, color_levels=contact_map, size=4)]

    # vis_data += [plot_point_cloud(augmented_obj_pts, color="blue", opacity=0.3)]

    # hand_mesh_pts = (
    #     target_model.get_fullmesh_points(q=best_q.unsqueeze(0))
    #     .squeeze(0)
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )
    # hand_surf_pts = (
    #     target_model.get_surface_points(q=best_q.unsqueeze(0))
    #     .squeeze(0)
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )
    # vis_data += [plot_point_cloud(hand_mesh_pts, color="black", size=2)]
    # vis_data += [plot_point_cloud(hand_surf_pts, color="red", size=2)]

    vis_data += [
        plot_trimesh_mesh(
            fetch_gripper_mesh.copy().apply_transform(RT_current),
            color="red",
            opacity=0.3,
        )
    ]
    # vis_data += [
    #     plot_trimesh_mesh(
    #         fetch_gripper_mesh.copy().apply_transform(RT_standoff),
    #         color="lightblue",
    #         opacity=0.6,
    #     )
    # ]
    vis_data += [
        plot_trimesh_mesh(
            fetch_gripper_mesh.copy().apply_transform(RT_optimized_grasp),
            color="lightgreen",
            opacity=0.3,
        )
    ]
    fig = go.Figure(data=vis_data)
    fig.show()
    return RT_optimized_grasp


def make_parser():
    parser = argparse.ArgumentParser(
        description="Main function only to be used for testing"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to demo data dir",
        default="/home/ninad/Datasets/MMDemo/newCamK",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        help="Task Name to run the test on",
        default="task_18_10s-move-chair",
    )
    parser.add_argument(
        "--ros",
        type=str,
        help="Task Name to run the test on",
        default="n",
    )
    parser.add_argument(
        "--frame_id",
        type=int,
        help="Frame ID of the grasp to perturb/distrub and test the optimization on",
        default=31,
    )
    return parser


if __name__ == "__main__":
    import sys
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ########## testing code ###########
    parser = make_parser()
    args = parser.parse_args()
    tasks_dir = args.dataset_dir
    task_id = args.task_id
    frame_id = args.frame_id
    if args.ros == "n":
        # Considering the "Task 18 Move Chair" and Frame 31 from it by default
        frame_id_str = f"{frame_id:06d}"  # = "000031"
        print("Demo Data dir:", tasks_dir)
        print("Task id:", task_id)
        print("Frame id:", frame_id_str)

        target_gripper = "fetch_gripper"

        input_dir = osp.join(tasks_dir, task_id)
        logged_data = test_data_utils.get_npz_data(
            tasks_dir=tasks_dir, task_id=task_id, frame_id=frame_id_str
        )
        npz_data = logged_data["npz_data"]
        obj_pc_first_view = logged_data["obj_pc_first_view"]
        RT_camera = logged_data["RT_camera"]
        RT_old = logged_data["RT_old"]
        import pdb

        pdb.set_trace()
        # NOTE: Perturb the given gripper pose
        # RT_current here reflect what we might see in real world,
        # i.e a potentially bad grasp pose
        # We obtain it by translating in +x by some distance 0.05m (reverse of standoff)
        RT_current = test_data_utils.translate_grasp_along_palm_normal(
            RT_old, delta=0.05
        )
        ################## End Loading of Dummy Data ###############

        # NOTE: Load any real world data here if needed
        # NOTE: Can also add saving the RT_opt_grasp here if needed
    else:
        # import rospy

        # rospy.init_node("testrfp")
        # listener = ImageListener()
        # time.sleep(3)
        # RT_camera, obj_pc_first_view = listener.get_data_to_save()
        # RT_current = np.eye(4, 4)
        # RT_current[0, 3] += 0.6
        pass

    ############## Sample Run of Grasp Opt ##############

    try:
        # 1. pointclod, grasp from npz file
        # 2. save grasp in another npz file
        # input_fname = "/tmp/opt_data.npz"
        # input_fname = "/home/ninad/Datasets/MMDemo/grasp_opt_trial_march17-selected/hammer-corrected/opt_data.npz"
        input_fname = "/home/ninad/Datasets/MMDemo/grasp_opt_trial_march17-selected/utd-bottle-shelf-corrected/opt_data.npz"
        data = np.load(input_fname)
        obj_pc_first_view = data["object_pc"]
        RT_current = data["RT_grasp"]
        RT_camera = data["RT_camera"]

        RT_gopt = optimize_grasp(
            obj_pc=obj_pc_first_view,
            RT_camera=RT_camera,
            RT_current=RT_current,
            device=device,
            sharp_factor=20,
            weight_collision=1,
            num_opt_iters=100,
            num_parallel_opt=8,
        )

        np.savez("/tmp/opt_grasp.npz", opt_RT_grasp=RT_gopt)
    except ValueError:
        print(f"No solution to the optimization found")

    print("Done....")
