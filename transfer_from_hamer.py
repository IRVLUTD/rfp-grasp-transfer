import sys
import os
import os.path as osp
import argparse
from tqdm import tqdm

import torch
import trimesh
import numpy as np

from transforms3d.axangles import axangle2mat, mat2axangle
import plotly.graph_objects as go

from mano_pybullet.hand_model import HandModel20

from utils.grasp_utils import get_handmodel, rotation_matrix_from_vectors
from utils.rot6d_utils import mat2rvec, robust_compute_rotation_matrix_from_ortho6d
from model.hand_opt import AdamGraspTransfer, BatchedAdamGraspTransfer
from model.hand_model import GcsHandModel

from typing import Dict
from collections import namedtuple


LeftRightTuple = namedtuple("LeftRightTuple", ["left", "right"])


def process_hamer_output(npz_data) -> Dict:
    """Process npz file saved by hamer containing data for inferred
    mano hand pose and optimized translation in camera frame. This function
    just unpacks the npz file for downstream convinience.

    Input:

    data: loaded hamer output npz file
    """
    # NOTE:
    # data['right'] can contain both singleton 0/1 or both [0,1] or [1,0]
    # indicating that hamer detected both left and right hands
    # 0 -> left_hand_idx ; 1 -> right_hand_idx
    # [0, 1] --> in the mano pose array, the "0th" index is for left hand
    # [1, 0] --> in the mano pose array, the "1th" index is for left hand
    rl_index = npz_data["right"]
    num_detected = npz_data["right"].shape[0]
    left_idxs = np.arange(rl_index.shape[0])[rl_index == 0]
    right_idxs = np.arange(rl_index.shape[0])[rl_index == 1]
    # at max, we should have data for only 1 left and 1 right hand
    assert (left_idxs.size <= 1) and (right_idxs.size <= 1)
    return {
        "rl_index": rl_index,
        "num_detected": num_detected,
        "mano_params": npz_data["pred_mano_params"].item(),
        "translation": npz_data["opt_translation"],
        "left_idxs": left_idxs,
        "right_idxs": right_idxs,
    }


def extract_source_data(mano_params, translation_array, idx_to_use):
    if idx_to_use.size == 0:
        return {}
    source_data = {
        "hand_rot_mat": mano_params["global_orient"][idx_to_use][0][0],
        "hand_thetas": mano_params["hand_pose"][idx_to_use][0],
        "translation": translation_array[idx_to_use][0],
    }
    return source_data


def transfer_grasp_handler(
    hamer_data: Dict,
    target_model: GcsHandModel,
    source_models: LeftRightTuple,
    manopyb_models: LeftRightTuple,
    grasp_transfer_opts: LeftRightTuple,
    want_vis_data: bool = False,
):
    """
    Can potentially contain data for both left and right hands so this
    function will try to do the transfer for both if needed and offload
    the usage of left/right hand to dowstream trajopt framework.

    Input:
    ------

    - source_models: Tuple for gcs hand models of (mano_left, mano_right)

    - manopyb_models: Tuple for mano_pybullet models of (mano_left, mano_right)

    - target_model: target gcs hand model

    - grasp_transfer_opts: Tuple of pre-built AdamGraspTransfer instances
        (left, right). These are reused across frames; building them per-frame
        re-runs URDF parsing and correspondence calc.
    """
    rl_index = hamer_data["rl_index"]
    num_detected = hamer_data["num_detected"]
    mano_params = hamer_data["mano_params"]
    translation = hamer_data["translation"]
    left_idxs = hamer_data["left_idxs"]
    right_idxs = hamer_data["right_idxs"]

    has_right = right_idxs.size > 0
    has_left = left_idxs.size > 0

    if num_detected > 1:
        assert has_left and has_right

        source_data_right = extract_source_data(mano_params, translation, right_idxs)
        RT_target_right, fig_right, mesh_right = transfer_grasp(
            source_data_right,
            source_models.right,
            manopyb_models.right,
            target_model,
            is_left=False,
            grasp_transfer_opt=grasp_transfer_opts.right,
            want_vis_data=want_vis_data,
        )

        source_data_left = extract_source_data(mano_params, translation, left_idxs)
        RT_target_left, fig_left, mesh_left = transfer_grasp(
            source_data_left,
            source_models.left,
            manopyb_models.left,
            target_model,
            is_left=True,
            grasp_transfer_opt=grasp_transfer_opts.left,
            want_vis_data=want_vis_data,
        )

        # right_idxs and left_idxs will be a list with single element, indexing into hamer output batched array
        # right_idxs[0] and left_idxs[0] give us this exact index integer
        result = [None, None]
        result[right_idxs[0]] = RT_target_right
        result[left_idxs[0]] = RT_target_left

        plots = LeftRightTuple(left=fig_left, right=fig_right)
        meshes = LeftRightTuple(left=mesh_left, right=mesh_right)

    else:
        assert has_left or has_right
        result = []
        if has_right:
            source_data = extract_source_data(mano_params, translation, right_idxs)
            RT_target_right, fig_right, mesh_right = transfer_grasp(
                source_data,
                source_models.right,
                manopyb_models.right,
                target_model,
                is_left=False,
                grasp_transfer_opt=grasp_transfer_opts.right,
                want_vis_data=want_vis_data,
            )
            result = [RT_target_right]
            plots = LeftRightTuple(left=None, right=fig_right)
            meshes = LeftRightTuple(left=None, right=mesh_right)
        if has_left:
            source_data = extract_source_data(mano_params, translation, left_idxs)
            RT_target_left, fig_left, mesh_left = transfer_grasp(
                source_data,
                source_models.left,
                manopyb_models.left,
                target_model,
                is_left=True,
                grasp_transfer_opt=grasp_transfer_opts.left,
                want_vis_data=want_vis_data,
            )
            result = [RT_target_left]
            plots = LeftRightTuple(left=fig_left, right=None)
            meshes = LeftRightTuple(left=mesh_left, right=None)

    return np.array(result), plots, meshes


def transfer_grasp(
    source_data: Dict,
    source_model: GcsHandModel,
    manopyb_model: HandModel20,
    target_model: GcsHandModel,
    is_left: bool,
    grasp_transfer_opt: "AdamGraspTransfer" = None,
    want_vis_data: bool = False,
):
    import time as _t
    _stage = {}
    _t0 = _t.time()
    hand_rot_mat = source_data["hand_rot_mat"]
    hand_theta_mat = source_data["hand_thetas"]
    trans = source_data["translation"]

    if is_left:
        hand_rot_mat[1::3] *= -1
        hand_rot_mat[2::3] *= -1
        hand_theta_mat[1::3] *= -1
        hand_theta_mat[1::3] *= -1

    hand_theta_full = np.array(
        [mat2rvec(hand_rot_mat)]
        + [mat2rvec(hand_theta_mat[i]) for i in range(hand_theta_mat.shape[0])]
    )

    angles, palm_basis = manopyb_model.mano_to_angles(hand_theta_full)
    pyb_model_origin = manopyb_model.origins()[0]
    palm_trans = trans + pyb_model_origin - palm_basis @ pyb_model_origin

    actual_trans = np.array(palm_trans)
    actual_basis = np.array(palm_basis)
    if is_left:
        # actual_trans -= trans
        # actual_trans[0] *= -1
        # actual_trans += trans
        # r_palm_normal = palm_basis @ np.array([0, -1, 0])
        # r_palm_normal_flip = np.array(r_palm_normal)
        # r_palm_normal_flip[0] *= -1
        # rotmat_flip = rotation_matrix_from_vectors(r_palm_normal, r_palm_normal_flip)
        # actual_basis = rotmat_flip @ palm_basis
        R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        actual_basis = np.dot(actual_basis, R_x)

    grasp_pose = torch.zeros(9)
    # Rotation in 6D representation looks like: (x1,x2,x3, y1,y2,y3) (1st 2 columns from the rot mat)
    grasp_pose[3:] = torch.tensor(actual_basis.T.reshape(-1)[:6])
    grasp_pose[:3] = torch.tensor(actual_trans)

    # grasp_dofs = torch.tensor(angles)
    grasp_dofs = -1 * torch.tensor(angles) if is_left else torch.tensor(angles)

    source_grasp_q = (
        torch.cat(
            [
                grasp_pose,
                grasp_dofs,
            ]
        )
        .unsqueeze(0)
        .to(source_model.device)
        .float()
    )

    # Build per-call as a fallback if a pre-built optimizer wasn't passed in.
    # Hot-path callers should pass grasp_transfer_opt to avoid the URDF reload
    # and correspondence rebuild that happens in __init__.
    if grasp_transfer_opt is None:
        grasp_transfer_opt = AdamGraspTransfer(
            source_model.robot_name,
            target_model.robot_name,
            learning_rate=1e-3,
            device=source_model.device,
        )

    _stage["pre_adam"] = _t.time() - _t0
    _t1 = _t.time()
    q_traj, energy, _ = grasp_transfer_opt.run_adam(
        source_grasp_q.squeeze(0), running_name="test"
    )
    _stage["run_adam"] = _t.time() - _t1
    _t2 = _t.time()
    min_energy_index = energy.min(dim=0)[1].item()
    best_target_q = q_traj[min_energy_index, -1]
    target_grasp_q = best_target_q.detach()

    # Convert the target grasp q (9-dim) to 4x4 pose transform RT
    target_rot6d = target_grasp_q[3:9]
    target_trans = target_grasp_q[:3].cpu().numpy()
    target_rot_mat = (
        robust_compute_rotation_matrix_from_ortho6d(target_rot6d.unsqueeze(0))
        .squeeze(1)
        .cpu()
        .numpy()
    )
    target_RT = np.eye(4)
    target_RT[:3, :3] = target_rot_mat
    target_RT[:3, 3] = target_trans

    if target_grasp_q.shape[0] != 9 + len(target_model.dynamic_joints):
        # We optimized only for pose, so need to provide dummy joints
        target_grasp_q = torch.cat(
            (
                target_grasp_q,
                (
                    target_model.dynamic_joints_q_upper[0]
                    - target_model.dynamic_joints_q_mid[0]
                ),
            ),
            dim=0,
        )

    # target_gripper_mesh_data is required (we extract trimesh data from it for
    # the mandatory PLY export). source_model.get_plotly_data is only consumed
    # by the optional --debug_plots viz path; skip it when not needed.
    _t3 = _t.time()
    target_gripper_mesh_data = target_model.get_plotly_data(
        q=target_grasp_q.unsqueeze(0).float().to(target_model.device),
        color="green",
        opacity=0.3,
    )
    if want_vis_data:
        vis_data = source_model.get_plotly_data(q=source_grasp_q, color="red", opacity=0.3)
        vis_data += target_gripper_mesh_data
    else:
        vis_data = []
    _stage["plotly"] = _t.time() - _t3

    # Target gripper mesh
    _t4 = _t.time()
    target_grp_trimesh = []
    for mesh in target_gripper_mesh_data:
        vertices = np.array([mesh.x, mesh.y, mesh.z]).T
        faces = np.array([mesh.i, mesh.j, mesh.k]).T
        target_grp_trimesh.append(trimesh.Trimesh(vertices=vertices, faces=faces))
    target_mesh = trimesh.util.concatenate(target_grp_trimesh)
    _stage["trimesh"] = _t.time() - _t4

    if os.environ.get("FASTEN_PROFILE"):
        print(
            "[transfer_grasp]",
            " | ".join(f"{k}={v*1000:.0f}ms" for k, v in _stage.items()),
        )
    return target_RT, vis_data, target_mesh


def main(args):
    input_dir = args.input_dir
    mano_dir = args.mano_model_dir
    target_gripper = args.target_gripper
    debug_plots = args.debug_plots

    if not osp.isdir(mano_dir):
        raise FileNotFoundError(
            f"Mano models dir not found: {mano_dir}! Please specify a correct path using `--mano_model_dir` argument."
        )
    if not osp.isdir(input_dir):
        raise FileNotFoundError(
            f"Input dir for demonstration data not found: {input_dir}"
        )
    hamer_root_dir = osp.join(input_dir, "out", "hamer")
    hamer_npz_dir = osp.join(hamer_root_dir, "model")
    if not osp.isdir(hamer_npz_dir):
        raise FileNotFoundError(
            f"The demo data dir does not contain output from hamer at the expected location: {hamer_npz_dir}"
        )
    # Set the MANO DIR for `mano_pybullet` interfacing
    os.environ["MANO_MODELS_DIR"] = mano_dir
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    assert device in {"cuda", "cpu"}

    # Initiliaze HandModels for mano left/right and target gripper
    _source_model_left = get_handmodel(
        "mano_left",
        1,
        device,
        json_path="urdf_assets_meta.json",
        datadir="./grippers/",
    )
    _source_model_right = get_handmodel(
        "mano_right",
        1,
        device,
        json_path="urdf_assets_meta.json",
        datadir="./grippers/",
    )
    target_model = get_handmodel(
        target_gripper,
        1,
        device,
        json_path="urdf_assets_meta.json",
        datadir="./grippers/",
    )

    # Initialize Mano Pybullet models for left/right (useful for conversion from mano to urdf equivalent)
    _manopyb_left = HandModel20(left_hand=True)
    _manopyb_right = HandModel20(left_hand=False)

    # Init the named tuples for mano models (gcs and mano_pybullet)
    source_models = LeftRightTuple(left=_source_model_left, right=_source_model_right)
    manopyb_models = LeftRightTuple(left=_manopyb_left, right=_manopyb_right)

    # Build the AdamGraspTransfer optimizers once (left/right) and reuse across frames.
    # Each instance loads URDF + builds the kinematic chain in its constructor; doing
    # this per-frame was a major cost.
    grasp_transfer_opts = LeftRightTuple(
        left=AdamGraspTransfer(
            _source_model_left.robot_name,
            target_model.robot_name,
            learning_rate=1e-3,
            max_iter=args.max_iter,
            num_particles=args.num_particles,
            device=device,
        ),
        right=AdamGraspTransfer(
            _source_model_right.robot_name,
            target_model.robot_name,
            learning_rate=1e-3,
            max_iter=args.max_iter,
            num_particles=args.num_particles,
            device=device,
        ),
    )

    # Populate a list of hamer output npz files to iterate over and transfer grasp
    npz_files = [
        f
        for f in os.listdir(hamer_npz_dir)
        if osp.isfile(osp.join(hamer_npz_dir, f)) and f.lower().endswith((".npz"))
    ]
    if not npz_files:
        raise ValueError(f"No npz files found in {hamer_npz_dir}!....")

    # Create extra dirs for grasp transfer plots: gripper mesh and plotly html figure viz
    transfer_mesh_dir = osp.join(hamer_root_dir, "transfer_hand_mesh")
    os.makedirs(transfer_mesh_dir, exist_ok=True)
    if debug_plots:
        transfer_extra_dir = osp.join(hamer_root_dir, "transfer_extra_plots")
        os.makedirs(transfer_extra_dir, exist_ok=True)
        print(
            f"\n[NOTE] Debuggig arg passed, creating/checking dir:{transfer_extra_dir}.\nConsider deleting it after debugging!\n"
        )

    import time as _time
    frame_times = []

    # ------------- Batched path (frame_batch_size > 1) -------------
    if args.frame_batch_size > 1:
        F = args.frame_batch_size
        print(f"[batched] frame_batch_size={F} num_particles={args.num_particles} "
              f"max_iter={args.max_iter}")
        # Build batched optimizers (left/right). These have target_handmodel
        # sized for F * num_particles.
        batched_opts = LeftRightTuple(
            left=BatchedAdamGraspTransfer(
                _source_model_left.robot_name, target_model.robot_name,
                frame_batch_size=F, num_particles=args.num_particles,
                max_iter=args.max_iter, learning_rate=1e-3, device=device,
            ),
            right=BatchedAdamGraspTransfer(
                _source_model_right.robot_name, target_model.robot_name,
                frame_batch_size=F, num_particles=args.num_particles,
                max_iter=args.max_iter, learning_rate=1e-3, device=device,
            ),
        )

        sorted_files = sorted(npz_files)
        # Process frames in chunks of size F. The last chunk may be smaller;
        # we pad it by repeating the last frame (results for padding are
        # discarded after).
        for chunk_start in tqdm(range(0, len(sorted_files), F)):
            t_chunk = _time.time()
            chunk = sorted_files[chunk_start:chunk_start + F]
            real_len = len(chunk)
            while len(chunk) < F:
                chunk.append(chunk[-1])  # pad

            # Pre-process: load npz, build source_grasp_q per (frame, hand-side).
            chunk_npz = []
            chunk_hamer = []
            for npz_f in chunk:
                npz_fpath = osp.join(hamer_npz_dir, npz_f)
                npz_data = dict(np.load(npz_fpath, allow_pickle=True))
                chunk_npz.append((npz_fpath, npz_data))
                chunk_hamer.append(process_hamer_output(npz_data))

            def _build_source_q(source_data, manopyb_model, target_model, is_left):
                """Reproduce the pre-Adam math from transfer_grasp.
                Returns source_grasp_q tensor (q_dim,)."""
                hand_rot_mat = source_data["hand_rot_mat"].copy()
                hand_theta_mat = source_data["hand_thetas"].copy()
                trans = source_data["translation"]
                if is_left:
                    hand_rot_mat[1::3] *= -1
                    hand_rot_mat[2::3] *= -1
                    # Matches transfer_grasp's exact (buggy-but-shipped) behavior:
                    # the original writes hand_theta_mat[1::3] *= -1 twice, which
                    # is a no-op. Do not collapse to a single negation — it would
                    # produce non-orthogonal matrices that mat2rvec rejects.
                    hand_theta_mat[1::3] *= -1
                    hand_theta_mat[1::3] *= -1
                hand_theta_full = np.array(
                    [mat2rvec(hand_rot_mat)]
                    + [mat2rvec(hand_theta_mat[i]) for i in range(hand_theta_mat.shape[0])]
                )
                angles, palm_basis = manopyb_model.mano_to_angles(hand_theta_full)
                pyb_model_origin = manopyb_model.origins()[0]
                palm_trans = trans + pyb_model_origin - palm_basis @ pyb_model_origin
                actual_trans = np.array(palm_trans)
                actual_basis = np.array(palm_basis)
                if is_left:
                    R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    actual_basis = np.dot(actual_basis, R_x)
                grasp_pose = torch.zeros(9)
                grasp_pose[3:] = torch.tensor(actual_basis.T.reshape(-1)[:6])
                grasp_pose[:3] = torch.tensor(actual_trans)
                grasp_dofs = -1 * torch.tensor(angles) if is_left else torch.tensor(angles)
                return torch.cat([grasp_pose, grasp_dofs]).float()

            # Collect per-side source_qs (only for frames that have that hand).
            left_qs, left_idxs_in_chunk = [], []
            right_qs, right_idxs_in_chunk = [], []
            for ci, hd in enumerate(chunk_hamer):
                if hd["right_idxs"].size:
                    sd = extract_source_data(
                        hd["mano_params"], hd["translation"], hd["right_idxs"]
                    )
                    right_qs.append(_build_source_q(
                        sd, manopyb_models.right, target_model, is_left=False))
                    right_idxs_in_chunk.append(ci)
                if hd["left_idxs"].size:
                    sd = extract_source_data(
                        hd["mano_params"], hd["translation"], hd["left_idxs"]
                    )
                    left_qs.append(_build_source_q(
                        sd, manopyb_models.left, target_model, is_left=True))
                    left_idxs_in_chunk.append(ci)

            # Pad each side's list to exactly F before batched call.
            def _pad_to_F(qs):
                if not qs:
                    return None
                while len(qs) < F:
                    qs.append(qs[-1])
                return qs

            right_qs_padded = _pad_to_F(right_qs[:])
            left_qs_padded = _pad_to_F(left_qs[:])

            # Run batched Adam for each side.
            right_results = batched_opts.right.run_adam_multi(right_qs_padded) \
                if right_qs_padded else []
            left_results = batched_opts.left.run_adam_multi(left_qs_padded) \
                if left_qs_padded else []

            # Per-frame post-processing: extract RT, plotly mesh, save PLY+npz.
            for ci in range(real_len):
                npz_fpath, npz_data = chunk_npz[ci]
                hd = chunk_hamer[ci]
                num_detected = hd["num_detected"]
                rl_index = hd["rl_index"]

                def _q_to_RT(target_grasp_q):
                    target_rot6d = target_grasp_q[3:9]
                    target_trans = target_grasp_q[:3].cpu().numpy()
                    target_rot_mat = (
                        robust_compute_rotation_matrix_from_ortho6d(target_rot6d.unsqueeze(0))
                        .squeeze(1).cpu().numpy()
                    )
                    RT = np.eye(4)
                    RT[:3, :3] = target_rot_mat
                    RT[:3, 3] = target_trans
                    return RT

                def _mesh_from_q(target_grasp_q):
                    if target_grasp_q.shape[0] != 9 + len(target_model.dynamic_joints):
                        target_grasp_q = torch.cat(
                            (target_grasp_q,
                             (target_model.dynamic_joints_q_upper[0]
                              - target_model.dynamic_joints_q_mid[0])),
                            dim=0,
                        )
                    plotly_meshes = target_model.get_plotly_data(
                        q=target_grasp_q.unsqueeze(0).float().to(target_model.device),
                        color="green", opacity=0.3,
                    )
                    parts = []
                    for m in plotly_meshes:
                        parts.append(trimesh.Trimesh(
                            vertices=np.array([m.x, m.y, m.z]).T,
                            faces=np.array([m.i, m.j, m.k]).T,
                        ))
                    return trimesh.util.concatenate(parts)

                # Locate this frame's results in the per-side lists.
                ri = right_idxs_in_chunk.index(ci) if ci in right_idxs_in_chunk else None
                li = left_idxs_in_chunk.index(ci) if ci in left_idxs_in_chunk else None
                RT_right = _q_to_RT(right_results[ri][0]) if ri is not None else None
                RT_left = _q_to_RT(left_results[li][0]) if li is not None else None

                # Re-merge into the same RT_result layout that transfer_grasp_handler produces.
                if num_detected > 1:
                    rt_arr = [None, None]
                    rt_arr[hd["right_idxs"][0]] = RT_right
                    rt_arr[hd["left_idxs"][0]] = RT_left
                    RT_result = np.array(rt_arr)
                else:
                    RT_result = np.array([RT_right if ri is not None else RT_left])

                npz_data["target_transfer_pose"] = RT_result
                np.savez(npz_fpath, **npz_data)

                fname, _ = os.path.splitext(os.path.basename(npz_fpath))
                if li is not None:
                    _mesh_from_q(left_results[li][0]).export(
                        osp.join(transfer_mesh_dir, f"{fname}_0.ply"))
                if ri is not None:
                    _mesh_from_q(right_results[ri][0]).export(
                        osp.join(transfer_mesh_dir, f"{fname}_1.ply"))

            frame_times.append((_time.time() - t_chunk) / real_len)

        if frame_times:
            avg = sum(frame_times) / len(frame_times)
            print(f"[grasp-transfer] processed {len(sorted_files)} frames "
                  f"(batched, F={F}) | avg {avg*1000:.1f} ms/frame | "
                  f"total {sum(frame_times)*F:.1f}s")
        return  # end batched path

    # ------------- Per-frame path (frame_batch_size == 1, original behavior) -------------
    for npz_f in tqdm(sorted(npz_files)):
        t_frame = _time.time()
        npz_fpath = osp.join(hamer_npz_dir, npz_f)
        npz_data = dict(
            np.load(npz_fpath, allow_pickle=True)
        )  # load the npz as dict to be able to update later
        hamer_data = process_hamer_output(npz_data)
        RT_result, plots, meshes = transfer_grasp_handler(
            hamer_data, target_model, source_models, manopyb_models, grasp_transfer_opts,
            want_vis_data=debug_plots,
        )
        frame_times.append(_time.time() - t_frame)
        npz_data["target_transfer_pose"] = RT_result
        np.savez(npz_fpath, **npz_data)

        # Save viz and transfer data logs
        fname, _ = os.path.splitext(os.path.basename(npz_fpath))
        if meshes.left:
            meshes.left.export(osp.join(transfer_mesh_dir, f"{fname}_0.ply"))
        if meshes.right:
            meshes.right.export(osp.join(transfer_mesh_dir, f"{fname}_1.ply"))

        if debug_plots:
            if plots.left:
                vis_data = plots.left
                mano_pc = trimesh.load_mesh(
                    osp.join(hamer_root_dir, "3dhand", f"{fname}_0.ply")
                )
                x, y, z = mano_pc.vertices.T
                vis_data += [
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(size=2, color="green"),
                    )
                ]
                fig = go.Figure(data=vis_data)
                fig.write_html(osp.join(transfer_extra_dir, f"{fname}_0.html"))
            if plots.right:
                vis_data = plots.right
                mano_pc = trimesh.load_mesh(
                    osp.join(hamer_root_dir, "3dhand", f"{fname}_1.ply")
                )
                x, y, z = mano_pc.vertices.T
                vis_data += [
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="markers",
                        marker=dict(size=2, color="green"),
                    )
                ]
                fig = go.Figure(data=vis_data)
                fig.write_html(osp.join(transfer_extra_dir, f"{fname}_1.html"))

    if frame_times:
        avg = sum(frame_times) / len(frame_times)
        print(f"[grasp-transfer] processed {len(frame_times)} frames | avg {avg*1000:.1f} ms/frame | total {sum(frame_times):.1f}s")


def make_parser():
    parser = argparse.ArgumentParser(
        prog="transfer_from_hamer",
        description="Runs the rfp-based grasp transfer on Mano hand (inferred by Hamer) to Target (Fetch) gripper",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        # required=True,
        default="/home/ninad/Datasets/MMDemo/whiteboard-eraser_interval_0.05/",
        help="Directory containing demonstration data including mano hand inference output from hamer",
    )
    parser.add_argument(
        "-m",
        "--mano_model_dir",
        type=str,
        default="/home/ninad/Projects/MANO/MANO_Hand_Model/mano_v1_2/models",
        help="Path to Mano `models` dir, example: `/home/ninad/Projects/MANO/MANO_Hand_Model/mano_v1_2/models`",
    )
    parser.add_argument(
        "-t",
        "--target_gripper",
        type=str,
        default="fetch_gripper",
        help="Name of the gripper to transfer the grasp to.",
    )
    parser.add_argument(
        "--debug_plots",
        action="store_true",
        help="This creates a ~ 5MB html plot for each frame, so only use for debugging and delete its folder after use!",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Adam optimization iterations per frame. With warm-starting and 16 particles, 50 typically converges (was 300).",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=16,
        help="Particle batch size for the Adam grasp-transfer optimizer (was 32). Lower = faster, slight quality hit.",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=1,
        help="Number of frames to optimize together in one Adam call via "
             "BatchedAdamGraspTransfer. Default 1 keeps the original per-frame "
             "loop. Values like 4-16 process multiple frames in parallel on GPU, "
             "amortizing kernel-launch overhead. Loses temporal warm-start within "
             "a batch; use only for offline preprocessing.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device to run torch optim on, kept default as cuda but change to 'cpu' if needed.",
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
