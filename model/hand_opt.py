import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np

import torch
import torch.nn.functional as F
from utils.grasp_utils import (
    get_handmodel,
    convert_aligned_to_gripper_pose,
    convert_7dpose_to_4x4,
    convert_4x4_to_7dpose,
    rotation_matrix_from_vectors,
    convert_gripper_to_aligned_pose,
)

from utils.rot6d_utils import robust_compute_rotation_matrix_from_ortho6d


class GcsGraspTransferOpt:

    def __init__(
        self,
        source_robot_name,
        target_robot_name,
        source_grasp_goal=None,
        source_pose_align=None,
        num_particles=32,
        init_rand_scale=0.5,
        learning_rate=5e-3,
        running_name=None,
        energy_func_name="euclidean_dist",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        source_robot_name: str
            source gripper name
        target_robot_name: str
            target gripper name
        source_grasp_goal: (9 + d, ) shape tensor
            source grasp q (9 + d = 3 trans, 6 rot, d joints) vector representing the grasp
        source_pose_align: (7,) numpy array for source grasp pose in aligned frame
            - Single 7-D Numpy Array for Pose with: [position, orientation_quaternion]
            - position_base: length 3 array for the base link's position ; orientation_base: length 3 array for the base link's orientation (x,y,z,w)
        num_particles: int
            number of parallel optimizations to run given the same init
        init_rand_scale: float
            scaling factor for the joint dof limits for initial grasp q during optimization
        learning_rate: float
            learning rate for adam optimization
        running_name: str
            some custom name for the optimization run
        energy_func_name: str
            right now only supports "euclidean_dist" (default)
        device: str
            cuda or cpu device to run the optimization on
        """
        self.running_name = running_name
        self.device = device
        self.target_robot_name = target_robot_name
        self.source_robot_name = source_robot_name

        self.num_particles = num_particles
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate

        self.global_step = None
        self.source_grasp_goal = None
        self.source_pose_align = None
        self.q_current = None
        self.energy = None

        self.compute_energy = None

        self.grp_corr_idxs = None
        # self.q_local = None
        self.optimizer = None

        ########### Init the Source Hand Model #################
        current_fpath = os.path.abspath(__file__)
        parent_dir = os.path.dirname(current_fpath)

        if source_robot_name in {
            "barrett",
            "allegro",
            "ezgripper",
            "shadowhand",
            "robotiq_3finger_gdx",
        }:
            self._source_gripper_json_path = "data/urdf/urdf_assets_meta.json"
            self._source_gripper_datadir = os.path.expanduser("~/Datasets/GenDexGrasp")
        else:
            self._source_gripper_json_path = "urdf_assets_meta.json"
            self._source_gripper_datadir = os.path.join(parent_dir, "../grippers")

        self.source_handmodel = get_handmodel(
            source_robot_name,
            1,
            device,
            hand_scale=1.0,
            json_path=self._source_gripper_json_path,
            datadir=self._source_gripper_datadir,
        )

        ########### Init the Target Hand Model #################
        if target_robot_name in {
            "barrett",
            "allegro",
            "ezgripper",
            "shadowhand",
            "robotiq_3finger_gdx",
        }:
            self._target_gripper_json_path = "data/urdf/urdf_assets_meta.json"
            self._target_gripper_datadir = os.path.expanduser("~/Datasets/GenDexGrasp")
        else:
            self._target_gripper_json_path = "urdf_assets_meta.json"
            self._target_gripper_datadir = os.path.join(parent_dir, "../grippers")

        self.target_handmodel = get_handmodel(
            target_robot_name,
            num_particles,
            device,
            hand_scale=1.0,
            json_path=self._target_gripper_json_path,
            datadir=self._target_gripper_datadir,
        )
        self.q_joint_lower = self.target_handmodel.dynamic_joints_q_lower.detach()
        self.q_joint_upper = self.target_handmodel.dynamic_joints_q_upper.detach()

        # We optimize only for the pose if the gripper is two finger gripper
        self.only_pose_opt = target_robot_name in {
            "fetch_gripper",
            "franka_panda",
            "wsg_50",
            "sawyer",
            "h5_hand",
        }

        if source_grasp_goal is not None:
            assert (
                source_pose_align is not None
            )  # if giving target grasp, also provide its aligned pose
            self.reset(
                source_grasp_goal=source_grasp_goal,
                source_pose_align=source_pose_align,
                running_name=running_name,
                energy_func_name=energy_func_name,
            )

    def reset(
        self,
        source_grasp_goal,
        source_pose_align,
        running_name,
        energy_func_name="euclidean_dist",
    ):

        self.target_handmodel = get_handmodel(
            self.target_robot_name,
            self.num_particles,
            self.device,
            hand_scale=1.0,
            json_path=self._target_gripper_json_path,
            datadir=self._target_gripper_datadir,
        )
        if energy_func_name not in {"euclidean_dist"}:
            raise NotImplementedError

        energy_func_map = {
            "euclidean_dist": self.compute_energy_euclidean_dist,
        }

        self.compute_energy = energy_func_map[energy_func_name]

        self.running_name = running_name
        self.is_pruned = False
        self.best_index = None
        self.global_step = 0
        self.distance_init = 1.0

        self.source_grasp_goal = source_grasp_goal.to(self.device)
        self.source_handmodel.update_kinematics(source_grasp_goal.unsqueeze(0))
        self.source_pose_align = source_pose_align

        # Get correspondence between source and target gripper coords
        # Use this for the closest energy distance computation
        self.source_corr_idxs, self.target_corr_idxs = (
            self.target_handmodel.grasp_transfer_correspondence(
                self.source_handmodel.gripper_coords_all
            )
        )

        # initialize the opt for grasp = (posn, rotn, dof joints)
        q_pose = torch.zeros(self.num_particles, 9, device=self.device)

        # Pose Init --> initialize as the source gripper pose
        palm_pose_7d = convert_aligned_to_gripper_pose(
            self.source_pose_align, self.target_robot_name
        )
        palm_pose_tf = torch.tensor(
            convert_7dpose_to_4x4(palm_pose_7d), device=self.device
        ).float()
        palm_position = palm_pose_tf[:3, 3]
        # ortho6d rotation representation: (x1, x2, x3, y1, y2, y3)
        palm_rotation = palm_pose_tf[:3, :3].transpose(0, 1).reshape(9)[:6]
        q_pose[:, 0:3] = palm_position.repeat(self.num_particles, 1)
        q_pose[:, 3:9] = palm_rotation.repeat(self.num_particles, 1)

        if not self.only_pose_opt:
            # DOFs initialization
            # Set the dof values to be initialized between (lower, lower + range * rand_0_1 * scale)
            self.q_current = torch.zeros(
                self.num_particles,
                3 + 6 + len(self.target_handmodel.dynamic_joints),
                device=self.device,
            )
            self.q_current[:, :9] = q_pose.clone()
            self.q_current[:, 9:] = (
                self.init_random_scale
                * torch.rand_like(self.q_current[:, 9:])
                * (self.q_joint_upper - self.q_joint_lower)
                + self.q_joint_lower
            )
        else:
            self.q_current = torch.zeros(
                self.num_particles,
                9,
                device=self.device,
            )
            self.q_current[:, :9] = q_pose.clone()
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

    def compute_energy_euclidean_dist(self):

        # shape (num_particles, N, 3) -- since there are `num_particles` candidate target grasps
        target_hand_pts = self.target_handmodel.get_surface_points().clone()[
            :,
            self.target_corr_idxs,
        ]
        # shape (N, 3) -- since there is only 1 source grasp
        source_hand_pts = self.source_handmodel.get_surface_points().clone()[
            0,
            self.source_corr_idxs,
        ]

        num_particles = self.num_particles

        ## Compute Eucledian distance between the corresponding hand-object pairs
        ## NOTE: They are already in order!
        npts_source = source_hand_pts.size()[0]  # note: shapes are different!
        npts_target = target_hand_pts.size()[1]
        assert npts_target == npts_source  # required due to our correspondence

        batch_source_point_cloud = source_hand_pts.unsqueeze(0).repeat(
            num_particles, 1, 1
        )  # shape (N,3) -> (B, N, 3)

        hand_obj_dist = (target_hand_pts - batch_source_point_cloud).norm(
            dim=2
        )  # shape (B, N)
        energy_contact = hand_obj_dist.mean(dim=1)  # shape (B, )

        energy_penetration = 0
        energy = energy_contact
        self.energy = energy

        if not self.only_pose_opt:
            # TODO: add a normalized energy?
            z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(
                self.q_joint_lower - self.q_current[:, 9:]
            )
            z_energy = z_norm.sum(dim=1)
            self.energy = energy + z_energy

        return energy

    def step(self):
        self.optimizer.zero_grad()
        if self.only_pose_opt:
            # since q_current is actually just the grasp pose, we also need some default dofs to update the kinematics
            sample_dofs = torch.zeros(
                self.num_particles, len(self.target_handmodel.dynamic_joints)
            ).to(self.device)

            self.target_handmodel.update_kinematics(
                q=torch.cat((self.q_current, sample_dofs), dim=1)
            )
        else:
            self.target_handmodel.update_kinematics(q=self.q_current)

        energy = self.compute_energy()
        energy.mean().backward()
        self.optimizer.step()
        self.global_step += 1

    def get_opt_q(self):
        return self.q_current.detach()

    def set_opt_q(self, opt_q):
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_plotly_data(self, index=0, color="lightblue", opacity=1):
        return self.target_handmodel.get_plotly_data(
            q=self.q_current, i=index, color=color, opacity=opacity
        )


class HandObjectGraspOpt:

    def __init__(
        self,
        target_robot_name,
        source_grasp=None,
        num_particles=32,
        init_rand_scale=0.5,
        learning_rate=5e-3,
        running_name=None,
        energy_func_name="euclidean_dist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        collision_weight=100,
        contact_weight=1,
        opt_only_trans=True,
        sharp_factor=20,
    ):
        """
        source_robot_name: str
            source gripper name
        target_robot_name: str
            target gripper name
        source_grasp_goal: (9 + d, ) shape tensor
            source grasp q (9 + d = 3 trans, 6 rot, d joints) vector representing the grasp
        source_pose_align: (7,) numpy array for source grasp pose in aligned frame
            - Single 7-D Numpy Array for Pose with: [position, orientation_quaternion]
            - position_base: length 3 array for the base link's position ; orientation_base: length 3 array for the base link's orientation (x,y,z,w)
        num_particles: int
            number of parallel optimizations to run given the same init
        init_rand_scale: float
            scaling factor for the joint dof limits for initial grasp q during optimization
        learning_rate: float
            learning rate for adam optimization
        running_name: str
            some custom name for the optimization run
        energy_func_name: str
            right now only supports "euclidean_dist" (default)
        device: str
            cuda or cpu device to run the optimization on
        """
        self.running_name = running_name
        self.device = device
        self.target_robot_name = target_robot_name

        self.num_particles = num_particles
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate

        self.global_step = None
        self.source_grasp = None
        self.q_current = None
        self.energy = None
        self.collision_weight = collision_weight
        self.contact_weight = contact_weight
        self.compute_energy = None
        self.opt_only_trans = opt_only_trans
        self.sharp_factor = sharp_factor

        self.grp_corr_idxs = None
        # self.q_local = None
        self.optimizer = None

        current_fpath = os.path.abspath(__file__)
        parent_dir = os.path.dirname(current_fpath)

        ########### Init the Target Hand Model #################
        if target_robot_name in {
            "barrett",
            "allegro",
            "ezgripper",
            "shadowhand",
            "robotiq_3finger_gdx",
        }:
            self._target_gripper_json_path = "data/urdf/urdf_assets_meta.json"
            self._target_gripper_datadir = os.path.expanduser("~/Datasets/GenDexGrasp")
        else:
            self._target_gripper_json_path = "urdf_assets_meta.json"
            self._target_gripper_datadir = os.path.join(parent_dir, "../grippers")

        self.target_handmodel = get_handmodel(
            target_robot_name,
            num_particles,
            device,
            hand_scale=1.0,
            json_path=self._target_gripper_json_path,
            datadir=self._target_gripper_datadir,
        )
        self.q_joint_lower = self.target_handmodel.dynamic_joints_q_lower.detach()
        self.q_joint_upper = self.target_handmodel.dynamic_joints_q_upper.detach()

        # We optimize only for the pose if the gripper is two finger gripper
        self.opt_only_pose = target_robot_name in {
            "fetch_gripper",
            "franka_panda",
            "wsg_50",
            "sawyer",
            "h5_hand",
        }

        if source_grasp is not None:
            self.reset(
                source_grasp=source_grasp,
                running_name=running_name,
                energy_func_name=energy_func_name,
            )

    def reset(
        self,
        contact_map_goal,
        source_grasp,
        running_name,
        energy_func_name="euclidean_dist",
    ):

        self.target_handmodel = get_handmodel(
            self.target_robot_name,
            self.num_particles,
            self.device,
            hand_scale=1.0,
            json_path=self._target_gripper_json_path,
            datadir=self._target_gripper_datadir,
        )
        if energy_func_name not in {"euclidean_dist", "align_dist"}:
            raise NotImplementedError

        energy_func_map = {
            "euclidean_dist": self.compute_energy_euclidean_dist,
            "align_dist": self.compute_energy_align_dist,
        }

        self.compute_energy = energy_func_map[energy_func_name]

        self.running_name = running_name
        self.is_pruned = False
        self.best_index = None
        self.global_step = 0
        self.distance_init = 1.0

        self.contact_map_goal = contact_map_goal.to(self.device)
        self.object_point_cloud = contact_map_goal[:, :3].to(self.device)
        self.object_normal_cloud = contact_map_goal[:, 3:6].to(self.device)
        self.contact_value_goal = contact_map_goal[:, 6].to(self.device)
        self.object_radius = torch.max(torch.norm(self.object_point_cloud, dim=1, p=2))

        self.source_grasp = source_grasp.detach().clone().to(self.device)
        # Pose Init --> initialize as the source pose
        # initialize the opt for grasp = (posn, rotn, dof joints)
        q_pose = torch.zeros(self.num_particles, 9, device=self.device)
        q_pose[:, 0:3] = (
            self.source_grasp[0:3].detach().clone().repeat(self.num_particles, 1)
        )
        q_pose[:, 3:9] = (
            self.source_grasp[3:9].detach().clone().repeat(self.num_particles, 1)
        )

        # randomly perturb the translation
        perturbation_scale = 5e-3
        noise = torch.randn_like(q_pose[:, :3]) * perturbation_scale
        q_pose[:, :3] += noise

        if not self.opt_only_pose:
            # DOFs initialization
            # Set the dof values to be initialized between (lower, lower + range * rand_0_1 * scale)
            self.q_current = torch.zeros(
                self.num_particles,
                3 + 6 + len(self.target_handmodel.dynamic_joints),
                device=self.device,
            )
            self.q_current[:, :9] = q_pose.detach().clone()
            self.q_current[:, 9:] = (
                self.init_random_scale
                * torch.rand_like(self.q_current[:, 9:])
                * (self.q_joint_upper - self.q_joint_lower)
                + self.q_joint_lower
            )
        else:
            self.q_current = torch.zeros(
                self.num_particles,
                9,
                device=self.device,
            )
            self.q_current[:, :9] = q_pose.detach().clone()
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

    def compute_energy_euclidean_dist(self):
        hand_mesh_points = self.target_handmodel.get_fullmesh_points()
        hand_mesh_points_ = self.target_handmodel.get_fullmesh_points().clone()
        hand_surface_points = self.target_handmodel.get_surface_points()
        # hand_surface_points_ = hand_surface_points.detach().clone()

        npts_object = self.object_point_cloud.size()[0]
        npts_hand_surf = hand_surface_points.size()[1]
        npts_hand_mesh = hand_mesh_points.size()[1]

        ############ Compute Contact Value ##################
        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
            self.num_particles, 1, 1
        )
        batch_object_point_cloud = batch_object_point_cloud.reshape(
            self.num_particles, 1, npts_object, 3
        )
        hand_surface_points = hand_surface_points.reshape(
            self.num_particles, 1, npts_hand_surf, 3
        )
        batch_object_point_cloud = batch_object_point_cloud.repeat(
            1, npts_hand_surf, 1, 1
        ).transpose(1, 2)
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1)

        object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)

        contact_dist = object_hand_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (
            torch.sigmoid(self.sharp_factor * contact_dist) - 0.5
        )
        energy_contact = torch.abs(
            contact_value_current - self.contact_value_goal.reshape(1, -1)
        ).mean(dim=1)
        # energy_contact = 0

        ############## Compute Collision Cost #################

        # TODO
        # 1. Use all gripper mesh points (not just the surface points?)
        # 2. Use full object/scene point cloud for collision?

        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
            self.num_particles, 1, 1
        )
        batch_object_point_cloud = batch_object_point_cloud.reshape(
            self.num_particles, 1, npts_object, 3
        )
        hand_mesh_points = hand_mesh_points_.reshape(
            self.num_particles, 1, npts_hand_mesh, 3
        )
        hand_mesh_points = hand_mesh_points.repeat(1, npts_object, 1, 1).transpose(1, 2)

        batch_object_point_cloud = batch_object_point_cloud.repeat(
            1, npts_hand_mesh, 1, 1
        )

        hand_object_dist = (hand_mesh_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack(
            [self.object_point_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_normal = torch.stack(
            [self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_signs = (
            (hand_object_points - hand_mesh_points_) * hand_object_normal
        ).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)

        ######################### Compute Total Energy #################################
        energy = (
            self.contact_weight * energy_contact
            + self.collision_weight * energy_penetration
        )

        if not self.opt_only_pose:
            # TODO: add a normalized energy?
            z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(
                self.q_joint_lower - self.q_current[:, 9:]
            )
            z_energy = z_norm.sum(dim=1)
            energy += z_energy

        self.energy = energy
        return energy

    def compute_energy_align_dist(self):

        # hand_mesh_points = self.target_handmodel.get_fullmesh_points()
        hand_mesh_points_ = self.target_handmodel.get_fullmesh_points().clone()

        hand_surface_points = self.target_handmodel.get_surface_points()
        hand_surface_points_ = hand_surface_points.clone()

        npts_object = self.object_point_cloud.size()[0]
        npts_hand = hand_surface_points.size()[1]
        npts_hand_mesh = hand_mesh_points_.size()[1]

        ############ Compute Contact Value ##################
        with torch.no_grad():
            batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
                self.num_particles, 1, 1
            )
            batch_object_point_cloud = batch_object_point_cloud.view(
                self.num_particles, 1, npts_object, 3
            )
            batch_object_point_cloud = batch_object_point_cloud.repeat(
                1, npts_hand, 1, 1
            ).transpose(1, 2)

        hand_surface_points = hand_surface_points.view(
            self.num_particles, 1, npts_hand, 3
        )
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1)

        with torch.no_grad():
            batch_object_normal_cloud = self.object_normal_cloud.unsqueeze(0).repeat(
                self.num_particles, 1, 1
            )
            batch_object_normal_cloud = batch_object_normal_cloud.view(
                self.num_particles, 1, npts_object, 3
            )
            batch_object_normal_cloud = batch_object_normal_cloud.repeat(
                1, npts_hand, 1, 1
            ).transpose(1, 2)

        object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)

        # Compute Alignment
        object_hand_align = (
            (hand_surface_points - batch_object_point_cloud) * batch_object_normal_cloud
        ).sum(dim=3)
        object_hand_align /= object_hand_dist + 1e-5

        object_hand_align_dist = object_hand_dist * torch.exp(
            2 * (1 - object_hand_align)
        )

        contact_dist = torch.sqrt(object_hand_align_dist.min(dim=2)[0])
        # contact_dist = object_hand_align_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (
            torch.sigmoid(self.sharp_factor * contact_dist) - 0.5
        )
        energy_contact = torch.abs(
            contact_value_current - self.contact_value_goal.view(1, -1)
        ).mean(dim=1)

        ############## Compute Collision Cost #################
        # TODO
        # 1. Use all gripper mesh points (not just the surface points?)
        # 2. Use full object/scene point cloud for collision?

        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
            self.num_particles, 1, 1
        )
        batch_object_point_cloud = batch_object_point_cloud.reshape(
            self.num_particles, 1, npts_object, 3
        )

        combined_hand_pts = torch.cat((hand_surface_points_, hand_mesh_points_), dim=1)
        npts_hand_mesh = combined_hand_pts.shape[1]
        hand_mesh_points = combined_hand_pts.reshape(self.num_particles, 1, -1, 3)

        hand_mesh_points = hand_mesh_points.repeat(1, npts_object, 1, 1).transpose(1, 2)

        batch_object_point_cloud = batch_object_point_cloud.repeat(
            1, npts_hand_mesh, 1, 1
        )

        hand_object_dist = (hand_mesh_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack(
            [self.object_point_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_normal = torch.stack(
            [self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_signs = (
            (hand_object_points - combined_hand_pts) * hand_object_normal
        ).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)

        ######################### Compute Total Energy #################################
        energy = (
            self.contact_weight * energy_contact
            + self.collision_weight * energy_penetration
        )

        if not self.opt_only_pose:
            # TODO: add a normalized energy?
            z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(
                self.q_joint_lower - self.q_current[:, 9:]
            )
            z_energy = z_norm.sum(dim=1)
            energy += z_energy

        self.energy = energy
        return energy

    def step(self):
        self.optimizer.zero_grad()

        if self.opt_only_pose:
            # since q_current is actually just the grasp pose, we also need some default dofs to update the kinematics
            sample_dofs = torch.zeros(
                self.num_particles, len(self.target_handmodel.dynamic_joints)
            ).to(self.device)

            if self.opt_only_trans:
                grasp_q = torch.cat(
                    [
                        self.q_current[:, :3],
                        self.source_grasp[3:9]
                        .detach()
                        .clone()
                        .repeat(self.num_particles, 1),
                        sample_dofs,
                    ],
                    dim=1,
                )
            else:
                grasp_q = torch.cat((self.q_current, sample_dofs), dim=1)

            self.target_handmodel.update_kinematics(q=grasp_q)
        else:
            self.target_handmodel.update_kinematics(q=self.q_current)

        energy = self.compute_energy()
        energy.mean().backward()
        self.optimizer.step()
        self.global_step += 1

    def get_opt_q(self):
        return self.q_current.detach()

    def set_opt_q(self, opt_q):
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_plotly_data(self, index=0, color="lightblue", opacity=1):
        return self.target_handmodel.get_plotly_data(
            q=self.q_current, i=index, color=color, opacity=opacity
        )


class AdamGraspTransfer:
    def __init__(
        self,
        source_robot_name,
        target_robot_name,
        source_grasp_goal=None,
        num_particles=32,
        init_rand_scale=0.5,
        max_iter=300,
        steps_per_iter=2,
        learning_rate=5e-3,
        device="cuda",
        energy_func_name="euclidean_dist",
        writer=None,
    ):
        self.writer = writer
        self.target_robot_name = target_robot_name
        self.source_robot_name = source_robot_name
        self.source_grasp_goal = source_grasp_goal
        self.num_particles = num_particles
        self.init_rand_scale = init_rand_scale
        self.learning_rate = learning_rate
        self.device = device
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter
        self.energy_func_name = energy_func_name

        self.opt_model = GcsGraspTransferOpt(
            source_robot_name,
            target_robot_name,
            source_grasp_goal=None,
            source_pose_align=None,
            num_particles=self.num_particles,
            init_rand_scale=init_rand_scale,
            learning_rate=learning_rate,
            energy_func_name=self.energy_func_name,
            device=device,
        )

    def run_adam(self, source_grasp_goal, running_name, source_pose_align=None):

        if not source_pose_align:
            # compute the aligned source pose here:
            grasp_tra_3d = source_grasp_goal[:3].detach().cpu().numpy()
            grasp_orn_6d = source_grasp_goal[3:9]
            grasp_rotmat = (
                robust_compute_rotation_matrix_from_ortho6d(grasp_orn_6d.unsqueeze(0))
                .detach()
                .cpu()
                .numpy()
            )
            pose_tf = np.eye(4)
            pose_tf[:3, :3] = grasp_rotmat
            pose_tf[:3, 3] = grasp_tra_3d
            pose_7d = convert_4x4_to_7dpose(pose_tf)
            source_pose_align = convert_gripper_to_aligned_pose(
                pose_7d, self.source_robot_name
            )

        q_trajectory = []

        self.opt_model.reset(
            source_grasp_goal, source_pose_align, running_name, self.energy_func_name
        )

        with torch.no_grad():
            opt_q = self.opt_model.get_opt_q()
            q_trajectory.append(opt_q.clone().detach())

        iters_per_print = self.max_iter // 2

        # for i_iter in tqdm(range(self.max_iter), desc=f"{running_name}"):
        for i_iter in range(self.max_iter):

            self.opt_model.step()

            with torch.no_grad():
                opt_q = self.opt_model.get_opt_q()
                q_trajectory.append(opt_q.clone().detach())

            # if i_iter % iters_per_print == 0 or i_iter == self.max_iter - 1:
            #     print(f"min energy: {self.opt_model.energy.min(dim=0)[0]:.4f}")
            #     print(f"min energy index: {self.opt_model.energy.min(dim=0)[1]}")

            with torch.no_grad():
                energy = self.opt_model.energy.detach().cpu().tolist()
                tag_scaler_dict = {
                    f"{i_energy}": energy[i_energy] for i_energy in range(len(energy))
                }
                if self.writer is not None:
                    self.writer.add_scalars(
                        main_tag=f"energy/{running_name}",
                        tag_scalar_dict=tag_scaler_dict,
                        global_step=i_iter,
                    )
                    self.writer.add_scalar(
                        tag=f"index/{running_name}",
                        scalar_value=energy.index(min(energy)),
                        global_step=i_iter,
                    )
        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1)
        return (
            q_trajectory,
            self.opt_model.energy.detach().cpu().clone(),
            self.steps_per_iter,
        )


class AdamGraspCmap:
    def __init__(
        self,
        target_robot_name,
        collision_weight,
        contact_weight,
        opt_only_trans,
        sharp_factor=20,
        source_grasp=None,
        num_particles=32,
        init_rand_scale=0.5,
        max_iter=300,
        steps_per_iter=2,
        learning_rate=5e-3,
        device="cuda",
        energy_func_name="euclidean_dist",
        writer=None,
    ):
        self.writer = writer
        self.target_robot_name = target_robot_name
        self.source_grasp = source_grasp
        self.num_particles = num_particles
        self.init_rand_scale = init_rand_scale
        self.learning_rate = learning_rate
        self.device = device
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter
        self.energy_func_name = energy_func_name

        self.opt_model = HandObjectGraspOpt(
            target_robot_name,
            source_grasp=None,
            num_particles=self.num_particles,
            init_rand_scale=init_rand_scale,
            learning_rate=learning_rate,
            energy_func_name=self.energy_func_name,
            device=device,
            collision_weight=collision_weight,
            contact_weight=contact_weight,
            opt_only_trans=opt_only_trans,
        )

    def run_adam(self, contact_map_goal, source_grasp, running_name):

        q_trajectory = []

        self.opt_model.reset(
            contact_map_goal, source_grasp, running_name, self.energy_func_name
        )

        with torch.no_grad():
            opt_q = self.opt_model.get_opt_q()
            q_trajectory.append(opt_q.detach().clone().detach())

        iters_per_print = self.max_iter // 2

        for i_iter in tqdm(range(self.max_iter), desc=f"opt_iters"):
            # for i_iter in range(self.max_iter):

            self.opt_model.step()

            with torch.no_grad():
                opt_q = self.opt_model.get_opt_q()
                q_trajectory.append(opt_q.detach().clone().detach())

            # if i_iter % iters_per_print == 0 or i_iter == self.max_iter - 1:
            #     print(f"min energy: {self.opt_model.energy.min(dim=0)[0]:.4f}")
            #     print(f"min energy index: {self.opt_model.energy.min(dim=0)[1]}")

            with torch.no_grad():
                energy = self.opt_model.energy.detach().cpu().tolist()
                tag_scaler_dict = {
                    f"{i_energy}": energy[i_energy] for i_energy in range(len(energy))
                }
                if self.writer is not None:
                    self.writer.add_scalars(
                        main_tag=f"energy/{running_name}",
                        tag_scalar_dict=tag_scaler_dict,
                        global_step=i_iter,
                    )
                    self.writer.add_scalar(
                        tag=f"index/{running_name}",
                        scalar_value=energy.index(min(energy)),
                        global_step=i_iter,
                    )
        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1)
        return (
            q_trajectory,
            self.opt_model.energy.detach().clone().cpu(),
            self.steps_per_iter,
        )


if __name__ == "__main__":
    # TODO: Add tests here
    pass
