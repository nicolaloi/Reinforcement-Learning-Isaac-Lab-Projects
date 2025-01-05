# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations
import time

import trimesh
import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.math import (
    quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat,
    quat_rotate, quat_from_angle_axis, axis_angle_from_quat)

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import CustomUniformPose2dCommandCfg


class CustomUniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: CustomUniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: CustomUniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        if self.cfg.z_based_on_terrain:
            self.terrain_mesh: trimesh.Trimesh = env.scene.terrain.meshes["terrain"]
        else:
            self.terrain_mesh = None
        self.precomputed_arange_envs = torch.arange(self.num_envs)
        self.segment_len_range = (1.0, 3.0)
        self.segment_rel_orientation_range = (-math.pi/3, math.pi/3)
        self.path_reached_pose_thresh = 0.45

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.path_pos_command_w = torch.zeros(self.num_envs, self.cfg.path_n_poses, 3, device=self.device)
        self.path_heading_command_w = torch.zeros(self.num_envs, self.cfg.path_n_poses, device=self.device)
        self.path_curr_pose_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        self.quat_base_w_yaw = torch.zeros(self.num_envs, 4, device=self.device)
        # -- metrics
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # -- trimesh query
        self.mesh_proximity_query = trimesh.proximity.ProximityQuery(self.terrain_mesh)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
    
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        
        axis_angle_yaw = axis_angle_from_quat(self.quat_base_w_yaw)
        yaws = wrap_to_pi(
            torch.norm(axis_angle_yaw, dim=-1)
        ) * torch.sign(axis_angle_yaw[:, 2])
        
        self.metrics["error_heading"] = (
            torch.abs(wrap_to_pi(self.heading_command_w - yaws + math.pi/2))
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.resample_around_robot:
            self.pos_command_w[env_ids] = self.robot.data.root_pos_w[env_ids]
        else:
            self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        
        if self.cfg.z_based_on_terrain:
            distances, mesh_indices = self.mesh_proximity_query.vertex(
                self.pos_command_w[env_ids].cpu().numpy())
            mesh_points_height = torch.tensor(self.terrain_mesh.vertices[mesh_indices][:, 2],
                                              dtype=self.pos_command_w.dtype, device=self.device)
            self.pos_command_w[env_ids, 2] = mesh_points_height + 1.2
        else:
            self.pos_command_w[env_ids, 2] = self.robot.data.root_pos_w[env_ids, 2] + 0.5

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            axis_angle_yaw = axis_angle_from_quat(self.quat_base_w_yaw)
            yaws = wrap_to_pi(
                torch.norm(axis_angle_yaw, dim=-1)
            ) * torch.sign(axis_angle_yaw[:, 2])
        
            curr_to_target = wrap_to_pi(target_direction - yaws[env_ids] + math.pi/2).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - yaws[env_ids] + math.pi/2).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            
        # create a path (sample multiple poses) if necessary
        if self.cfg.path_n_poses > 1:
            self.path_curr_pose_idx[env_ids] = 0
            self.path_pos_command_w[env_ids, 0] = self.pos_command_w[env_ids]
            self.path_heading_command_w[env_ids, 0] = self.heading_command_w[env_ids]
            for i in range(1, self.cfg.path_n_poses):
                segment_length = r.uniform_(*self.segment_len_range).clone()
                path_direction = wrap_to_pi(self.path_heading_command_w[env_ids, i-1] +
                                            r.uniform_(*self.segment_rel_orientation_range))
                path_vector = torch.zeros(len(env_ids), 3, device=self.device)
                path_vector[:, 0] = torch.cos(path_direction)
                path_vector[:, 1] = torch.sin(path_direction)
                next_pos_command_w = self.path_pos_command_w[env_ids, i-1] + segment_length[:, None]*path_vector
                if self.cfg.z_based_on_terrain:
                    distances, mesh_indices = self.mesh_proximity_query.vertex(
                        next_pos_command_w.cpu().numpy())
                    mesh_points_height = torch.tensor(self.terrain_mesh.vertices[mesh_indices][:, 2],
                                                      dtype=next_pos_command_w.dtype, device=self.device)
                    next_pos_command_w[:, 2] = mesh_points_height + 1.2
                else:
                    next_pos_command_w[:, 2] = self.robot.data.root_pos_w[env_ids, 2] + 0.5
                
                self.path_pos_command_w[env_ids, i] = next_pos_command_w
                self.path_heading_command_w[env_ids, i] = path_direction

    def _update_command(self):
        """Re-target the position command to the current root state."""
        if self.cfg.path_n_poses > 1:
            # check we reached the current target pose
            reached_target = torch.norm(
                self.robot.data.root_pos_w[:, :2] -
                self.path_pos_command_w[self.precomputed_arange_envs, self.path_curr_pose_idx, :2],
                dim=1
            ) < self.path_reached_pose_thresh
            self.path_curr_pose_idx[reached_target] += 1
            self.path_curr_pose_idx = torch.clamp(self.path_curr_pose_idx, 0, self.cfg.path_n_poses-1)
            self.pos_command_w[reached_target] = \
                self.path_pos_command_w[reached_target, self.path_curr_pose_idx[reached_target]]
            self.heading_command_w[reached_target] = \
                self.path_heading_command_w[reached_target, self.path_curr_pose_idx[reached_target]]

        self.quat_base_w_yaw = self._get_quat_base_w_yaw(self.robot.data.root_quat_w)
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(self.quat_base_w_yaw, target_vec)

        axis_angle_yaw = axis_angle_from_quat(self.quat_base_w_yaw)
        yaws = wrap_to_pi(
            torch.norm(axis_angle_yaw, dim=-1)
        ) * torch.sign(axis_angle_yaw[:, 2])
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - yaws + math.pi/2)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "path_visualizers"):
                self.path_visualizers = [VisualizationMarkers(self.cfg.path_visualizer_cfg)
                                        for _ in range(self.cfg.path_n_poses)]
            # set their visibility to true
            for idx in range(len(self.path_visualizers)-1):
                self.path_visualizers[idx].set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "path_visualizers"):
                for idx in range(len(self.path_visualizers)):
                    self.path_visualizers[idx].set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )
        
        if self.cfg.path_n_poses > 1:
            for idx in range(len(self.path_visualizers)):
                marker_indices = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
                marker_indices[idx == self.path_curr_pose_idx] = 0
                marker_indices[idx < self.path_curr_pose_idx] = 2
                self.path_visualizers[idx].visualize(
                    translations=self.path_pos_command_w[:, idx],
                    orientations=quat_from_euler_xyz(
                        torch.zeros_like(self.heading_command_w),
                        torch.zeros_like(self.heading_command_w),
                        self.path_heading_command_w[:, idx],
                    ),
                    marker_indices=marker_indices,
                )
        
    """
    Internal helpers.
    """
    @staticmethod
    @torch.jit.script
    def _get_quat_base_w_yaw(base_quat_w: torch.Tensor) -> torch.Tensor:
        n_envs = base_quat_w.shape[0]
        test_vector_b = torch.tensor([[0.0, 1.0, 0.0]], device=base_quat_w.device, dtype=torch.float32)
        test_vectors_b = test_vector_b.repeat(n_envs, 1)
        test_vectors_w = quat_rotate(base_quat_w, test_vectors_b)
        
        projected_test_vectors_w = test_vectors_w.clone()
        projected_test_vectors_w[:, 2] = 0
        projected_test_vectors_w = projected_test_vectors_w / torch.norm(projected_test_vectors_w, dim=1, keepdim=True)
        
        yaws = torch.atan2(projected_test_vectors_w[:, 1], projected_test_vectors_w[:, 0])
        
        quat_base_w_yaw = quat_from_angle_axis(
            yaws, torch.tensor(
                [0.0, 0.0, 1.0], device=base_quat_w.device, dtype=torch.float32
            ).repeat(n_envs, 1)
        )
        
        return quat_base_w_yaw