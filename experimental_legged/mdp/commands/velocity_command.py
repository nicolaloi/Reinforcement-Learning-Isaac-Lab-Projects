# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import CustomUniformVelocityCommandCfg

class CustomUniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the base frame.

    """

    cfg: CustomUniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: CustomUniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.quat_base_w_yaw = torch.zeros(self.num_envs, 4, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)
                
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired velocity command in the world frame. Shape is (num_envs, 3)."""
        return self.vel_command_b
    
    @property
    def vel_command_w_with_base_yaw(self) -> torch.Tensor:
        return math_utils.quat_rotate(
            self.quat_base_w_yaw,
            torch.cat([self.vel_command_b[:, :2], torch.zeros_like(self.vel_command_b[:, 0:1])], dim=1))
    

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        # convert velocity back from base to world frame (only yaw)
        
        # extract the used quantities (to enable type-hinting)
        # compute the reward    
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_w_with_base_yaw[:, :2] - 
                self.robot.data.root_lin_vel_w[:, :2], dim=-1
            ) / max_command_step
        )
        
        self.metrics["error_vel_yaw"] += (
            # using root_ang_vel_b[:, 1] because robot is standing up
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 1]) / max_command_step
        ) 
        
        axis_angle_yaw = math_utils.axis_angle_from_quat(self.quat_base_w_yaw)
        yaws = math_utils.wrap_to_pi(
            torch.norm(axis_angle_yaw, dim=-1)
        ) * torch.sign(axis_angle_yaw[:, 2])
        
        self.metrics["error_yaw"] += (
            torch.abs(math_utils.wrap_to_pi(self.heading_target - yaws + math.pi/2)) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        r2 = torch.empty(len(env_ids), device=self.device)        
        
        # -- linear velocity - x direction
        if isinstance(self.cfg.ranges.lin_vel_x[0], (int, float)):
            self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        else:
            rand = torch.rand(len(env_ids), device=self.device)
            self.vel_command_b[env_ids, 0] = torch.where(
                rand < 0.5, r.uniform_(*self.cfg.ranges.lin_vel_x[0]), r2.uniform_(*self.cfg.ranges.lin_vel_x[1])
            )
        # -- linear velocity - y direction
        if isinstance(self.cfg.ranges.lin_vel_y[0], (int, float)):
            self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        else:
            rand = torch.rand(len(env_ids), device=self.device)
            self.vel_command_b[env_ids, 1] = torch.where(
                rand < 0.5, r.uniform_(*self.cfg.ranges.lin_vel_y[0]), r2.uniform_(*self.cfg.ranges.lin_vel_y[1])
            )
        # -- ang vel yaw - rotation around z (in reality y when standing up)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = math_utils.wrap_to_pi(r.uniform_(*self.cfg.ranges.heading))
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments.
        """
        
        self.quat_base_w_yaw = self._get_quat_base_w_yaw(self.robot.data.root_quat_w)
        
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            
            axis_angle_yaw = math_utils.axis_angle_from_quat(self.quat_base_w_yaw)
            yaws = math_utils.wrap_to_pi(
                torch.norm(axis_angle_yaw, dim=-1)
            ) * torch.sign(axis_angle_yaw[:, 2])
            
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - yaws[env_ids] + math.pi/2)
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                # target heading
                if self.cfg.heading_command:
                    self.target_heading_visualizer = VisualizationMarkers(self.cfg.heading_target_visualizer_cfg)
                    self.target_heading_visualizer.set_visibility(True)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
                if self.cfg.heading_command:
                    self.target_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_w[:, :2], False)
        if self.cfg.heading_command:
            pseudo_vel = torch.zeros_like(self.robot.data.root_lin_vel_w)
            pseudo_vel[:, 0] = torch.cos(self.heading_target)
            pseudo_vel[:, 1] = torch.sin(self.heading_target)
            pseudo_vel[self.is_standing_env] = 0.0
            target_heading_arrow_scale, target_heading_arrow_quat = self._resolve_xy_velocity_to_arrow(
                pseudo_vel[:, :2], False
            )
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        if self.cfg.heading_command:
            heading_arrow_pos_w = self.robot.data.root_pos_w.clone()
            heading_arrow_pos_w[:, 2] += 0.65
            self.target_heading_visualizer.visualize(heading_arrow_pos_w, target_heading_arrow_quat, target_heading_arrow_scale)

    """
    Internal helpers.
    """
    @staticmethod
    @torch.jit.script
    def _get_quat_base_w_yaw(base_quat_w: torch.Tensor) -> torch.Tensor:
        n_envs = base_quat_w.shape[0]
        test_vector_b = torch.tensor([[0.0, 1.0, 0.0]], device=base_quat_w.device, dtype=torch.float32)
        test_vectors_b = test_vector_b.repeat(n_envs, 1)
        test_vectors_w = math_utils.quat_rotate(base_quat_w, test_vectors_b)
        
        projected_test_vectors_w = test_vectors_w.clone()
        projected_test_vectors_w[:, 2] = 0
        projected_test_vectors_w = projected_test_vectors_w / torch.norm(projected_test_vectors_w, dim=1, keepdim=True)
        
        yaws = torch.atan2(projected_test_vectors_w[:, 1], projected_test_vectors_w[:, 0])
        
        quat_base_w_yaw = math_utils.quat_from_angle_axis(
            yaws, torch.tensor(
                [0.0, 0.0, 1.0], device=base_quat_w.device, dtype=torch.float32
            ).repeat(n_envs, 1)
        )
        
        return quat_base_w_yaw

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, to_world: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        
        # arrow-direction
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # convert everything back from base to world frame
        if to_world:
            arrow_quat = math_utils.quat_mul(self.quat_base_w_yaw, arrow_quat)
            
        return arrow_scale, arrow_quat