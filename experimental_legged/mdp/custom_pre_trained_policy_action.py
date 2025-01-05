# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    
from .commands import CustomUniformPose2dCommandCfg


class CustomPreTrainedPolicyAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: CustomPreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: CustomPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._action_dim = cfg.low_level_command_dim

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions
        
        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_link_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des = torch.zeros((self.num_envs, 2), device=self.device)
        vel_des[:, 1] = self._raw_actions[:, 0]
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(vel_des)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_com_lin_vel_w[:, :2], False)
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

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
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
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
            arrow_quat = math_utils.quat_mul(
                self._get_quat_base_w_yaw(self.robot.data.root_link_quat_w), arrow_quat)
            
        return arrow_scale, arrow_quat


@configclass
class CustomPreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = CustomPreTrainedPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_command_dim: int = MISSING
    """Dimension of the low level command."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
