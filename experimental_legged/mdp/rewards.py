# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations
from collections.abc import Callable

import torch
from typing import TYPE_CHECKING, List, Optional

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat

from omni.isaac.lab.assets import RigidObject

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_flying_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward the feet to stay indefinitely in the air.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((current_air_time - threshold), dim=1)
    return reward

def stable_feet_standing_up(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, value: float,
    sensor_cfg_condition: SceneEntityCfg, in_air_max: float
) -> torch.Tensor:
    """Reward all feet to stay on the ground until another contact sensor
        detached for at least in_air_max seconds.
    """
    # extract the used quantities (to enable type-hinting)
    front_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg_condition.name]
    # compute the reward
    front_current_air_time = front_contact_sensor.data.current_air_time[:, sensor_cfg_condition.body_ids]
    front_in_air = front_current_air_time > in_air_max
    front_all_in_air = torch.sum(front_in_air.int(), dim=1) == len(sensor_cfg_condition.body_ids)
    
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    slack_not_in_air = current_air_time < 0.2
    in_air = current_air_time > 0
    slack_all_not_in_air = torch.sum(slack_not_in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    all_in_air = torch.sum(in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    all_not_in_air = torch.where(all_in_air, torch.zeros_like(slack_all_not_in_air), slack_all_not_in_air)
    reward = torch.min(torch.where(all_not_in_air.unsqueeze(-1), value, 0.0), dim=1)[0]
    reward *= front_all_in_air == 0 # reward only if front feet are not in air

    return reward

def all_feet_flying_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
    in_air_min: float, not_in_air_penalty: float, threshold: float
) -> torch.Tensor:
    """Reward all the feet to stay indefinitely in the air at the same time.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_air = current_air_time > in_air_min
    in_mode_time = torch.where(in_air, current_air_time, -current_contact_time)
    all_in_air = torch.sum(in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    reward = torch.min(torch.where(all_in_air.unsqueeze(-1), in_mode_time, not_in_air_penalty), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    return reward
    
    
def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_gait_walk_standing_up(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg,
    sensor_cfg_condition: SceneEntityCfg, in_air_min: float = 0.0
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    Reward is also zero if another contact sensor is not detached for at least in_air_min seconds.
    """
    front_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg_condition.name]
    front_current_air_time = front_contact_sensor.data.current_air_time[:, sensor_cfg_condition.body_ids]
    front_in_air = front_current_air_time > in_air_min
    front_all_in_air = torch.sum(front_in_air.int(), dim=1) == len(sensor_cfg_condition.body_ids)
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    if command_name is not None:
        reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= front_all_in_air == 1 # reward only if front feet are in air
    return reward

# when anymal body is rotated along its direction of motion axis (a normal roll), the projected gravity vector is:
# tensor([[ 0.0410, -0.9699]])
       
def flat_roll_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the y-components (roll) of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])
    

def flat_pitch_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the x-components (pitch) of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 0])

def flat_pitch_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation.

    This is computed by penalizing the x-components (pitch) of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 0]


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_world_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ang_vel_thresh: float = 0.25, scale: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_command_w_with_base_yaw = env.command_manager.get_term(command_name).vel_command_w_with_base_yaw
    lin_vel_error = torch.sum(
        torch.square(vel_command_w_with_base_yaw[:, :2] - asset.data.root_lin_vel_w[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # if the angular velocity command is too large, it means the robot is not correctly aligned to the heading target
    # in this case, the reward to track the linear velocity is reduced, to force the robot to align first
    reward = torch.where(env.command_manager.get_command(command_name)[:, 2] < ang_vel_thresh, reward, reward*scale)
    return reward


def track_vel_command_if_feet_up(
    env, track_func_reward: Callable, in_air_min: float, std: float, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Reward tracking of linear velocity commands (xy axes) if both front feet are up."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_air = current_air_time > in_air_min
    all_in_air = torch.sum(in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    reward = track_func_reward(env, std, command_name, asset_cfg)
    reward = torch.where(all_in_air, reward, torch.zeros_like(reward))
    return reward


def track_ang_vel_z_world_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 1])
    return torch.exp(-ang_vel_error / std**2)


def track_pose_command_if_feet_up(
    env, track_func_reward: Callable, in_air_min: float, std: Optional[float], command_name: str, sensor_cfg: SceneEntityCfg,
):
    """Reward tracking of position commands (xy axes) if both front feet are up."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_air = current_air_time > in_air_min
    all_in_air = torch.sum(in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    if std is not None:
        reward = track_func_reward(env, std, command_name)
    else:
        reward = track_func_reward(env, command_name)
    reward = torch.where(all_in_air, reward, torch.zeros_like(reward))
    return reward

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def joint_penalty_if_feet_up(
    env, joint_func_penalty: Callable, in_air_min: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    slack: float | None = None
):
    """Penalize joint positions if both front feet are up."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_air = current_air_time > in_air_min
    all_in_air = torch.sum(in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    if slack is None:
        reward = joint_func_penalty(env, asset_cfg)
    else:
        reward = joint_func_penalty(env, asset_cfg, slack)
    reward = torch.where(all_in_air, reward, torch.zeros_like(reward))
    return reward

def undesired_contacts_if_feet_up(
    env, undesidered_contacts_func: Callable, in_air_min: float, sensor_cfg: SceneEntityCfg, threshold: float
):
    """Penalize undesidered contacts if both front feet are up."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_air = current_air_time > in_air_min
    all_in_air = torch.sum(in_air.int(), dim=1) == len(sensor_cfg.body_ids)
    reward = undesidered_contacts_func(env, threshold, sensor_cfg)
    reward = torch.where(all_in_air, reward, torch.zeros_like(reward))
    return reward


def conditional_reward(
    env, condition_func: Callable, condition_threshold: float, func_reward: Callable, *args
) -> torch.Tensor:
    """Compute a reward based on a condition.

    This function computes a reward based on a condition. If the condition is not met, the reward is zero.
    """
    condition = condition_func(env, condition_threshold)
    reward = func_reward(env, *args)
    return torch.where(condition, reward, torch.zeros_like(reward))


def sliced_generated_commands(env: ManagerBasedRLEnv, command_name: str, idxs: List[int] | None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    command =  env.command_manager.get_command(command_name)
    if idxs is not None:
        return command[:, idxs]
    else:
        return command