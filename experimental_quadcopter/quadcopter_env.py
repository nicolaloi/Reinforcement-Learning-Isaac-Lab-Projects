# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms, random_yaw_orientation
from omni.isaac.lab.utils.noise import gaussian_noise

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

from ..experimental_utils.experimental_terrains import QUADCOPTER_ROUGH_TERRAINS_CFG_FACTORY



class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@configclass
class EventCfg:
    """Configuration for randomization."""

    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (0.975, 1.025),
            "operation": "scale",
        },
    )

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 7.5
    dt= 1 / 50
    decimation = 2
    action_space = 4
    observation_space = 59
    state_space = 0
    debug_vis = True
    size_terrain = 25.0
    objects_density_min = 0.17
    objects_density_max = 0.7
    random_respawn = False
    avoid_reset_spikes_in_training = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=QUADCOPTER_ROUGH_TERRAINS_CFG_FACTORY(
            size=size_terrain, density_min=objects_density_min, density_max=objects_density_max),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    
    # change viewer settings
    viewer = ViewerCfg(eye=(18.0, 18.0, 18.0), lookat=(6.5, 6.5, 0), resolution=(1280, 720))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    
    #events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot").\
                                           replace(spawn = CRAZYFLIE_CFG.spawn.replace(activate_contact_sensors=True))
    simple_lidar = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period= dt * decimation,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.01)),
        max_distance=15,
        attach_yaw_only=False,
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(-0.0, 0.0), 
                                             horizontal_fov_range=(-90.0, 90.0),
                                             horizontal_res=3.99),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    scaling_lidar_data_b = 1/6.0
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=2, update_period= dt, track_air_time=False
    )
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True
    # )
    thrust_to_weight = 1.9
    thrust_scale = 0.75
    moment_scale = 0.01
    
    # task
    desired_pos_w_height_limits = (2.0, 3.0)
    desired_pos_b_xy_limits = (size_terrain/2-1.0, size_terrain/2)
    desired_pos_b_obs_clip = 4.0
    height_w_limits = (0.5, 4.5)
    lin_vel_max_soft_thresh = 0
    ang_vel_final_dist_goal_thresh = 0.3
    progress_to_goal_std = math.sqrt(0.1)
    distance_to_goal_std = math.sqrt(1.25)
    distance_to_goal_fine_std = math.sqrt(0.3)
    threshold_obstacle_proximity = 0.4
    threshold_height_bounds_proximity = 0.3
    height_w_soft_limits = (
        height_w_limits[0] + threshold_height_bounds_proximity,
        height_w_limits[1] - threshold_height_bounds_proximity)

    # reward scales
    lin_vel_reward_scale = -0.018
    ang_vel_reward_scale = -0.06
    ang_vel_final_reward_scale = -0.2
    actions_reward_scale = -0.2
    progress_to_goal_reward_scale = 2.0
    distance_to_goal_reward_scale = 1.0
    distance_to_goal_fine_reward_scale = 0.7
    undesired_contacts_reward_scale = -4.0
    flat_orientation_reward_scale = -1.0
    obstacle_proximity_reward_scale = -6.0
    height_bounds_proximity_reward_scale = -4.0
    terminated_reward_scale = -200.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "ang_vel_final",
                "actions",
                "progress_to_goal",
                "distance_to_goal",
                "distance_to_goal_fine",
                "undesired_contacts",
                "flat_orientation",
                "obstacle_proximity",
                "height_bounds_proximity",
                "terminated",
            ]
        }
        # Get specific body indices
        self._body_id, _ = self._robot.find_bodies("body")
        self._undesired_contact_body_ids = SceneEntityCfg("contact_sensor", body_names=".*").body_ids
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._simple_lidar = RayCaster(self.cfg.simple_lidar)
        self.scene.sensors["simple_lidar"] = self._simple_lidar

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-2.0, 2.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.cfg.thrust_scale * self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self.prev_pos_w = self._robot.data.root_link_pos_w.clone()
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        noisy_desired_pos_b = self._add_uniform_noise(desired_pos_b, -0.02, 0.02)
        noisy_desired_pos_b = (noisy_desired_pos_b / self.cfg.desired_pos_b_obs_clip).clip(-1.0, 1.0)
        height = self._robot.data.root_state_w[:, 2].unsqueeze(1)
        noisy_height = self._add_uniform_noise(height, -0.02, 0.02)
        noisy_height /= self.cfg.height_w_limits[1]
        # create a self._simple_lidar.data.ray_hits_w.shape tensor and store repeated root state
        root_state_w = self._robot.data.root_state_w.unsqueeze(1).repeat(1, self._simple_lidar.data.ray_hits_w.shape[1], 1)
        simple_lidar_data_ray_hits_b, _ = subtract_frame_transforms(
            root_state_w[..., :3], root_state_w[..., 3:7], self._simple_lidar.data.ray_hits_w)
        # obtain the distance to the lidar hits
        simple_lidar_data_ray_hits_b = torch.nan_to_num(simple_lidar_data_ray_hits_b, nan=float('inf'))
        simple_lidar_data_b = torch.norm(simple_lidar_data_ray_hits_b, dim=-1)
        noisy_simple_lidar_data_b = self._add_uniform_noise(simple_lidar_data_b, -0.02, 0.02)
        noisy_simple_lidar_data_b = (self.cfg.scaling_lidar_data_b * noisy_simple_lidar_data_b).clip(-1.0, 1.0)
        
        noisy_root_lin_vel = self._add_uniform_noise(self._robot.data.root_com_lin_vel_b, -0.2, 0.2)
        noisy_root_ang_vel = self._add_uniform_noise(self._robot.data.root_com_ang_vel_b, -0.1, 0.1)
        noisy_projected_gravity_b = self._add_uniform_noise(self._robot.data.projected_gravity_b, -0.03, 0.03)
        # add uniform noise
        obs = torch.cat(
            [
                noisy_root_lin_vel,
                noisy_root_ang_vel,
                noisy_projected_gravity_b,
                noisy_height,
                noisy_desired_pos_b,
                noisy_simple_lidar_data_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # base velocities
        root_lin_vel_processed = self._robot.data.root_com_lin_vel_b.clone()
        root_lin_vel_processed[:, 0] *= torch.where(root_lin_vel_processed[:, 0] < -1.0, 2, 1)
        root_lin_vel_processed[:, 1] *= torch.where(torch.abs(root_lin_vel_processed[:, 1]) > 2.0, 2, 1)
        lin_vel = torch.sum(torch.square(root_lin_vel_processed), dim=1)
        lin_vel = torch.where(torch.torch.linalg.norm(root_lin_vel_processed, dim=1) > self.cfg.lin_vel_max_soft_thresh,
                              lin_vel + self.cfg.lin_vel_max_soft_thresh**2 - 2*(lin_vel**0.5)*self.cfg.lin_vel_max_soft_thresh,
                              torch.zeros_like(lin_vel))
        ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)
        # actions
        actions = torch.sum(torch.abs(self._actions), dim=1)
        # distance to goal task
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        
        prev_distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.prev_pos_w, dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        
        progress_to_goal_mapped = -torch.tanh((distance_to_goal-prev_distance_to_goal) / (self.cfg.progress_to_goal_std**2))
        
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / (self.cfg.distance_to_goal_std**2))
        distance_to_goal_mapped_fine = 1 - torch.tanh(distance_to_goal / (self.cfg.distance_to_goal_fine_std**2))
        
        desired_pos_b_clipped = (desired_pos_b / self.cfg.desired_pos_b_obs_clip).clip(-1.0, 1.0)
        # close_to_goal True for every env for which the desired_pos_b_clipped is within the limits
        close_to_goal = torch.logical_and(torch.all(-0.99 < desired_pos_b_clipped, dim=1), torch.all(desired_pos_b_clipped < 0.99, dim=1))
        distance_to_goal_mapped = torch.where(
            close_to_goal, distance_to_goal_mapped, torch.zeros_like(distance_to_goal_mapped))
        distance_to_goal_mapped_fine =torch.where(
            close_to_goal, distance_to_goal_mapped_fine, torch.zeros_like(distance_to_goal_mapped_fine))
        # undesired contacts
        ang_vel_final = torch.where(distance_to_goal < self.cfg.ang_vel_final_dist_goal_thresh, ang_vel, torch.zeros_like(ang_vel))
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 0.01
        )
        undesired_contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # too close to obstacles
        root_state_w = self._robot.data.root_state_w.unsqueeze(1).repeat(1, self._simple_lidar.data.ray_hits_w.shape[1], 1)
        simple_lidar_data_ray_hits_b, _ = subtract_frame_transforms(
            root_state_w[..., :3],root_state_w[..., 3:7], self._simple_lidar.data.ray_hits_w)
        # obtain the distance to the lidar hits
        simple_lidar_data_ray_hits_b = torch.nan_to_num(simple_lidar_data_ray_hits_b, nan=float('inf'))
        simple_lidar_data_b = torch.norm(simple_lidar_data_ray_hits_b, dim=-1)
        obstacle_proximity = torch.max(torch.where(simple_lidar_data_b < self.cfg.threshold_obstacle_proximity,
                                         self.cfg.threshold_obstacle_proximity - simple_lidar_data_b,
                                         torch.zeros_like(simple_lidar_data_b)), dim=1)[0]
        # too close to height bounds
        height_low_bound_proximity = torch.where(self._robot.data.root_state_w[:, 2] < self.cfg.height_w_soft_limits[0],
                                                self.cfg.height_w_soft_limits[0] - self._robot.data.root_state_w[:, 2],
                                                torch.zeros_like(self._robot.data.root_state_w[:, 2]))
        height_high_bound_proximity = torch.where(self._robot.data.root_state_w[:, 2] > self.cfg.height_w_soft_limits[1],
                                                self._robot.data.root_state_w[:, 2] - self.cfg.height_w_soft_limits[1],
                                                torch.zeros_like(self._robot.data.root_state_w[:, 2]))
        height_bounds_proximity = torch.max(height_low_bound_proximity, height_high_bound_proximity)
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "ang_vel_final": ang_vel_final * self.cfg.ang_vel_final_reward_scale * self.step_dt,
            "actions": actions * self.cfg.actions_reward_scale * self.step_dt,
            "progress_to_goal": progress_to_goal_mapped * self.cfg.progress_to_goal_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "distance_to_goal_fine": distance_to_goal_mapped_fine * self.cfg.distance_to_goal_fine_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contacts_reward_scale * self.step_dt,
            "flat_orientation": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "height_bounds_proximity": height_bounds_proximity * self.cfg.height_bounds_proximity_reward_scale * self.step_dt,
            "terminated": self.reset_terminated * self.cfg.terminated_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_height_bounds = torch.logical_or(
                self._robot.data.root_link_pos_w[:, 2] < self.cfg.height_w_limits[0],
                self._robot.data.root_link_pos_w[:, 2] > self.cfg.height_w_limits[1])
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collided = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids],
                                           dim=-1), dim=1)[0] > 0.0001, dim=1)
        died = torch.logical_or(out_of_height_bounds, collided)
        return died, time_out
        
    def _add_uniform_noise(self, data: torch.Tensor, min_noise: float, max_noise: float):
        return data + torch.rand_like(data) * (max_noise - min_noise) + min_noise
    
    def _sample_points_square_hole(self, n: int, side_length: float, hole_length: float):
        
        indices = torch.arange(n, device=self.device)
        shuffled_indices = indices[torch.randperm(n)]
        # Calculate the size of each split
        split_sizes = [n // 4] * 4
        for i in range(n % 4):
            split_sizes[i] += 1  # Distribute the remainder among the first splits
        
        # Split the shuffled tensor into 4 parts
        idx_splits = torch.split(shuffled_indices, split_sizes)
        
        points = torch.empty(n, 2, device=self.device)
        # Half sizes for convenience
        half_s, half_h = side_length / 2, hole_length / 2

        # Sampling from each region
        # Region 1: Top rectangle
        points[idx_splits[0]] = torch.stack([
            torch.empty(len(idx_splits[0]), device=self.device).uniform_(-half_h, half_s),       # x-coordinates
            torch.empty(len(idx_splits[0]), device=self.device).uniform_(half_h, half_s)         # y-coordinates (above the hole)
        ], dim=1)
        
        # Region 2: Bottom rectangle
        points[idx_splits[1]] = torch.stack([
            torch.empty(len(idx_splits[1]), device=self.device).uniform_(-half_s, half_h),
            torch.empty(len(idx_splits[1]), device=self.device).uniform_(-half_s, -half_h)       # y-coordinates (below the hole)
        ], dim=1)
        
        # Region 3: Left rectangle
        points[idx_splits[2]] = torch.stack([
            torch.empty(len(idx_splits[2]), device=self.device).uniform_(-half_s, -half_h),      # x-coordinates (left of the hole)
            torch.empty(len(idx_splits[2]), device=self.device).uniform_(-half_h, half_s)        # y-coordinates
        ], dim=1)
        
        # Region 4: Right rectangle
        points[idx_splits[3]] = torch.stack([
            torch.empty(len(idx_splits[3]), device=self.device).uniform_(half_h, half_s), # x-coordinates (right of the hole)
            torch.empty(len(idx_splits[3]), device=self.device).uniform_(-half_s, half_h)
        ], dim=1)
        
        return points

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        distances_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        )
        final_distance_to_goal = distances_to_goal.mean()
        final_distance_to_goal_10_percentile = torch.quantile(distances_to_goal, 0.10, interpolation="linear")
        final_distance_to_goal_90_percentile = torch.quantile(distances_to_goal, 0.90, interpolation="linear")
        idxs = torch.argwhere(distances_to_goal <= final_distance_to_goal_10_percentile + 1e-4).flatten()
        final_ang_vel_distance_to_goal_10_percentile = torch.abs(self._robot.data.root_com_ang_vel_b[idxs]).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/final_distance_to_goal_10_percentile"] = final_distance_to_goal_10_percentile.item()
        extras["Metrics/final_distance_to_goal_90_percentile"] = final_distance_to_goal_90_percentile.item()
        extras["Metrics/final_ang_vel_goal_10_percentile"] = final_ang_vel_distance_to_goal_10_percentile.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.cfg.avoid_reset_spikes_in_training and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = self._sample_points_square_hole(len(env_ids),
            2*self.cfg.desired_pos_b_xy_limits[1], 2*self.cfg.desired_pos_b_xy_limits[0])
        if self.cfg.random_respawn:
            rand_terrain_value = torch.rand(len(env_ids), device=self.device)
            move_up = rand_terrain_value < 0.33
            move_down = rand_terrain_value > 0.66
            # update terrain levels
            self._terrain.update_env_origins(env_ids, move_up, move_down)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = self._desired_pos_w[env_ids, 2].uniform_(
                self.cfg.desired_pos_w_height_limits[0], self.cfg.desired_pos_w_height_limits[1])
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 2] = (2*torch.rand_like(default_root_state[:, 2])-1) * \
            0.25*(self.cfg.height_w_limits[1] - self.cfg.height_w_limits[0]) + \
                (self.cfg.height_w_limits[1] - self.cfg.height_w_limits[0]) / 2.0
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        # apply random yaw
        default_root_state[:, 3:7] = random_yaw_orientation(default_root_state.shape[0], device=self.device)
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


class QuadcopterEnvCfg_PLAY(QuadcopterEnvCfg):
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
    
        self.episode_length_s *= 4
        self.size_terrain *= 4
        self.objects_density_min *= 1
        self.objects_density_max *= 0.8
        self.desired_pos_b_xy_limits = (self.size_terrain/2-1.0, self.size_terrain/2)
        self.random_respawn = True
        self.avoid_reset_spikes_in_training = False
        
        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=QUADCOPTER_ROUGH_TERRAINS_CFG_FACTORY(
                size=self.size_terrain, density_min=self.objects_density_min,
                density_max=self.objects_density_max, num_rows=5, num_cols=5),
            max_init_terrain_level=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            ),
            debug_vis=False,
        )