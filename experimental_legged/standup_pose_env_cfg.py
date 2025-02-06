# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .config.velocity.anymal_c.standup_velocity_rough_env_cfg import AnymalCStandUpVelocityRoughEnvCfg
from . import EXPERIMENTAL_LEGGED_STANDUP_VELOCITY_ROUGH_POLICY_PATH

LOW_LEVEL_ENV_CFG = AnymalCStandUpVelocityRoughEnvCfg()

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    pose_command = mdp.CustomUniformPose2dCommandCfg(
        class_type=mdp.CustomUniformPose2dCommand,
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(7.0, 7.0),
        debug_vis=True,
        z_based_on_terrain=False,
        resample_around_robot=False,
        ranges=mdp.CustomUniformPose2dCommandCfg.Ranges(
            pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    pre_trained_policy_action: mdp.CustomPreTrainedPolicyActionCfg = mdp.CustomPreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=EXPERIMENTAL_LEGGED_STANDUP_VELOCITY_ROUGH_POLICY_PATH,
        low_level_command_dim=2, # y vel and yaw rate
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pose_command = ObsTerm(func=mdp.sliced_generated_commands,
                                    params={"command_name": "pose_command", "idxs": [0,1,3]})

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- pose task
    track_pose = RewTerm(
        func=mdp.track_pose_command_if_feet_up, weight=0.5,
        params={"track_func_reward": mdp.position_command_error_tanh, "in_air_min": 0.5,
                "command_name": "pose_command", "std": math.sqrt(2.0),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*F_FOOT")}
    )
    
    track_pose_fine = RewTerm(
        func=mdp.track_pose_command_if_feet_up, weight=1.0,
        params={"track_func_reward": mdp.position_command_error_tanh, "in_air_min": 0.5,
                "command_name": "pose_command", "std": math.sqrt(0.2),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*F_FOOT")}
    )
    
    track_orientation = RewTerm(
        func=mdp.track_pose_command_if_feet_up, weight=-0.2,
        params={"track_func_reward": mdp.heading_command_error_abs, "in_air_min": 0.5,
                "command_name": "pose_command", "std": None,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*F_FOOT")}
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.15)

    terminated = RewTerm(func=mdp.is_terminated, weight=-25.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    

##
# Environment configuration
##


@configclass
class LocomotionStandUpPoseRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation pose-tracking environment."""

    # Scene settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = LOW_LEVEL_ENV_CFG.events

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        # simulation settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
