# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.experimental.experimental_legged.standup_pose_env_cfg import LocomotionStandUpPoseRoughEnvCfg
import omni.isaac.lab_tasks.experimental.experimental_legged.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class AnymalCStandUpPoseRoughEnvCfg(LocomotionStandUpPoseRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-c
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class AnymalCStandUpPoseRoughEnvCfg_PLAY(AnymalCStandUpPoseRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
@configclass
class AnymalCStandUpPosePathRoughEnvCfg_PLAY(AnymalCStandUpPoseRoughEnvCfg_PLAY):
    """ Simple modification of the PLAY environment to generate 
        a path instead of individual poses"""
    def __post_init__(self):
        
        # modify command generation to create a path
        self.commands.pose_command = \
            mdp.CustomUniformPose2dCommandCfg(
                class_type=mdp.CustomUniformPose2dCommand,
                asset_name="robot",
                simple_heading=True,
                resampling_time_range=(15.0, 15.0),
                debug_vis=True,
                z_based_on_terrain=True,
                resample_around_robot=True,
                ranges=mdp.CustomUniformPose2dCommandCfg.Ranges(
                    pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-3.14159, 3.14159)
                ),
                path_n_poses=7,
            )
        
        # post init of parent
        super().__post_init__()
        
        self.episode_length_s = 30.0
