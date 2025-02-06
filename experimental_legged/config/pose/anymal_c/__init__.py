# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, standup_pose_flat_env_cfg, standup_pose_rough_env_cfg

##
# Register Gym environments.
##

# standup

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Pose-Flat-Anymal-C-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_pose_flat_env_cfg.AnymalCStandUpPoseFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpPoseFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Pose-Flat-Anymal-C-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_pose_flat_env_cfg.AnymalCStandUpPoseFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpPoseFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Pose-Rough-Anymal-C-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_pose_rough_env_cfg.AnymalCStandUpPoseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpPoseRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Pose-Rough-Anymal-C-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_pose_rough_env_cfg.AnymalCStandUpPoseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpPoseRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Pose-Path-Rough-Anymal-C-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_pose_rough_env_cfg.AnymalCStandUpPosePathRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpPoseRoughPPORunnerCfg",
    },
)
