# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, standup_velocity_flat_env_cfg, standup_velocity_rough_env_cfg

##
# Register Gym environments.
##

# standup

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Velocity-Flat-Anymal-C-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_velocity_flat_env_cfg.AnymalCStandUpVelocityFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpVelocityFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Velocity-Flat-Anymal-C-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_velocity_flat_env_cfg.AnymalCStandUpVelocityFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpVelocityFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Velocity-Rough-Anymal-C-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_velocity_rough_env_cfg.AnymalCStandUpVelocityRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpVelocityRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Legged-StandUp-Velocity-Rough-Anymal-C-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": standup_velocity_rough_env_cfg.AnymalCStandUpVelocityRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCStandUpVelocityRoughPPORunnerCfg",
    },
)
