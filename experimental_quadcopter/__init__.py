# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym
from pathlib import Path

from . import agents
from .quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg, QuadcopterEnvCfg_PLAY

##
# Register Gym environments.
##

experimental_quadcopter_parent_folder = str(Path(__file__).parent.parent)

gym.register(
    id="Isaac-Experimental-Quadcopter-Forest-Pose-Direct-v0",
    entry_point=f"omni.isaac.lab_tasks.{experimental_quadcopter_parent_folder}.experimental_quadcopter:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Experimental-Quadcopter-Forest-Pose-Direct-Play-v0",
    entry_point=f"omni.isaac.lab_tasks.{experimental_quadcopter_parent_folder}.experimental_quadcopter:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)
