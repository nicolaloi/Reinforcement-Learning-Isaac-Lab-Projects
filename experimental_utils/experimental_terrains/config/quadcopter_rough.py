# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from typing import Iterable
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from ..trimesh.mesh_terrains_cfg import CustomMeshRepeatedCylindersTerrainCfg

def QUADCOPTER_ROUGH_TERRAINS_CFG_FACTORY(
    size: float, density_min = 0.17, density_max = 0.7, num_rows: int = 15, num_cols: int = 15):
    
    border_width_subterrain = 2.0
    platform_width_subterrain = 0.5
    
    occupied_subterrain_area = (size - 2 * border_width_subterrain) ** 2 - (platform_width_subterrain) ** 2
    num_objects_min = int(density_min * occupied_subterrain_area)
    num_objects_max = int(density_max * occupied_subterrain_area)
    
    return TerrainGeneratorCfg(
        size=(size, size),
        border_width=10.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "repeated_cylinders": CustomMeshRepeatedCylindersTerrainCfg(
                proportion=0.975,
                border_width=border_width_subterrain,
                platform_width=platform_width_subterrain,
                platform_height=0.0,
                object_params_start=CustomMeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=num_objects_min, height=10.0, radius=0.3, max_yx_angle=0.0
                ),
                object_params_end=CustomMeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=num_objects_max, height=12.0, radius=0.06, max_yx_angle=30.0
                ),
            ),
            "empty": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.025,
            )
        },
    )

QUADCOPTER_ROUGH_TERRAINS_CFG = QUADCOPTER_ROUGH_TERRAINS_CFG_FACTORY(25.0)
"""Rough terrains configuration."""
