# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.terrains import trimesh
from omni.isaac.lab.terrains.utils import * # noqa: F401, F403
from omni.isaac.lab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    import mesh_terrains_cfg
    

def custom_repeated_objects_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.CustomMeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects.

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Depending on the object type, the objects are generated with different parameters. The objects
    The types of objects that can be generated are: ``"cylinder"``, ``"box"``, ``"cone"``.

    The object parameters are specified in the configuration as curriculum parameters. The difficulty
    is used to linearly interpolate between the minimum and maximum values of the parameters.

    .. image:: ../../_static/terrains/trimesh/repeated_objects_cylinder_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_box_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_pyramid_terrain.jpg
       :width: 30%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    # import the object functions -- this is done here to avoid circular imports
    from .mesh_terrains_cfg import (
        CustomMeshRepeatedBoxesTerrainCfg,
        CustomMeshRepeatedCylindersTerrainCfg,
        CustomMeshRepeatedPyramidsTerrainCfg,
    )

    # if object type is a string, get the function: make_{object_type}
    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    # -- common parameters
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    if isinstance(cfg, CustomMeshRepeatedBoxesTerrainCfg):
        cp_0: CustomMeshRepeatedBoxesTerrainCfg.ObjectCfg
        cp_1: CustomMeshRepeatedBoxesTerrainCfg.ObjectCfg
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, CustomMeshRepeatedPyramidsTerrainCfg):  # noqa: SIM114
        cp_0: CustomMeshRepeatedPyramidsTerrainCfg.ObjectCfg
        cp_1: CustomMeshRepeatedPyramidsTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    elif isinstance(cfg, CustomMeshRepeatedCylindersTerrainCfg):  # noqa: SIM114
        cp_0: CustomMeshRepeatedCylindersTerrainCfg.ObjectCfg
        cp_1: CustomMeshRepeatedCylindersTerrainCfg.ObjectCfg
        object_kwargs = {
            "radius": cp_0.radius + difficulty * (cp_1.radius - cp_0.radius),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")
    # constants for the terrain
    platform_clearance = 0.1

    # initialize list of meshes
    meshes_list = list()
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * height))
    platform_corners = np.asarray([
        [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
        [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
    ])
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # sample center for objects
    object_centers = np.zeros((num_objects, 3))
    masks_left = np.ones((num_objects,), dtype=bool)
    while np.any(masks_left):
        num_objects_left = masks_left.sum()
        object_centers[masks_left, 0] = np.random.uniform(cfg.border_width, cfg.size[0]-cfg.border_width, num_objects_left)
        object_centers[masks_left, 1] = np.random.uniform(cfg.border_width, cfg.size[1]-cfg.border_width, num_objects_left)
        # filter out the centers that are on the platform
        is_within_platform_x = np.logical_and(
            object_centers[masks_left, 0] >= platform_corners[0, 0],
            object_centers[masks_left, 0] <= platform_corners[1, 0]
        )
        is_within_platform_y = np.logical_and(
            object_centers[masks_left, 1] >= platform_corners[0, 1],
            object_centers[masks_left, 1] <= platform_corners[1, 1]
        )
        masks_left[masks_left] = np.logical_and(is_within_platform_x, is_within_platform_y)

    # generate obstacles (but keep platform clean)
    for index in range(len(object_centers)):
        # randomize the height of the object
        ob_height = height + np.random.uniform(-cfg.max_height_noise, cfg.max_height_noise)
        if ob_height > 0.0:
            object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
            meshes_list.append(object_mesh)

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate a platform in the middle
    # platform height = None produces a platform as the original code
    platform_height = cfg.platform_height if cfg.platform_height is not None else 0.5*height
    dim = (cfg.platform_width, cfg.platform_width, platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    return meshes_list, origin