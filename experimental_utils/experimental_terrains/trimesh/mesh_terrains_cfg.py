# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal, Optional

import omni.isaac.lab.terrains.trimesh.utils as mesh_utils_terrains
from omni.isaac.lab.utils import configclass

from .mesh_terrains import custom_repeated_objects_terrain

from omni.isaac.lab.terrains.terrain_generator_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""

@configclass
class CustomMeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        height: float = MISSING
        """The height (along z) of the object (in m)."""

    function = custom_repeated_objects_terrain

    object_type: Literal["cylinder", "box", "cone"] | callable = MISSING
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """
    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float = 0.0
    """The maximum amount of noise to add to the height of the objects (in m). Defaults to 0.0."""
    platform_width: float = 1.0
    """The width of the platform at the center of the terrain. Defaults to 1.0."""
    platform_height: Optional[float] = None
    """The height of the platform at the center of the terrain. If None,
        the platform height is set to the maximum height of the objects. Defaults to None."""
    border_width: float = 0.0
    """The width of the external border of terrain (in m). Defaults to 0.0.
        The actual width of the surface where objects are placed is the size of the terrain reduced by this value."""


@configclass
class CustomMeshRepeatedPyramidsTerrainCfg(CustomMeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated pyramids."""

    @configclass
    class ObjectCfg(CustomMeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for a curriculum of repeated pyramids."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cone

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""


@configclass
class CustomMeshRepeatedBoxesTerrainCfg(CustomMeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""

    @configclass
    class ObjectCfg(CustomMeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_box

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""


@configclass
class CustomMeshRepeatedCylindersTerrainCfg(CustomMeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated cylinders."""

    @configclass
    class ObjectCfg(CustomMeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated cylinder."""

        radius: float = MISSING
        """The radius of the pyramids (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = mesh_utils_terrains.make_cylinder

    object_params_start: ObjectCfg = MISSING
    """The box curriculum parameters at the start of the curriculum."""
    object_params_end: ObjectCfg = MISSING
    """The box curriculum parameters at the end of the curriculum."""