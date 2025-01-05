# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from dataclasses import MISSING

from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.sensors import RayCasterCfg

from .custom_ray_caster import CustomRayCaster


@configclass
class CustomRayCasterCfg(RayCasterCfg):

    class_type: type = CustomRayCaster