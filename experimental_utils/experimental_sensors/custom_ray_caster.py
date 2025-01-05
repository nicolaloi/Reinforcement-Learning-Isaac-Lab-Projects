# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from omni.isaac.core.prims import XFormPrimView
from pxr import UsdGeom, UsdPhysics

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.terrains.trimesh.utils import make_plane
from omni.isaac.lab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from omni.isaac.lab.utils.warp import convert_to_warp_mesh, raycast_mesh
from omni.isaac.lab.utils import math as math_utils

from omni.isaac.lab.sensors import RayCaster, RayCasterData, SensorBase

if TYPE_CHECKING:
    from .custom_ray_caster_cfg import CustomRayCasterCfg
    



class CustomRayCaster(RayCaster):
    """A ray-casting sensor.
    
    This derived class modifies the _update_buffers_impl method to support
    the attach_yaw_only when a legged robot is standing up.
    """

    cfg: CustomRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: CustomRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrimView):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            quat_base_w_yaw = self._get_quat_base_w_yaw(quat_w)
            ray_starts_w = quat_apply_yaw(quat_base_w_yaw.repeat(1, self.num_rays), self.ray_starts[env_ids])
            # ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids]) # old
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._data.ray_hits_w[env_ids] = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            max_dist=self.cfg.max_distance,
            mesh=self.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]