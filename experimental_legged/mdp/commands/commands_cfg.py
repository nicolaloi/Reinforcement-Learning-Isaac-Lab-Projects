# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import RED_ARROW_X_MARKER_CFG
from omni.isaac.lab.envs.mdp.commands import UniformVelocityCommandCfg, UniformPose2dCommandCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from .velocity_command import CustomUniformVelocityCommand
from .pose_2d_command import CustomUniformPose2dCommand

PATH_ARROWS_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "current_target": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.2, 0.2, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "future_target": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.2, 0.2, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.5)),
        ),
        "past_target": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.2, 0.2, 0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            visible=False,
        ),
    }
)

@configclass
class CustomUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    
    class_type: type = CustomUniformVelocityCommand

    heading_target_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_goal"
    )
    
@configclass
class CustomUniformPose2dCommandCfg(UniformPose2dCommandCfg):
    
    class_type: type = CustomUniformPose2dCommand
    
    path_n_poses: int = 1
    """Number of poses to sample along the path.
    If equal to 1, nothing happens differently from the original
    UniformPose2dCommand. If greater than 1, multiples pose2d commands
    are sampled to form a path. When the robot's position is close
    to the current target pose, the next pose in the path
    becomes the new target pose. Used for path following task examples."""
    
    resample_around_robot: bool = False
    """Whether to resample the position around the robot.
    If False, the position is sampled around the environment origin."""
    
    z_based_on_terrain: bool = False
    """Whether to set the z pose command based on the terrain.
    This is time consuming, used only for visualization."""
    
    path_visualizer_cfg: VisualizationMarkersCfg = PATH_ARROWS_MARKER_CFG.replace(
        prim_path="/Visuals/Command/path_goal"
    )
    """The configuration for the path visualization marker.
    Makes sense only if path_n_poses > 1."""