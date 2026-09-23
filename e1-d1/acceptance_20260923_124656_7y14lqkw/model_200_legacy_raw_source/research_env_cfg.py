"""Independent Flat G1 research configuration for E1-D1.

The legacy Flat task is intentionally not modified.  This variant separates
the conservative fall signal (root height/orientation) from contact diagnostics.
Contact pairs remain observable through the ContactSensor/PhysX diagnostic path;
they are not silently discarded or used as a ground-fall proxy.
"""

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import G1FlatEnvCfg


def primary_fall(
    env,
    minimum_height: float = 0.45,
    max_tilt_rad: float = 1.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Terminate only on root collapse or a clearly fallen root orientation.

    This is deliberately independent of the broad legacy contact sensor term.
    Contact object identity is still recorded by the D1 probe and remains an
    unresolved classification when the GPU filter cannot report terrain pairs.
    """
    asset = env.scene[asset_cfg.name]
    height_failed = asset.data.root_pos_w[:, 2] < minimum_height
    # projected gravity z is -cos(tilt); acos(-z) is the tilt from upright.
    tilt_failed = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)) > max_tilt_rad
    return torch.logical_or(height_failed, tilt_failed)


@configclass
class G1ResearchTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    primary_fall = TerminationTermCfg(
        func=primary_fall,
        time_out=False,
        params={"minimum_height": 0.45, "max_tilt_rad": 1.20},
    )


@configclass
class G1ResearchEnvCfg(G1FlatEnvCfg):
    """Independent environment; actions, observations, rewards and physics match Flat G1."""

    terminations: G1ResearchTerminationsCfg = G1ResearchTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "G1ResearchPrimaryFall"


@configclass
class G1ResearchEnvCfg_PLAY(G1ResearchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
