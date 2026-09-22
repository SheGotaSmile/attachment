"""Read-only measurements at the reward-computation boundary, before auto-reset."""

import torch
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def measurements(raw):
    robot = raw.scene['robot']
    velocity = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    command = raw.command_manager.get_command('base_velocity')
    error = (velocity[:, :2] - command[:, :2]).norm(dim=-1)
    return {
        'base_velocity_error': error,
        'velocity_tracking': torch.exp(-error.square() / 0.5**2),
        'forward_velocity': velocity[:, 0],
        'speed_xy': robot.data.root_lin_vel_w[:, :2].norm(dim=-1),
        'terminated': raw.termination_manager.terminated.clone(),
        'truncated': raw.termination_manager.time_outs.clone(),
        'timeout': raw.termination_manager.get_term('time_out').clone(),
        'base_contact': raw.termination_manager.get_term('base_contact').clone(),
    }
