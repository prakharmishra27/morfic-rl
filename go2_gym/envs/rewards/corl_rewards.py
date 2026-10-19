import torch
import numpy as np
from go2_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return 4*torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_jump(self):
        reference_heights = 0
        body_height = self.env.base_pos[:, 2] - reference_heights
        jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)# - reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    def _reward_stability_upright(self):
        """
        Positive reward in [0, 1]:
        1 when upright, -> 0 as tilt increases.
        Works with only_positive_rewards=True.
        """
        # projected_gravity is unit-ish; xy magnitude ~ sin(tilt)
        tilt_sin2 = torch.sum(self.env.projected_gravity[:, :2] ** 2, dim=1)  # ~ sin(tilt)^2

        # Pick a "tolerance" tilt in radians (0.3 rad ~ 17 deg like you wrote)
        theta0 = 0.3
        # convert to the same space (sin^2)
        sigma2 = (torch.sin(torch.tensor(theta0, device=tilt_sin2.device)) ** 2).clamp_min(1e-6)

        # print('statbility rewards ', {torch.exp(-tilt_sin2 / sigma2)})

        # gaussian-ish upright reward
        return torch.exp(-tilt_sin2 / sigma2)
    

    # def _reward_stability_rate_upright(self):
    #     """
    #     Positive reward in [0,1]:
    #     1 when roll/pitch angular rates are small, ->0 when tumbling.
    #     """
    #     # roll/pitch angular velocity components
    #     wxy2 = torch.sum(self.env.base_ang_vel[:, :2] ** 2, dim=1)

    #     # rate tolerance (rad/s). Start ~2.0, tune.
    #     w0 = 2.0
    #     sigma2 = (w0 ** 2)

    #     return torch.exp(-wxy2 / sigma2)


    def _reward_dynamic_stability_cop(self):
        """
        Positive reward in [0, 1]:
        1 = stable CoP close to support center, 0 = unstable.
        Active only when >=2 feet are in contact; otherwise returns 0 (neutral).
        """
        fz = self.env.contact_forces[:, self.env.feet_indices, 2].clamp(min=0.0)  # [N,4]
        in_contact = fz > 1.0                                                    # [N,4]
        nC = in_contact.sum(dim=1)                                               # [N]
        valid = nC >= 2                                                          # [N]

        fp = self.env.foot_positions[:, :, :2]                                   # [N,4,2]

        total_fz = fz.sum(dim=1, keepdim=True).clamp(min=10.0)                   # [N,1]
        cop = (fp * fz.unsqueeze(-1)).sum(dim=1) / total_fz                      # [N,2]

        mask = in_contact.float().unsqueeze(-1)                                  # [N,4,1]
        denom = mask.sum(dim=1).clamp(min=1.0)                                   # [N,1]
        center = (fp * mask).sum(dim=1) / denom                                  # [N,2]

        dev = torch.norm(cop - center, dim=1)                                    # [N]

        tau = 0.10  # meters; start here
        stability_reward = torch.exp(-dev / tau)                                 # [N] in (0,1]

        return torch.where(valid, stability_reward, torch.zeros_like(dev))


    def _reward_feet_contact_stability_width(self):
        """
        Positive reward in [0,1]:
        Rewards not being too narrow when unstable (tilted or fast).
        Width is measured in the robot yaw/body frame (so turning doesn't break it).
        """
        # foot positions relative to base in world
        fp = self.env.foot_positions - self.env.base_pos.unsqueeze(1)  # [N,4,3]

        # --- rotate fp into base yaw frame safely for [N,4,3] ---
        # Build yaw-only quaternion per env (same logic as quat_apply_yaw)
        quat = self.env.base_quat  # [N,4]
        quat_yaw = quat.clone()
        quat_yaw[:, :2] = 0.0  # keep only z,w (yaw)
        quat_yaw = normalize(quat_yaw)

        # Need inverse for world->body rotation
        quat_yaw_inv = quat_conjugate(quat_yaw)  # [N,4]

        N = fp.shape[0]
        fp_flat = fp.reshape(N * 4, 3)                                # [N*4,3]
        quat_rep = quat_yaw_inv.repeat_interleave(4, dim=0)           # [N*4,4]
        fp_body_flat = quat_apply(quat_rep, fp_flat)                  # [N*4,3]
        fp_body = fp_body_flat.view(N, 4, 3)                          # [N,4,3]
        # --------------------------------------------------------

        # left-right width in body frame (y-axis)
        fr_fl = torch.abs(fp_body[:, 0, 1] - fp_body[:, 1, 1])
        rr_rl = torch.abs(fp_body[:, 2, 1] - fp_body[:, 3, 1])
        width = 0.5 * (fr_fl + rr_rl)  # [N]

        # instability signals
        tilt_sin2 = torch.sum(self.env.projected_gravity[:, :2] ** 2, dim=1)  # ~sin(tilt)^2
        speed = torch.norm(self.env.base_lin_vel[:, :2], dim=1)

        # gate: only care when unstable
        gate = ((tilt_sin2 > 0.02) | (speed > 1.5)).float()

        # minimum stance width (meters) when unstable
        w_min = 0.22
        penalty = (w_min - width).clamp(min=0.0)

        # convert penalty -> positive reward in [0,1]
        sigma_w = 0.05
        r = torch.exp(-penalty / sigma_w)

        # neutral when gate=0
        return gate * r + (1.0 - gate) * 1.0
