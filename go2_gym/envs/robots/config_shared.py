"""Shared robot configuration defaults.

Apply before robot-specific settings and train/play overrides.
"""
from .morphology import config_morphology


def config_shared(cfg):
    # Control defaults; robot configs override gains and action scaling.
    config_morphology(cfg)
    cfg.control.control_type = "P"
    cfg.control.action_scale = 0.25
    cfg.control.hip_scale_reduction = 0.5
    cfg.control.decimation = 4

    # Asset defaults; paths, body names, and joint names are robot-specific.
    cfg.asset.self_collisions = 0
    cfg.asset.fix_base_link = False

    cfg.rewards.soft_dof_pos_limit = 0.9

    # Reward defaults; preserve any robot-specific overrides.
    cfg.reward_scales.torques = -0.0001
    cfg.reward_scales.action_rate = -0.01
    cfg.reward_scales.dof_pos_limits = -10.0
    cfg.reward_scales.orientation = -5.0
    cfg.reward_scales.base_height = -30.0

    # Terrain
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.measure_heights = False
    cfg.terrain.terrain_noise_magnitude = 0.0
    cfg.terrain.teleport_robots = True
    cfg.terrain.border_size = 50
    cfg.terrain.terrain_proportions = [
        0, 0, 0, 0, 0, 0, 0, 0, 1.0
    ]
    cfg.terrain.curriculum = False

    # Environment defaults; training scripts may override these.
    cfg.env.num_observations = 42
    cfg.env.observe_vel = False
    cfg.env.num_envs = 4000

    # Commands: retain the final assigned values from the original config.
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 10.0
    cfg.commands.command_curriculum = True
    cfg.commands.num_lin_vel_bins = 30
    cfg.commands.num_ang_vel_bins = 30
    cfg.commands.lin_vel_x = [-0.6, 0.6]
    cfg.commands.lin_vel_y = [-0.6, 0.6]
    cfg.commands.ang_vel_yaw = [-1.0, 1.0]

    # Domain randomization defaults
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.added_mass_range = [-1, 3]

    cfg.domain_rand.push_robots = False
    cfg.domain_rand.max_push_vel_xy = 0.5

    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [0.05, 5.0]

    cfg.domain_rand.randomize_restitution = True
    cfg.domain_rand.restitution_range = [0.0, 1.0]
    cfg.domain_rand.restitution = 0.5

    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.com_displacement_range = [-0.1, 0.1]

    cfg.domain_rand.randomize_motor_strength = True
    cfg.domain_rand.motor_strength_range = [0.9, 1.3]

    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.Kp_factor_range = [0.8, 1.3]

    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.Kd_factor_range = [0.5, 1.5]

    cfg.domain_rand.rand_interval_s = 6