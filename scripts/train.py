#!/usr/bin/env python3
"""Train MorFiC or an internal PPO baseline."""

import argparse
import random
from datetime import datetime
from pathlib import Path

# Isaac Gym must be imported before Torch.
import isaacgym
assert isaacgym

import numpy as np
import torch
from ml_logger import logger

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.robots import ROBOT_CONFIGS, configure_robot
from go2_gym.envs.robots.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper

from go2_gym_learn.ppo_cse import Runner, RunnerArgs
from go2_gym_learn.ppo_cse.actor_critic import AC_Args, VARIANTS
from go2_gym_learn.ppo_cse.ppo import PPO_Args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--robot",
        choices=tuple(ROBOT_CONFIGS),
        default="go2",
        help="Source robot template. Paper experiments train on Go2.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="morfic_film",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=4000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--save-interval", type=int, default=5
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=100,
        help="Video interval in iterations; 0 disables recording.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(MINI_GYM_ROOT_DIR) / "runs",
    )

    args = parser.parse_args()

    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive.")
    if args.save_interval < 0 or args.video_interval < 0:
        parser.error("Save/video intervals must be nonnegative.")

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_training(cfg):
    """Apply the training overrides from the original train.py."""

    cfg.commands.num_lin_vel_bins = 30
    cfg.commands.num_ang_vel_bins = 30

    cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    cfg.commands.distributional_commands = True

    cfg.domain_rand.lag_timesteps = 6
    cfg.domain_rand.randomize_lag_timesteps = True
    cfg.control.control_type = "actuator_net"

    cfg.domain_rand.randomize_rigids_after_start = False

    cfg.env.priv_observe_motion = False
    cfg.env.priv_observe_gravity_transformed_motion = False

    cfg.domain_rand.randomize_friction_indep = False
    cfg.env.priv_observe_friction_indep = False
    cfg.domain_rand.randomize_friction = True
    cfg.env.priv_observe_friction = True
    cfg.domain_rand.friction_range = [0.1, 3.0]

    cfg.domain_rand.randomize_restitution = True
    cfg.env.priv_observe_restitution = True
    cfg.domain_rand.restitution_range = [0.0, 0.4]

    cfg.domain_rand.randomize_base_mass = True
    cfg.env.priv_observe_base_mass = False
    cfg.domain_rand.added_mass_range = [-1.0, 3.0]

    cfg.domain_rand.randomize_gravity = True
    cfg.domain_rand.gravity_range = [-1.0, 1.0]
    cfg.domain_rand.gravity_rand_interval_s = 8.0
    cfg.domain_rand.gravity_impulse_duration = 0.99
    cfg.env.priv_observe_gravity = False

    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    cfg.env.priv_observe_com_displacement = False

    cfg.domain_rand.randomize_ground_friction = True
    cfg.env.priv_observe_ground_friction = False
    cfg.env.priv_observe_ground_friction_per_foot = False
    cfg.domain_rand.ground_friction_range = [0.0, 0.0]

    cfg.domain_rand.randomize_motor_strength = True
    cfg.domain_rand.motor_strength_range = [0.9, 1.3]
    cfg.env.priv_observe_motor_strength = False

    cfg.domain_rand.randomize_motor_offset = True
    cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    cfg.env.priv_observe_motor_offset = False

    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.env.priv_observe_Kp_factor = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.env.priv_observe_Kd_factor = False

    cfg.env.priv_observe_body_velocity = False
    cfg.env.priv_observe_body_height = False
    cfg.env.priv_observe_desired_contact_states = False
    cfg.env.priv_observe_contact_forces = False
    cfg.env.priv_observe_foot_displacement = False
    cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    cfg.env.num_privileged_obs = 2
    cfg.env.num_observation_history = 30
    cfg.reward_scales.feet_contact_forces = 0.0

    cfg.domain_rand.rand_interval_s = 4
    cfg.commands.num_commands = 15
    cfg.env.observe_two_prev_actions = True
    cfg.env.observe_yaw = False
    cfg.env.num_observations = 70
    cfg.env.num_scalar_observations = 70
    cfg.env.observe_gait_commands = True
    cfg.env.observe_timing_parameter = False
    cfg.env.observe_clock_inputs = True

    cfg.domain_rand.tile_height_range = [0.0, 0.0]
    cfg.domain_rand.tile_height_curriculum = False
    cfg.domain_rand.tile_height_update_interval = 1000000
    cfg.domain_rand.tile_height_curriculum_step = 0.01

    cfg.terrain.border_size = 0.0
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.num_cols = 30
    cfg.terrain.num_rows = 30
    cfg.terrain.terrain_width = 1.0
    cfg.terrain.terrain_length = 1.0
    cfg.terrain.x_init_range = 0.2
    cfg.terrain.y_init_range = 0.2
    cfg.terrain.teleport_thresh = 0.3
    cfg.terrain.teleport_robots = False
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 4
    cfg.terrain.horizontal_scale = 0.10

    cfg.rewards.use_terminal_foot_height = False
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.terminal_body_height = 0.05
    cfg.rewards.use_terminal_roll_pitch = True
    cfg.rewards.terminal_body_ori = 1.6
    cfg.rewards.stability_upright = 1.0

    cfg.commands.resampling_time = 10

    cfg.reward_scales.feet_slip = -0.04
    cfg.reward_scales.action_smoothness_1 = -0.1
    cfg.reward_scales.action_smoothness_2 = -0.1
    cfg.reward_scales.dof_vel = -1e-4
    cfg.reward_scales.dof_pos = 0.0
    cfg.reward_scales.jump = 10.0
    cfg.reward_scales.base_height = 0.0
    cfg.rewards.base_height_target = 0.30
    cfg.reward_scales.estimation_bonus = 0.0
    cfg.reward_scales.raibert_heuristic = -10.0
    cfg.reward_scales.feet_impact_vel = 0.0
    cfg.reward_scales.feet_clearance = 0.0
    cfg.reward_scales.feet_clearance_cmd = 0.0
    cfg.reward_scales.feet_clearance_cmd_linear = -30.0
    cfg.reward_scales.orientation = 0.0
    cfg.reward_scales.orientation_control = -5.0
    cfg.reward_scales.tracking_stance_width = 0.0
    cfg.reward_scales.tracking_stance_length = 0.0
    cfg.reward_scales.lin_vel_z = -0.02
    cfg.reward_scales.ang_vel_xy = -0.001
    cfg.reward_scales.feet_air_time = 0.0
    cfg.reward_scales.hop_symmetry = 0.0

    cfg.rewards.kappa_gait_probs = 0.07
    cfg.rewards.gait_force_sigma = 100.0
    cfg.rewards.gait_vel_sigma = 10.0

    cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    cfg.reward_scales.collision = -5.0
    cfg.reward_scales.stability_upright = 0.2

    cfg.rewards.reward_container_name = "CoRLRewards"
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.only_positive_rewards_ji22_style = True
    cfg.rewards.sigma_rew_neg = 0.02

    cfg.commands.lin_vel_x = [-1.0, 1.0]
    cfg.commands.lin_vel_y = [-0.6, 0.6]
    cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    cfg.commands.body_height_cmd = [-0.25, 0.15]

    cfg.commands.gait_frequency_cmd_range = [2.0, 4.0]
    cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    cfg.commands.gait_duration_cmd_range = [0.5, 0.5]

    cfg.commands.footswing_height_range = [0.03, 0.35]
    cfg.commands.body_pitch_range = [-0.4, 0.4]
    cfg.commands.body_roll_range = [0.0, 0.0]
    cfg.commands.stance_width_range = [0.10, 0.45]
    cfg.commands.stance_length_range = [0.35, 0.45]

    cfg.commands.limit_vel_x = [-5.0, 5.0]
    cfg.commands.limit_vel_y = [-0.6, 0.6]
    cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    cfg.commands.limit_body_height = [-0.25, 0.15]
    cfg.commands.limit_gait_frequency = [2.0, 4.0]
    cfg.commands.limit_gait_phase = [0.0, 1.0]
    cfg.commands.limit_gait_offset = [0.0, 1.0]
    cfg.commands.limit_gait_bound = [0.0, 1.0]
    cfg.commands.limit_gait_duration = [0.5, 0.5]
    cfg.commands.limit_footswing_height = [0.03, 0.35]
    cfg.commands.limit_body_pitch = [-0.4, 0.4]
    cfg.commands.limit_body_roll = [0.0, 0.0]
    cfg.commands.limit_stance_width = [0.10, 0.45]
    cfg.commands.limit_stance_length = [0.35, 0.45]

    cfg.commands.num_bins_vel_x = 21
    cfg.commands.num_bins_vel_y = 1
    cfg.commands.num_bins_vel_yaw = 21
    cfg.commands.num_bins_body_height = 1
    cfg.commands.num_bins_gait_frequency = 1
    cfg.commands.num_bins_gait_phase = 1
    cfg.commands.num_bins_gait_offset = 1
    cfg.commands.num_bins_gait_bound = 1
    cfg.commands.num_bins_gait_duration = 1
    cfg.commands.num_bins_footswing_height = 1
    cfg.commands.num_bins_body_roll = 1
    cfg.commands.num_bins_body_pitch = 1
    cfg.commands.num_bins_stance_width = 1

    cfg.normalization.friction_range = [0, 1]
    cfg.normalization.ground_friction_range = [0, 1]
    cfg.normalization.clip_actions = 10.0
    cfg.terrain.yaw_init_range = 3.14

    cfg.commands.exclusive_phase_offset = False
    cfg.commands.pacing_offset = False
    cfg.commands.binary_phases = True
    cfg.commands.gaitwise_curricula = True


def configure_logger(args):
    timestamp = datetime.now().strftime("%Y-%m-%d/%H%M%S_%f")
    run_name = (
        f"morfic/{args.robot}/{args.variant}/"
        f"seed_{args.seed}/{timestamp}"
    )

    logger.configure(
        run_name,
        root=str(args.run_root.expanduser().resolve()),
    )

    logger.log_text(
        """
        charts:
        - yKey: train/episode/rew_total/mean
          xKey: iterations
        - yKey: train/episode/rew_tracking_lin_vel/mean
          xKey: iterations
        - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
          xKey: iterations
        - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
          xKey: iterations
        - yKey: mean_value_loss/mean
          xKey: iterations
        - yKey: adaptation_loss/mean
          xKey: iterations
        - type: video
          glob: "videos/*.mp4"
        """,
        filename=".charts.yml",
        dedent=True,
    )



def check_morphology(env):
    """Check Go2 morphology readback and persistence across explicit resets."""
    import numpy as np

    ids = [0, 1, 2]
    ids = [i for i in ids if i < env.num_envs]
    env_ids = torch.tensor(ids, device=env.device, dtype=torch.long)

    vector_before = env.morphology_vector.detach().clone()
    raw = (
        vector_before * (env.morph_max - env.morph_min)
        + env.morph_min
    ).cpu().numpy()

    if vector_before.shape != (env.num_envs, 11):
        raise AssertionError("Expected morphology shape [num_envs, 11].")
    if not torch.isfinite(vector_before).all().item():
        raise AssertionError("Nonfinite morphology vector.")
    if len(ids) > 1 and torch.equal(vector_before[0], vector_before[1]):
        raise AssertionError("First two environments have identical samples.")

    def read_and_check(stage):
        snapshots = {}

        for env_id in ids:
            handle = env.actor_handles[env_id]
            sim_env = env.envs[env_id]
            bodies = env.gym.get_actor_rigid_body_properties(sim_env, handle)
            dofs = env.gym.get_actor_dof_properties(sim_env, handle)
            descriptor = raw[env_id]

            mass_indices = {
                "thigh": 0, "calf": 1, "hip": 2,
                "foot": 3, "base": 4,
            }
            matched_groups = set()

            for index, name in enumerate(env.body_names):
                # Match the precedence used by apply_morphology().
                for group, morph_index in mass_indices.items():
                    if group in name:
                        np.testing.assert_allclose(
                            bodies[index].mass,
                            descriptor[morph_index],
                            rtol=1e-5,
                            atol=1e-6,
                            err_msg=f"{stage}: env {env_id}, body {name}",
                        )
                        matched_groups.add(group)
                        break

            for required in ("thigh", "calf", "hip", "base"):
                if required not in matched_groups:
                    raise AssertionError(
                        f"No {required} body matched; check body naming."
                    )

            if len(dofs) != len(env.dof_names):
                raise AssertionError("DOF names/properties length mismatch.")

            for index, name in enumerate(env.dof_names):
                group = next(
                    (g for g in ("hip", "thigh", "calf") if g in name),
                    None,
                )
                if group is None:
                    raise AssertionError(f"Unrecognized Go2 joint: {name}")

                offset = {"hip": 5, "thigh": 7, "calf": 9}[group]
                np.testing.assert_allclose(
                    [dofs["lower"][index], dofs["upper"][index]],
                    descriptor[offset:offset + 2],
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"{stage}: env {env_id}, joint {name}",
                )

            snapshots[env_id] = (
                np.array([body.mass for body in bodies]),
                dofs["lower"].copy(),
                dofs["upper"].copy(),
            )

            print(
                f"[MORPH CHECK] {stage}, env={env_id}, "
                f"base_mass={descriptor[4]:.4f}, "
                f"hip_limits={descriptor[5:7]}"
            )

        return snapshots

    before = read_and_check("before reset")

    for reset_number in range(3):
        env.reset_idx(env_ids)

        if not torch.equal(env.morphology_vector, vector_before):
            raise AssertionError("Morphology vectors changed during reset.")

        after = read_and_check(f"after reset {reset_number + 1}")
        for env_id in ids:
            for old, new in zip(before[env_id], after[env_id]):
                np.testing.assert_allclose(
                    new, old, rtol=0, atol=1e-6,
                    err_msg=f"Physical morphology changed in env {env_id}",
                )

    print("[MORPH CHECK] PASS: masses, joint limits, and reset persistence.")





def train(args):
    set_seed(args.seed)

    # Apply robot defaults, then the original training overrides.
    configure_robot(Cfg, args.robot)
    configure_training(Cfg)

    Cfg.env.num_envs = args.num_envs
    AC_Args.variant = args.variant

    RunnerArgs.resume = False
    RunnerArgs.max_iterations = args.iterations
    RunnerArgs.save_interval = args.save_interval
    RunnerArgs.save_video_interval = args.video_interval

    configure_logger(args)

    actor_conditioning, critic_conditioning = VARIANTS[args.variant]

    print("=" * 70)
    print(f"[ROBOT] {args.robot}")
    print(f"[VARIANT] {args.variant}")
    print(f"[ACTOR CONDITIONING] {actor_conditioning}")
    print(f"[CRITIC CONDITIONING] {critic_conditioning}")
    print(f"[SEED] {args.seed}")
    print(f"[ENVIRONMENTS] {args.num_envs}")
    print(f"[ITERATIONS] {args.iterations}")
    print(f"[DEVICE] {args.device}")
    print("[MORPHOLOGY] Sample/apply once per environment")
    print("=" * 70)

    env = VelocityTrackingEasyEnv(
        sim_device=args.device,
        headless=args.headless,
        cfg=Cfg,
        morphology_mode="train",
    )

    # Optional diagnostic: verify morphology readback and reset persistence. Uncomment to see if its working or not. 

    # check_morphology(env)

    env = HistoryWrapper(env)

    # Preserve parameters.pkl for existing play/evaluation loaders.
    experiment = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    experiment["morphology_mode"] = "train"
    experiment["morphology_sampling"] = "once_per_environment"

    logger.log_params(
        AC_Args=vars(AC_Args),
        PPO_Args=vars(PPO_Args),
        RunnerArgs=vars(RunnerArgs),
        Cfg=vars(Cfg),
        experiment=experiment,
    )

    runner = Runner(env, device=args.device)
    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=True,
        eval_freq=100,
    )


if __name__ == "__main__":
    train(parse_args())