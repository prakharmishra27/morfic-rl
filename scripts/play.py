#!/usr/bin/env python3
"""Play a trained policy on a selected robot."""

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401

import numpy as np
import torch
from tqdm import trange

from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.base.legged_robot_config import Cfg
from go2_gym.envs.robots import ROBOT_CONFIGS, configure_robot
from go2_gym.envs.robots.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go2_gym_learn.ppo_cse.actor_critic import (
    AC_Args,
    ActorCritic,
    VARIANTS,
)


DEFAULT_RUN = (
    Path(MINI_GYM_ROOT_DIR)
    / "runs"
    / "gait-conditioned-agility"
    / "2026-05-15"
    / "train"
    / "003739.836224"
)

GAITS = {
    "pronking": (0.0, 0.0, 0.0),
    "trotting": (0.5, 0.0, 0.0),
    "bounding": (0.0, 0.5, 0.0),
    "pacing": (0.0, 0.0, 0.5),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--robot", choices=ROBOT_CONFIGS, default="go2")
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default=None,
        help="Override the saved variant; old checkpoints default to morfic_film.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--plot", action="store_true")

    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--gait", choices=GAITS, default="trotting")
    parser.add_argument("--frequency", type=float, default=3.0)
    parser.add_argument("--body-height", type=float, default=0.0)
    parser.add_argument("--footswing-height", type=float, default=0.08)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--stance-length", type=float, default=0.40)

    args = parser.parse_args()

    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive.")
    if args.frequency <= 0:
        parser.error("--frequency must be positive.")

    return args


def load_parameters(run_dir):
    path = run_dir / "parameters.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")

    with path.open("rb") as file:
        parameters = pickle.load(file)

    if not isinstance(parameters, Mapping):
        raise TypeError("parameters.pkl must contain a configuration dictionary.")

    return parameters


def restore_training_config(parameters):
    """Restore saved environment settings before selecting the target robot."""
    saved_cfg = parameters.get("Cfg")
    if not isinstance(saved_cfg, Mapping):
        raise ValueError("parameters.pkl does not contain a valid Cfg dictionary.")

    for section_name, values in saved_cfg.items():
        if not hasattr(Cfg, section_name):
            continue
        if not isinstance(values, Mapping):
            continue

        section = getattr(Cfg, section_name)
        for name, value in values.items():
            setattr(section, name, value)


def configure_environment(parameters, args):
    restore_training_config(parameters)

    # Robot configuration also applies shared defaults. Preserve the saved
    # training settings except the target asset, pose, control, and morphology.
    saved_sections = {}
    for name, values in parameters["Cfg"].items():
        if isinstance(values, Mapping):
            saved_sections[name] = dict(values)

    configure_robot(Cfg, args.robot)

    robot_sections = {"asset", "init_state", "control", "Morphology"}
    for section_name, values in saved_sections.items():
        if section_name in robot_sections:
            continue
        if not hasattr(Cfg, section_name):
            continue

        section = getattr(Cfg, section_name)
        for name, value in values.items():
            setattr(section, name, value)

    # Normalization always uses source training bounds, even for target robots.
    source_morphology = saved_sections.get("Morphology", {})
    for name in ("morphology_min", "morphology_max"):
        value = source_morphology.get(name)
        if value is not None:
            setattr(
                Cfg.Morphology,
                name,
                torch.as_tensor(value, dtype=torch.float32).cpu(),
            )
        else:
            print(
                f"[play] Saved {name} unavailable; using the source bounds "
                "defined in robots/morphology.py."
            )

    # Keep the training action timing and scaling. Target-specific gains and
    # control type come from the selected robot configuration.
    source_control = saved_sections.get("control", {})
    for name in ("decimation", "action_scale", "hip_scale_reduction"):
        if name in source_control:
            setattr(Cfg.control, name, source_control[name])

    Cfg.env.num_envs = args.num_envs
    Cfg.env.num_eval_envs = 0
    Cfg.env.num_recording_envs = 0
    Cfg.env.record_video = False

    # Nominal playback: keep physical morphology consistent with its descriptor.
    for name, value in vars(Cfg.domain_rand).items():
        if name.startswith("randomize_") and isinstance(value, bool):
            setattr(Cfg.domain_rand, name, False)

    Cfg.domain_rand.push_robots = False
    Cfg.noise.add_noise = False
    Cfg.commands.command_curriculum = False
    Cfg.commands.heading_command = False   

    return Cfg


def load_policy(
    env, parameters, run_dir, variant_override, return_model=False
):
    saved_ac = parameters.get("AC_Args", {})
    if not isinstance(saved_ac, Mapping):
        saved_ac = {}

    # Restore network settings before constructing the model.
    for name, value in saved_ac.items():
        if hasattr(AC_Args, name):
            setattr(AC_Args, name, value)

    variant = variant_override or saved_ac.get("variant") or "morfic_film"
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported policy variant: {variant!r}")
    AC_Args.variant = variant

    state_path = run_dir / "checkpoints" / "ac_weights_last.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Policy weights not found: {state_path}")

    state = torch.load(str(state_path), map_location="cpu")

    # The morphology encoder's final weight records its output dimension.
    latent_dim = 64
    encoder_weight = state.get("morph_encoder.net.2.weight")
    if encoder_weight is not None:
        latent_dim = encoder_weight.shape[0]

    model = ActorCritic(
        num_obs=env.num_obs,
        num_privileged_obs=env.num_privileged_obs,
        num_obs_history=env.num_obs_history,
        num_actions=env.num_actions,
        morphology_dim=env.morphology_vector.shape[-1],
        morphology_latent_dim=latent_dim,
        variant=variant,
    )

    # Fail visibly if an old checkpoint needs a compatibility adjustment.
    model.load_state_dict(state, strict=True)
    model.to(env.device)
    model.eval()

    print(f"[play] variant: {variant}")
    print(f"[play] weights: {state_path}")

    @torch.no_grad()
    def policy(obs):
        morphology = obs.get("morphology_vector")
        if morphology is None:
            morphology = env.morphology_vector

        return model.act_student(
            obs["obs"].to(env.device),
            obs["obs_history"].to(env.device),
            morphology.to(env.device),
        )

    if return_model:
        return policy, model
    return policy


def set_commands(env, args):
    values = (
        args.vx,
        args.vy,
        args.yaw,
        args.body_height,
        args.frequency,
        *GAITS[args.gait],
        0.5,  # Stance duration.
        args.footswing_height,
        0.0,  # Body pitch.
        0.0,  # Body roll.
        args.stance_width,
        args.stance_length,
        0.0,  # Auxiliary command.
    )

    count = min(env.commands.shape[1], len(values))
    env.commands[:, :count] = torch.tensor(
        values[:count], device=env.device, dtype=env.commands.dtype
    )


def refresh_command_observations(env, obs):
    """Update the latest history frame after changing commands."""
    env.unwrapped.compute_observations()
    obs["obs"] = env.obs_buf
    obs["obs_history"][:, -env.num_obs:] = env.obs_buf
    obs["morphology_vector"] = env.morphology_vector
    return obs


@torch.no_grad()
def play_loop(env, policy, args):
    obs = env.reset()

    error_sum = torch.zeros((env.num_envs, 3), device=env.device)
    velocity_sum = torch.zeros_like(error_sum)
    target = torch.tensor(
        [args.vx, args.vy, args.yaw], device=env.device
    )

    has_done = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    first_done = torch.full(
        (env.num_envs,), args.steps, dtype=torch.long, device=env.device
    )

    trajectory = np.zeros((args.steps, 3))
    joint_positions = np.zeros((args.steps, env.num_actions))

    for step in trange(args.steps, desc="Play"):
        set_commands(env, args)
        obs = refresh_command_observations(env, obs)
        actions = policy(obs)

        obs, _, done, _ = env.step(actions)

        velocity = torch.stack(
            (
                env.base_lin_vel[:, 0],
                env.base_lin_vel[:, 1],
                env.base_ang_vel[:, 2],
            ),
            dim=-1,
        )
        velocity_sum += velocity
        error_sum += (velocity - target).square()

        done = done.to(device=env.device).reshape(-1).bool()
        newly_done = (~has_done) & done
        first_done[newly_done] = step + 1
        has_done |= done

        trajectory[step] = velocity[0].cpu().numpy()
        joint_positions[step] = env.dof_pos[0].cpu().numpy()

    rmse = torch.sqrt(error_sum / args.steps)
    mean_velocity = velocity_sum / args.steps

    print("\nPlayback summary")
    print(f"Robot: {args.robot}")
    print(f"Environments: {env.num_envs}")
    print(f"Horizon: {args.steps * env.dt:.2f} s")

    for index, name in enumerate(("vx", "vy", "yaw")):
        unit = "rad/s" if name == "yaw" else "m/s"
        print(
            f"{name}: target={target[index].item():.3f}, "
            f"mean={mean_velocity[:, index].mean().item():.3f}, "
            f"RMSE={rmse[:, index].mean().item():.3f} {unit}"
        )

    print(
        "Mean time to first done or horizon: "
        f"{(first_done.float() * env.dt).mean().item():.2f} s"
    )
    print(
        "No-done fraction: "
        f"{100.0 * (~has_done).float().mean().item():.1f}%"
    )
    print("Tracking summaries include samples after automatic resets.")

    if args.plot:
        plot_trajectory(trajectory, joint_positions, target.cpu().numpy(), env.dt)


def plot_trajectory(trajectory, joints, target, dt):
    import matplotlib.pyplot as plt

    time = (np.arange(len(trajectory)) + 1) * dt
    figure, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    for index, name in enumerate(("vx", "vy", "yaw")):
        axes[index].plot(time, trajectory[:, index], label="measured")
        axes[index].axhline(target[index], linestyle="--", label="command")
        axes[index].set_ylabel(name)
        axes[index].legend()

    axes[3].plot(time, joints)
    axes[3].set_ylabel("Joint position (rad)")
    axes[3].set_xlabel("Time (s)")
    figure.tight_layout()
    plt.show()


def main():
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    parameters = load_parameters(args.run_dir)
    cfg = configure_environment(parameters, args)

    env = VelocityTrackingEasyEnv(
        sim_device=args.device,
        headless=args.headless,
        cfg=cfg,
        morphology_mode="eval",
    )
    env = HistoryWrapper(env)

    try:
        policy = load_policy(
            env, parameters, args.run_dir, args.variant
        )
        print(f"[play] robot: {args.robot}")
        print(f"[play] morphology: {env.morphology_vector[0]}")
        play_loop(env, policy, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()