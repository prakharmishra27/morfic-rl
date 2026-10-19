#!/usr/bin/env python3
"""Export a student-policy rollout and GAE-based value diagnostics.

Run from the repository root:
    python -m scripts.evaluation.export_rollout --robot go2 --headless

Every done cuts GAE and receives zero bootstrap, including timeouts.
These are GAE-target diagnostics, not independent Monte Carlo calibration.
"""

import argparse
from datetime import datetime
from pathlib import Path

import isaacgym  # noqa: F401 — must precede torch

import numpy as np
import torch
from tqdm import trange

from scripts.play import (
    DEFAULT_RUN,
    GAITS,
    configure_environment,
    load_parameters,
    load_policy,
    refresh_command_observations,
    set_commands,
)
from go2_gym import MINI_GYM_ROOT_DIR
from go2_gym.envs.robots import ROBOT_CONFIGS
from go2_gym.envs.robots.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go2_gym_learn.ppo_cse.actor_critic import VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(MINI_GYM_ROOT_DIR) / "outputs" / "evaluation",
    )
    parser.add_argument("--robot", choices=ROBOT_CONFIGS, default="go2")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

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
    args.num_envs = 1

    if args.steps < 1:
        parser.error("--steps must be positive.")
    if args.frequency <= 0:
        parser.error("--frequency must be positive.")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be between zero and one.")
    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gae-lambda must be between zero and one.")

    return args


def compute_metrics(rewards, dones, values, gamma, gae_lambda):
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    values = np.asarray(values, dtype=np.float64)

    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional.")
    if dones.shape != rewards.shape:
        raise ValueError("dones must have the same shape as rewards.")
    if values.shape != (len(rewards) + 1,):
        raise ValueError("values must contain T+1 entries.")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError("Rollout contains nonfinite rewards or values.")

    advantages = np.zeros_like(rewards)
    last_advantage = 0.0

    for step in range(len(rewards) - 1, -1, -1):
        nonterminal = float(not dones[step])
        delta = (
            rewards[step]
            + gamma * values[step + 1] * nonterminal
            - values[step]
        )
        last_advantage = (
            delta
            + gamma * gae_lambda * nonterminal * last_advantage
        )
        advantages[step] = last_advantage

    targets = advantages + values[:-1]
    errors = targets - values[:-1]
    target_variance = np.var(targets)

    # Explained variance is undefined for constant targets.
    explained_variance = (
        float(1.0 - np.var(errors) / target_variance)
        if target_variance > 1e-8
        else float("nan")
    )

    metrics = {
        "gae_target_explained_variance": explained_variance,
        "gae_target_value_mse": float(np.mean(errors ** 2)),
        "advantage_variance": float(np.var(advantages)),
        "mean_absolute_advantage": float(np.mean(np.abs(advantages))),
    }
    return metrics, advantages, targets


@torch.no_grad()
def evaluate_value(model, env, obs):
    privileged_obs = obs.get("privileged_obs")
    if privileged_obs is None:
        privileged_obs = env.get_privileged_observations()
    if privileged_obs is None:
        raise RuntimeError(
            "Critic evaluation requires the privileged observations "
            "used during training."
        )

    morphology = obs.get("morphology_vector")
    if morphology is None:
        morphology = env.morphology_vector

    value = model.evaluate(
        obs["obs_history"].to(env.device),
        privileged_obs.to(env.device),
        morphology.to(env.device),
    )
    return value.reshape(-1)[0].item()


@torch.no_grad()
def collect_rollout(env, policy, model, args):
    rewards = np.zeros(args.steps, dtype=np.float32)
    dones = np.zeros(args.steps, dtype=bool)
    timeouts = np.zeros(args.steps, dtype=bool)
    values = np.zeros(args.steps + 1, dtype=np.float32)
    measured = np.zeros((args.steps, 3), dtype=np.float32)
    commands = np.tile(
        np.array([args.vx, args.vy, args.yaw], dtype=np.float32),
        (args.steps, 1),
    )

    obs = env.reset()

    for step in trange(args.steps, desc=f"Export {args.robot}"):
        # Ensure both actor and critic see the commanded observation.
        set_commands(env, args)
        obs = refresh_command_observations(env, obs)
        values[step] = evaluate_value(model, env, obs)

        actions = policy(obs)
        obs, reward, done, info = env.step(actions)

        rewards[step] = reward.reshape(-1)[0].item()
        dones[step] = bool(done.reshape(-1)[0].item())

        timeout = info.get("time_outs")
        if timeout is not None:
            timeouts[step] = bool(
                torch.as_tensor(timeout).reshape(-1)[0].item()
            )

        measured[step] = (
            env.base_lin_vel[0, 0].item(),
            env.base_lin_vel[0, 1].item(),
            env.base_ang_vel[0, 2].item(),
        )

    set_commands(env, args)
    obs = refresh_command_observations(env, obs)
    values[-1] = evaluate_value(model, env, obs)

    return rewards, dones, timeouts, values, measured, commands


def main():
    args = parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.save_dir = args.save_dir.expanduser().resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    parameters = load_parameters(args.run_dir)
    cfg = configure_environment(parameters, args)

    env = HistoryWrapper(
        VelocityTrackingEasyEnv(
            sim_device=args.device,
            headless=args.headless,
            cfg=cfg,
            morphology_mode="eval",
        )
    )

    try:
        policy, model = load_policy(
            env,
            parameters,
            args.run_dir,
            args.variant,
            return_model=True,
        )

        rewards, dones, timeouts, values, measured, commands = (
            collect_rollout(env, policy, model, args)
        )

        metrics, advantages, targets = compute_metrics(
            rewards, dones, values, args.gamma, args.gae_lambda
        )
        tracking_rmse = np.sqrt(
            np.mean((measured - commands) ** 2, axis=0)
        )

        args.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        saved_ac = parameters.get("AC_Args", {})
        variant = (
            args.variant
            or saved_ac.get("variant")
            or "morfic_film"
        )
        output = args.save_dir / (
            f"rollout_{variant}_{args.robot}_{timestamp}.npz"
        )

        np.savez_compressed(
            output,
            rewards=rewards,
            dones=dones,
            timeouts=timeouts,
            values=values,
            advantages=advantages,
            gae_targets=targets,
            commands=commands,
            measured_velocities=measured,
            # Retain the original forward-tracking field names.
            v_cmd_x=commands[:, 0],
            v_meas_x=measured[:, 0],
            morphology_vector=(
                env.morphology_vector[0].detach().cpu().numpy()
            ),
            morphology_min=env.morph_min.detach().cpu().numpy(),
            morphology_max=env.morph_max.detach().cpu().numpy(),
            variant=variant,
            robot=args.robot,
            run_dir=str(args.run_dir),
            seed=args.seed,
            steps=args.steps,
            dt=float(env.dt),
            gamma=args.gamma,
            lam=args.gae_lambda,
            timeout_handling="zero_bootstrap_at_every_done",
            **metrics,
        )

        print(f"\nSaved: {output}")
        for index, name in enumerate(("vx", "vy", "yaw")):
            print(f"Tracking RMSE {name}: {tracking_rmse[index]:.4f}")
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")
        print(
            f"Done transitions: {int(dones.sum())}; "
            f"timeouts: {int(timeouts.sum())}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()