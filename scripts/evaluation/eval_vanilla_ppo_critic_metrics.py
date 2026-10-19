#!/usr/bin/env python3
"""Evaluate tracking, critic calibration, and advantage diagnostics.

Returns bootstrap from the critic at the rollout horizon.
Advantage-sign agreement uses centered returns as a reference proxy.
"""

import argparse
from pathlib import Path

import isaacgym  # noqa: F401 — import before torch

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

    location = parser.add_mutually_exclusive_group()
    location.add_argument("--run-dir", type=Path, default=None)
    location.add_argument(
        "--label",
        help="Run path relative to the repository's runs directory.",
    )

    parser.add_argument("--robot", choices=ROBOT_CONFIGS, default="go2")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--num-envs", "--num_envs", dest="num_envs",
        type=int, default=64,
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--gae-lambda", "--gae_lambda", dest="gae_lambda",
        type=float, default=0.95,
    )

    parser.add_argument("--vx", "--cmd_vx", dest="vx", type=float, default=2.0)
    parser.add_argument("--vy", "--cmd_vy", dest="vy", type=float, default=0.0)
    parser.add_argument(
        "--yaw", "--cmd_yaw", dest="yaw", type=float, default=0.0
    )
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
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be between zero and one.")
    if not 0.0 <= args.gae_lambda <= 1.0:
        parser.error("--gae-lambda must be between zero and one.")

    if args.label is not None:
        args.run_dir = Path(MINI_GYM_ROOT_DIR) / "runs" / args.label
    elif args.run_dir is None:
        args.run_dir = DEFAULT_RUN

    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def extract_timeout_mask(info, done):
    for key in ("time_outs", "timeouts"):
        if key in info and info[key] is not None:
            timeout = torch.as_tensor(
                info[key], device=done.device
            ).reshape(-1).bool()
            if timeout.shape != done.shape:
                raise ValueError("Timeout mask does not match done shape.")
            return timeout
    return torch.zeros_like(done)


@torch.no_grad()
def value_fn(model, env, obs):
    privileged = obs.get("privileged_obs")
    if privileged is None:
        privileged = env.get_privileged_observations()
    if privileged is None:
        raise RuntimeError(
            "Critic evaluation requires the training privileged observations."
        )

    morphology = obs.get("morphology_vector")
    if morphology is None:
        morphology = env.morphology_vector

    return model.evaluate(
        obs["obs_history"].to(env.device),
        privileged.to(env.device),
        morphology.to(env.device),
    ).reshape(env.num_envs)


@torch.no_grad()
def collect_rollout(env, policy, model, args):
    shape = (args.steps, env.num_envs)
    values = torch.zeros(shape, device=env.device)
    rewards = torch.zeros_like(values)
    failures = torch.zeros(shape, dtype=torch.bool, device=env.device)
    masks = torch.zeros_like(failures)
    velocities = torch.zeros_like(values)

    alive = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    obs = env.reset()

    for step in trange(args.steps, desc="PPO metrics"):
        set_commands(env, args)
        obs = refresh_command_observations(env, obs)

        values[step] = value_fn(model, env, obs)
        masks[step] = alive

        actions = policy(obs)
        obs, reward, done, info = env.step(actions)

        done = done.to(env.device).reshape(-1).bool()
        timeout = extract_timeout_mask(info, done)

        if (done & timeout).any().item():
            raise RuntimeError(
                "A timeout occurred inside the evaluation horizon. "
                "Returns must not continue through an automatic reset."
            )

        rewards[step] = reward.reshape(-1)
        failures[step] = done & ~timeout
        velocities[step] = env.base_lin_vel[:, 0]
        alive &= ~failures[step]

    set_commands(env, args)
    obs = refresh_command_observations(env, obs)
    last_value = value_fn(model, env, obs)

    return values, rewards, failures, masks, velocities, last_value


def compute_metrics(
    values, rewards, failures, mask, velocities, last_value, dt, args
):
    steps, num_envs = rewards.shape
    nan = torch.tensor(float("nan"), device=rewards.device)

    # Discounted returns, bootstrapped only at the rollout horizon.
    returns = torch.zeros_like(rewards)
    running_return = last_value.clone()

    for step in reversed(range(steps)):
        nonterminal = (~failures[step]).float()
        running_return = (
            rewards[step] + args.gamma * running_return * nonterminal
        )
        returns[step] = running_return

    # GAE advantages.
    extended_values = torch.cat(
        (values, last_value.unsqueeze(0)), dim=0
    )
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros(num_envs, device=rewards.device)

    for step in reversed(range(steps)):
        nonterminal = (~failures[step]).float()
        delta = (
            rewards[step]
            + args.gamma * extended_values[step + 1] * nonterminal
            - extended_values[step]
        )
        running_advantage = (
            delta
            + args.gamma * args.gae_lambda
            * nonterminal * running_advantage
        )
        advantages[step] = running_advantage

    # Include the transition that causes the first failure;
    # exclude subsequent transitions from that environment.
    predicted = values[mask]
    targets = returns[mask]
    errors = predicted - targets

    if targets.numel():
        return_std = targets.std(unbiased=False)
        bias = errors.mean()
        error_variance = errors.var(unbiased=False)
        mse = errors.square().mean()
    else:
        return_std = bias = error_variance = mse = nan

    return_std_eps = 1e-2
    valid_calibration = (
        targets.numel() > num_envs * 20
        and torch.isfinite(return_std).item()
        and return_std.item() > return_std_eps
    )

    if valid_calibration:
        explained_variance = (
            1.0
            - errors.var(unbiased=False)
            / (targets.var(unbiased=False) + return_std_eps ** 2)
        )
        normalized_bias = bias / return_std
        nve = torch.sqrt(mse) / return_std
    else:
        explained_variance = normalized_bias = nve = nan

    valid_advantages = advantages[mask]
    advantage_variance = (
        valid_advantages.var(unbiased=False)
        if valid_advantages.numel() else nan
    )

    # Centered-return reference proxy; not ground-truth advantages.
    sign_agreement = sign_flip_rate = nan
    if valid_calibration:
        reference_advantages = targets - targets.mean()
        sign_mask = reference_advantages.abs() > 1e-6

        if sign_mask.any().item():
            sign_agreement = (
                valid_advantages[sign_mask].sign()
                == reference_advantages[sign_mask].sign()
            ).float().mean()
            sign_flip_rate = 1.0 - sign_agreement

    first_failure = torch.full(
        (num_envs,), float(steps), device=rewards.device
    )
    has_failed = torch.zeros(
        num_envs, dtype=torch.bool, device=rewards.device
    )

    for step in range(steps):
        newly_failed = ~has_failed & failures[step]
        first_failure[newly_failed] = step + 1
        has_failed |= failures[step]

    alive_time = first_failure * dt
    no_failure_rate = (~has_failed).float().mean()

    valid_velocity = velocities[mask]
    tracking_error = valid_velocity - args.vx

    if valid_velocity.numel():
        mean_vx = valid_velocity.mean()
        std_vx = valid_velocity.std(unbiased=False)
        rmse_vx = tracking_error.square().mean().sqrt()
        mean_absolute_error = tracking_error.abs().mean()
    else:
        mean_vx = std_vx = rmse_vx = mean_absolute_error = nan

    return {
        "samples_pre_first_failure": int(mask.sum().item()),
        "valid_calibration": bool(valid_calibration),
        "mean_vx": mean_vx.item(),
        "std_vx": std_vx.item(),
        "rmse_vx": rmse_vx.item(),
        "mean_abs_tracking_error": mean_absolute_error.item(),
        "explained_variance": explained_variance.item(),
        "advantage_variance": advantage_variance.item(),
        "value_bias": bias.item(),
        "normalized_value_bias": normalized_bias.item(),
        "value_error_variance": error_variance.item(),
        "value_mse": mse.item(),
        "nve": nve.item(),
        "advantage_sign_agreement": sign_agreement.item(),
        "sign_flip_rate": sign_flip_rate.item(),
        "mean_alive_time": alive_time.mean().item(),
        "std_alive_time": alive_time.std(unbiased=False).item(),
        "no_failure_rate": no_failure_rate.item(),
        "success_rate_percent": 100.0 * no_failure_rate.item(),
    }


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    parameters = load_parameters(args.run_dir)
    cfg = configure_environment(parameters, args)

    # Prevent time-limit resets inside this diagnostic rollout.
    policy_dt = float(cfg.sim.dt) * int(cfg.control.decimation)
    cfg.env.episode_length_s = max(
        float(cfg.env.episode_length_s),
        (args.steps + 10) * policy_dt,
    )

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

        rollout = collect_rollout(env, policy, model, args)
        metrics = compute_metrics(*rollout, float(env.dt), args)

        print("\n[WTW_CRITIC_METRICS]")
        print(f"run_dir: {args.run_dir}")
        print(f"robot: {args.robot}")
        print(f"cmd_vx: {args.vx:.6f}")
        print(f"cmd_vy: {args.vy:.6f}")
        print(f"cmd_yaw: {args.yaw:.6f}")
        print(f"gamma: {args.gamma:.6f}")
        print(f"gae_lambda: {args.gae_lambda:.6f}")
        print(f"num_envs: {env.num_envs}")
        print(f"steps: {args.steps}")
        print(f"horizon_time_sec: {args.steps * env.dt:.6f}")

        for name, value in metrics.items():
            if isinstance(value, (bool, int)):
                print(f"{name}: {value}")
            else:
                print(f"{name}: {value:.6f}")

        print("return_reference: discounted_horizon_bootstrapped")
        print("sign_reference: centered_returns_proxy")
        print("[/WTW_CRITIC_METRICS]\n")

    finally:
        env.close()


if __name__ == "__main__":
    main()