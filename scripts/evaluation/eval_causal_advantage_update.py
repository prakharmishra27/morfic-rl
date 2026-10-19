#!/usr/bin/env python3
"""Compare critic-derived advantages on one fixed MorFiC teacher rollout.

All gradients use the same source actor and sampled actions.
Returns are finite-horizon discounted rewards with zero horizon bootstrap.
The centered-return reference is a proxy, not ground-truth advantages.
No optimizer updates are performed.
"""

import argparse
from pathlib import Path

import isaacgym  # noqa: F401 — before torch

import numpy as np
import torch
from tqdm import trange

from scripts.play import (
    DEFAULT_RUN,
    configure_environment,
    load_parameters,
    load_policy,
    refresh_command_observations,
    set_commands,
)
from scripts.evaluation.eval_vanilla_ppo_critic_metrics import extract_timeout_mask
from go2_gym.envs.robots import ROBOT_CONFIGS
from go2_gym.envs.robots.velocity_tracking import VelocityTrackingEasyEnv
from go2_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go2_gym_learn.ppo_cse.actor_critic import VARIANTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--robot", choices=ROBOT_CONFIGS, default="go2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--sign-eps", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1024)

    parser.add_argument(
        "--critic",
        action="append",
        default=[],
        metavar="VARIANT=RUN_DIR",
        help="Optional comparison checkpoint; repeat for multiple critics.",
    )
    parser.add_argument("--vx", type=float, default=2.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)

    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.batch_size) < 1:
        parser.error("Environment count, steps, and batch size must be positive.")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be between zero and one.")
    if args.clip_param < 0 or args.sign_eps < 0:
        parser.error("Clip parameter and sign threshold cannot be negative.")

    # Shared playback command defaults.
    args.gait = "trotting"
    args.frequency = 3.0
    args.body_height = 0.0
    args.footswing_height = 0.08
    args.stance_width = 0.25
    args.stance_length = 0.40
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def inputs(env, obs):
    priv = obs.get("privileged_obs")
    if priv is None:
        priv = env.get_privileged_observations()
    if priv is None:
        raise RuntimeError("Teacher evaluation requires privileged observations.")

    return (
        obs["obs_history"].to(env.device),
        priv.to(env.device),
        env.morphology_vector.to(env.device),
    )


@torch.no_grad()
def collect_rollout(env, model, args):
    buffers = {
        key: [] for key in (
            "hist", "priv", "morph", "actions",
            "logprob", "rewards", "failures", "mask",
        )
    }
    alive = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    obs = env.reset()

    for _ in trange(args.steps, desc="Fixed MorFiC teacher rollout"):
        set_commands(env, args)
        obs = refresh_command_observations(env, obs)
        hist, priv, morph = inputs(env, obs)

        actions = model.act(hist, priv, morph)
        logprob = model.get_actions_log_prob(actions)

        for key, value in (
            ("hist", hist), ("priv", priv), ("morph", morph),
            ("actions", actions), ("logprob", logprob), ("mask", alive),
        ):
            buffers[key].append(value.detach().clone())

        obs, reward, done, info = env.step(actions)
        done = done.to(env.device).reshape(-1).bool()
        timeout = extract_timeout_mask(info, done)

        if (done & timeout).any().item():
            raise RuntimeError(
                "Unexpected timeout inside rollout; refusing to join "
                "returns across an automatic reset."
            )

        failure = done & ~timeout
        buffers["rewards"].append(reward.detach().reshape(-1).clone())
        buffers["failures"].append(failure.clone())
        alive &= ~failure

    return {key: torch.stack(value) for key, value in buffers.items()}


def discounted_returns(rewards, failures, gamma):
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        running = rewards[step] + gamma * running * (~failures[step])
        result[step] = running
    return result


@torch.no_grad()
def critic_values(model, rollout, batch_size):
    hist = rollout["hist"].flatten(0, 1)
    priv = rollout["priv"].flatten(0, 1)
    morph = rollout["morph"].flatten(0, 1)

    values = []
    for start in range(0, len(hist), batch_size):
        end = start + batch_size
        values.append(
            model.evaluate(
                hist[start:end], priv[start:end], morph[start:end]
            ).reshape(-1)
        )
    return torch.cat(values).reshape_as(rollout["rewards"])


def actor_gradient(model, rollout, advantage, args):
    """Full valid-sample gradient, accumulated in memory-bounded batches."""
    mask = rollout["mask"].flatten()
    hist = rollout["hist"].flatten(0, 1)[mask]
    priv = rollout["priv"].flatten(0, 1)[mask]
    morph = rollout["morph"].flatten(0, 1)[mask]
    actions = rollout["actions"].flatten(0, 1)[mask]
    old_logprob = rollout["logprob"].flatten()[mask]
    advantage = advantage.flatten()[mask].detach()

    if len(advantage) < 2:
        raise RuntimeError("Not enough valid rollout samples.")

    advantage = (
        advantage - advantage.mean()
    ) / (advantage.std(unbiased=False) + 1e-8)

    # Match the effective gradient function in the original file:
    # actor body + std + morphology encoder.
    parameters = list(model.actor_body.parameters()) + [model.std]
    if model.morph_encoder is not None:
        parameters += list(model.morph_encoder.parameters())

    previous_flags = [p.requires_grad for p in model.parameters()]
    previous_mode = model.training

    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in parameters:
            parameter.requires_grad_(True)
        model.eval()
        model.zero_grad(set_to_none=True)

        count = len(advantage)
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            model.update_distribution(
                hist[start:end], priv[start:end], morph[start:end]
            )
            logprob = model.get_actions_log_prob(actions[start:end])
            ratio = torch.exp(logprob - old_logprob[start:end])

            unclipped = -advantage[start:end] * ratio
            clipped = -advantage[start:end] * ratio.clamp(
                1 - args.clip_param, 1 + args.clip_param
            )
            # Sum / full count preserves the full-batch mean gradient.
            loss = torch.maximum(unclipped, clipped).sum() / count
            loss.backward()

        return torch.cat([
            (
                parameter.grad.detach().reshape(-1)
                if parameter.grad is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter in parameters
        ]).clone()
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, flag in zip(model.parameters(), previous_flags):
            parameter.requires_grad_(flag)
        model.train(previous_mode)


def correlation(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    if denominator.item() <= 1e-12:
        return float("nan")
    return (torch.dot(a, b) / denominator).item()


def cosine(a, b):
    a, b = a.double(), b.double()
    denominator = a.norm() * b.norm()
    if denominator.item() <= 1e-12:
        return float("nan")
    return (torch.dot(a, b) / denominator).item()


def check_comparison_config(source, comparison):
    # Equal dimensions alone do not establish compatible state semantics.
    for section in ("env", "normalization"):
        if repr(source["Cfg"].get(section)) != repr(
            comparison["Cfg"].get(section)
        ):
            raise ValueError(
                f"Comparison checkpoint has different {section} settings. "
                "Verify and align observation semantics before comparing."
            )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source_parameters = load_parameters(args.run_dir)
    cfg = configure_environment(source_parameters, args)

    # Prevent timeout resets inside the diagnostic horizon.
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
        _, source_model = load_policy(
            env, source_parameters, args.run_dir,
            "morfic_film", return_model=True,
        )
        models = {"MorFiC": source_model}

        for specification in args.critic:
            variant, separator, directory = specification.partition("=")
            if not separator or variant not in VARIANTS:
                raise ValueError(
                    "--critic must be VARIANT=RUN_DIR with a valid variant."
                )
            if variant in models:
                raise ValueError(f"Duplicate critic name: {variant}")

            directory = Path(directory).expanduser().resolve()
            parameters = load_parameters(directory)
            check_comparison_config(source_parameters, parameters)

            # Condition each critic using the same source normalization.
            saved_morphology = parameters["Cfg"].get("Morphology", {})
            for name, expected in (
                ("morphology_min", env.morph_min),
                ("morphology_max", env.morph_max),
            ):
                if name not in saved_morphology:
                    raise ValueError(
                        f"{directory}: missing {name}; verify the original "
                        "training bounds before using this comparison."
                    )
                actual = torch.as_tensor(
                    saved_morphology[name],
                    device=env.device, dtype=torch.float32,
                )
                if (
                    actual.shape != expected.shape
                    or not torch.allclose(actual, expected)
                ):
                    raise ValueError(
                        f"{directory}: different source morphology bounds."
                    )

            _, model = load_policy(
                env, parameters, directory, variant, return_model=True
            )
            models[variant] = model

        rollout = collect_rollout(env, source_model, args)
        returns = discounted_returns(
            rollout["rewards"], rollout["failures"], args.gamma
        )
        mask = rollout["mask"].bool()
        reference = returns - returns[mask].mean()

        sources = {"MC_reference": reference}
        for name, model in models.items():
            values = critic_values(model, rollout, args.batch_size)
            sources[name] = returns - values
            print(
                f"{name}: "
                f"V_mean={values[mask].mean().item():.4f}, "
                f"V_std={values[mask].std(unbiased=False).item():.4f}"
            )

        reference_gradient = actor_gradient(
            source_model, rollout, reference, args
        )

        print("\n[CRITIC_EFFECT_METRICS]")
        print(f"robot: {args.robot}")
        print(f"valid_samples: {mask.sum().item()}")
        print("return_reference: finite_horizon_zero_bootstrap")
        print("actor_path: stochastic_teacher")
        print("gradient_parameters: actor_body,std,morph_encoder")

        sign_mask = mask & (reference.abs() > args.sign_eps)

        for name, advantage in sources.items():
            gradient = (
                reference_gradient if name == "MC_reference"
                else actor_gradient(source_model, rollout, advantage, args)
            )
            flip = (
                (
                    advantage[sign_mask].sign()
                    != reference[sign_mask].sign()
                ).float().mean().item()
                if sign_mask.any().item() else float("nan")
            )
            print(
                f"{name}: "
                f"sign_flip_vs_mc={flip:.6f}, "
                f"adv_corr_vs_mc="
                f"{correlation(advantage[mask], reference[mask]):.6f}, "
                f"grad_cos_vs_mc="
                f"{cosine(gradient, reference_gradient):.6f}"
            )

        print("[/CRITIC_EFFECT_METRICS]")

    finally:
        env.close()


if __name__ == "__main__":
    main()