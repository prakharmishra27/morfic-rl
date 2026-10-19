#!/usr/bin/env python3
"""Plot learned morphology representations for one checkpoint per process."""

import argparse
from importlib import import_module
from pathlib import Path

import isaacgym  # noqa: F401 — before torch

import numpy as np
import torch
from tqdm import tqdm

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
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--robot", choices=ROBOT_CONFIGS, default="go2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--rep",
        choices=(
            "encoder",
            "policy_hidden",
            "critic_signature",
            "action_signature",
        ),
        default="encoder",
    )
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--state-steps", type=int, default=128)
    parser.add_argument("--max-states", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(MINI_GYM_ROOT_DIR) / "outputs" / "representations",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("pca", "tsne", "umap"),
        default=("pca", "tsne"),
    )

    parser.add_argument("--vx", type=float, default=2.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--gait", choices=GAITS, default="trotting")
    parser.add_argument("--frequency", type=float, default=3.0)
    parser.add_argument("--body-height", type=float, default=0.0)
    parser.add_argument("--footswing-height", type=float, default=0.08)
    parser.add_argument("--stance-width", type=float, default=0.25)
    parser.add_argument("--stance-length", type=float, default=0.40)

    args = parser.parse_args()
    for name in (
        "n_train", "num_envs", "state_steps", "max_states", "batch_size"
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.frequency <= 0:
        parser.error("--frequency must be positive.")

    args.run_dir = args.run_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    return args


def named_descriptors():
    """Read physical descriptors without mutating the active configuration."""
    names, descriptors = [], []
    for name, (module_name, _) in ROBOT_CONFIGS.items():
        module = import_module(
            module_name, package="go2_gym.envs.robots"
        )
        descriptor = np.asarray(
            module.MORPHOLOGY_DESCRIPTOR, dtype=np.float32
        )
        if descriptor.shape != (11,) or not np.isfinite(descriptor).all():
            raise ValueError(f"Invalid morphology descriptor for {name}.")
        names.append(name)
        descriptors.append(descriptor)
    return names, np.stack(descriptors)


@torch.no_grad()
def get_states(env, policy, args):
    if args.state_file is not None and args.state_file.exists():
        with np.load(args.state_file, allow_pickle=False) as data:
            hist = torch.as_tensor(
                data["hist"], device=env.device, dtype=torch.float32
            )
            priv = torch.as_tensor(
                data["priv"], device=env.device, dtype=torch.float32
            )
    else:
        obs = env.reset()
        histories, privileged = [], []

        for _ in range(args.state_steps):
            set_commands(env, args)
            obs = refresh_command_observations(env, obs)

            priv = obs.get("privileged_obs")
            if priv is None:
                priv = env.get_privileged_observations()
            if priv is None:
                raise RuntimeError("Privileged observations are required.")

            histories.append(obs["obs_history"].detach().cpu().clone())
            privileged.append(priv.detach().cpu().clone())
            obs, _, _, _ = env.step(policy(obs))

        hist = torch.cat(histories).to(env.device)
        priv = torch.cat(privileged).to(env.device)

    if (
        hist.ndim != 2
        or priv.ndim != 2
        or hist.shape[0] != priv.shape[0]
        or hist.shape[1] != env.num_obs_history
        or priv.shape[1] != env.num_privileged_obs
        or hist.shape[0] == 0
    ):
        raise ValueError("Shared-state dimensions do not match this model.")

    if hist.shape[0] > args.max_states:
        indices = torch.linspace(
            0, hist.shape[0] - 1, args.max_states,
            device=env.device,
        ).long()
        hist, priv = hist[indices], priv[indices]

    if args.state_file is not None and not args.state_file.exists():
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.state_file,
            hist=hist.cpu().numpy(),
            priv=priv.cpu().numpy(),
            source_run=str(args.run_dir),
            source_robot=args.robot,
            seed=args.seed,
        )

    return hist, priv


def actor_output_layer(model):
    body = model.actor_body
    if isinstance(body, torch.nn.Sequential):
        return list(body.children())[-1]
    if hasattr(body, "action_head"):
        return body.action_head
    raise ValueError("Cannot locate this actor's final action layer.")


@torch.no_grad()
def extract_representation(model, env, morphology, policy, args):
    if args.rep == "encoder":
        if getattr(model, "morph_encoder", None) is None:
            raise ValueError("This variant has no morphology encoder.")
        chunks = [
            model.morph_encoder(batch).cpu().numpy()
            for batch in morphology.split(args.batch_size)
        ]
        return np.concatenate(chunks)

    if args.rep == "policy_hidden":
        # Capture the feature entering the actor's final action layer.
        features = []

        def capture_input(module, inputs):
            features.append(inputs[0].detach().cpu().numpy().copy())

        handle = actor_output_layer(model).register_forward_pre_hook(
            capture_input
        )
        try:
            for batch in morphology.split(args.batch_size):
                count = len(batch)
                hist = torch.zeros(
                    count, env.num_obs_history, device=env.device
                )
                obs = torch.zeros(
                    count, env.num_obs, device=env.device
                )
                model.act_student(obs, hist, batch)
        finally:
            handle.remove()

        return np.concatenate(features)

    hist, priv = get_states(env, policy, args)
    signatures = []

    for descriptor in tqdm(morphology, desc=args.rep):
        parts = []
        for start in range(0, len(hist), args.batch_size):
            h = hist[start:start + args.batch_size]
            p = priv[start:start + args.batch_size]
            m = descriptor.unsqueeze(0).expand(len(h), -1)

            if args.rep == "critic_signature":
                result = model.evaluate(h, p, m)
            else:
                # Preserve the original teacher-conditioned action-mean
                # signature, but avoid sampling actions unnecessarily.
                model.update_distribution(h, p, m)
                result = model.action_mean

            parts.append(result.detach().cpu().numpy().reshape(-1))

        signatures.append(np.concatenate(parts))

    representation = np.stack(signatures).astype(np.float32)

    # Preserve the original per-feature standardization for signatures.
    return (
        representation - representation.mean(axis=0, keepdims=True)
    ) / (representation.std(axis=0, keepdims=True) + 1e-8)


def project(representation, method, seed):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(
            representation
        )

    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2,
            perplexity=min(30.0, float(len(representation) - 1)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(representation)

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP requires the optional umap-learn package."
        ) from exc

    return umap.UMAP(
        n_components=2,
        n_neighbors=min(25, len(representation) - 1),
        min_dist=0.10,
        random_state=seed,
    ).fit_transform(representation)


def distance_correlation(representation, morphology):
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    # Unique pairs only; exclude diagonal zeros and duplicated pairs.
    latent_distance = pdist(representation)
    physical_distance = pdist(morphology)

    if np.ptp(latent_distance) == 0 or np.ptp(physical_distance) == 0:
        return float("nan")
    return float(spearmanr(latent_distance, physical_distance).statistic)


def plot_projection(points, names, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        points[:args.n_train, 0],
        points[:args.n_train, 1],
        s=8, alpha=0.22,
        label="Sampled training morphologies",
    )

    for index, name in enumerate(names, start=args.n_train):
        ax.scatter(
            *points[index],
            s=130,
            marker="X" if name in ("aliengo", "anymal_c", "b1") else "*",
            edgecolors="black",
            linewidths=0.7,
            label=name,
        )
        ax.annotate(
            name, points[index],
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )

    ax.set(
        title=f"{args.rep}: {path.stem.rsplit('_', 1)[-1].upper()}",
        xlabel="Dimension 1", ylabel="Dimension 2",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
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
            env, parameters, args.run_dir, args.variant,
            return_model=True,
        )
        lower = env.morph_min.detach().cpu().numpy()
        upper = env.morph_max.detach().cpu().numpy()
        width = upper - lower

        if lower.shape != (11,) or not np.all(width > 0):
            raise ValueError("Invalid source training morphology bounds.")

        names, robot_raw = named_descriptors()
        rng = np.random.default_rng(args.seed)
        train_raw = rng.uniform(
            lower, upper, size=(args.n_train, 11)
        ).astype(np.float32)
        raw = np.concatenate((train_raw, robot_raw))

        # Exactly the input convention used by training. Do not clip OOD bots.
        normalized = (raw - lower) / width
        morphology = torch.as_tensor(
            normalized, device=env.device, dtype=torch.float32
        )

        for name, descriptor in zip(names, robot_raw):
            excess = (
                np.maximum(lower - descriptor, 0)
                + np.maximum(descriptor - upper, 0)
            ) / width
            print(
                f"{name}: outside-range score={np.linalg.norm(excess):.3f}, "
                f"outside indices={np.flatnonzero(excess > 1e-9).tolist()}"
            )

        representation = extract_representation(
            model, env, morphology, policy, args
        )
        if not np.isfinite(representation).all():
            raise ValueError("Representation contains nonfinite values.")

        # Report both conventions instead of hiding the effect of units.
        print(
            "Distance Spearman rho, raw morphology: "
            f"{distance_correlation(representation, raw):.4f}"
        )
        print(
            "Distance Spearman rho, normalized morphology: "
            f"{distance_correlation(representation, normalized):.4f}"
        )
        print("Distance correlations are descriptive; pairs are dependent.")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        variant = args.variant or parameters.get(
            "AC_Args", {}
        ).get("variant") or "morfic_film"
        prefix = f"{variant}_{args.rep}"

        np.savez_compressed(
            args.out_dir / f"{prefix}.npz",
            representation=representation,
            morphology_raw=raw,
            morphology_normalized=normalized,
            morphology_min=lower,
            morphology_max=upper,
            robot_names=np.asarray(names),
            num_train=args.n_train,
            seed=args.seed,
            run_dir=str(args.run_dir),
            state_robot=args.robot,
            signature_actor="teacher_mean",
        )

        for method in args.methods:
            points = project(representation, method, args.seed)
            path = args.out_dir / f"{prefix}_{method}.pdf"
            plot_projection(points, names, args, path)
            np.save(args.out_dir / f"{prefix}_{method}.npy", points)
            print(f"Saved: {path}")

    finally:
        env.close()


if __name__ == "__main__":
    main()