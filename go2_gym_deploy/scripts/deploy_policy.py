#!/usr/bin/env python3
"""Deploy the released MorFiC policy with a Go1 or Go2 morphology input."""

import argparse
import pickle as pkl
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUN = (
    PROJECT_ROOT
    / "runs"
    / "gait-conditioned-agility"
    / "2026-05-15"
    / "train"
    / "003739.836224"
)

MORPHOLOGIES = {
    "go1": [
        0.898919, 0.158015, 0.510299, 0.06, 4.8,
        -0.8028514559, 0.8028514559,
        -1.0471975512, 4.1887902048,
        -2.6965336943, -0.9162978573,
    ],
    "go2": [
        1.152, 0.154, 0.678, 0.04, 6.921,
        -1.0472, 1.0472,
        -1.5708, 3.4907,
        -2.7227, -0.83776,
    ],
}

# Source training bounds for the released checkpoint.
MORPH_MIN = [
    0.634, 0.064, 0.510299, 0.0, 3.3,
    -1.10, 0.80, -1.65, 3.49, -2.72, -0.95,
]

MORPH_MAX = [
    1.152, 0.166, 0.696, 0.06, 6.921,
    -0.80, 1.10, -1.05, 4.05, -2.60, -0.78,
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=MORPHOLOGIES, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--experiment-name", default="morfic_deployment")
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-yaw-vel", type=float, default=1.0)
    parser.add_argument("--y-scale", type=float, default=0.6)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Test JIT inference without initializing hardware communication.",
    )

    args = parser.parse_args()
    if args.max_steps < 1 or args.control_dt <= 0:
        parser.error("max-steps and control-dt must be positive.")
    if min(args.max_vel, args.max_yaw_vel, args.y_scale) < 0:
        parser.error("Command scales must be nonnegative.")

    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def load_policy(run_dir, robot_name, cfg):
    checkpoint_dir = run_dir / "checkpoints"

    body = torch.jit.load(
        str(checkpoint_dir / "body_latest.jit"),
        map_location="cpu",
    ).eval()

    adaptation_module = torch.jit.load(
        str(checkpoint_dir / "adaptation_module_latest.jit"),
        map_location="cpu",
    ).eval()

    morph_encoder = torch.jit.load(
        str(checkpoint_dir / "morph_encoder_latest.jit"),
        map_location="cpu",
    ).eval()

    morph_raw = torch.tensor(
        [MORPHOLOGIES[robot_name]], dtype=torch.float32
    )

    saved_morphology = cfg.get("Morphology", {})
    lower = saved_morphology.get("morphology_min")
    upper = saved_morphology.get("morphology_max")

    if lower is None and upper is None:
        lower, upper = MORPH_MIN, MORPH_MAX
    elif lower is None or upper is None:
        raise ValueError("Saved morphology bounds are incomplete.")

    lower = torch.as_tensor(lower, dtype=torch.float32).cpu()
    upper = torch.as_tensor(upper, dtype=torch.float32).cpu()

    if lower.shape != (11,) or upper.shape != (11,):
        raise ValueError("Morphology bounds must each contain 11 values.")
    if not (
        torch.isfinite(lower).all()
        and torch.isfinite(upper).all()
        and torch.all(upper > lower)
    ):
        raise ValueError("Invalid morphology normalization bounds.")

    morph_vec = (morph_raw - lower) / (upper - lower)

    with torch.no_grad():
        z_morph = morph_encoder(morph_vec)

    print(f"[deploy] Robot morphology: {robot_name}")
    print(f"[deploy] Normalized vector: {morph_vec}")

    @torch.no_grad()
    def policy(obs, info):
        history = obs["obs_history"].to(
            device="cpu", dtype=torch.float32
        )
        if history.ndim == 1:
            history = history.unsqueeze(0)

        latent = adaptation_module(history)
        z = z_morph.expand(history.shape[0], -1)

        actions = body(torch.cat((history, latent, z), dim=-1))

        if not torch.isfinite(actions).all().item():
            raise RuntimeError("Policy produced nonfinite actions.")

        info["latent"] = latent
        return actions

    return policy


def main():
    args = parse_args()

    with (args.run_dir / "parameters.pkl").open("rb") as file:
        parameters = pkl.load(file)
    cfg = parameters["Cfg"]

    variant = parameters.get("AC_Args", {}).get("variant")
    if variant is not None and variant != "morfic_film":
        raise ValueError("This script expects the released morfic_film policy.")

    training_dt = (
        float(cfg["sim"]["dt"])
        * int(cfg["control"]["decimation"])
    )
    if abs(training_dt - args.control_dt) > 1e-6:
        raise ValueError(
            f"Control interval mismatch: training={training_dt}, "
            f"deployment={args.control_dt}."
        )

    policy = load_policy(args.run_dir, args.robot, cfg)

    history_size = (
        int(cfg["env"]["num_observations"])
        * int(cfg["env"]["num_observation_history"])
    )
    actions = policy(
        {"obs_history": torch.zeros(1, history_size)}, {}
    )
    expected_actions = int(cfg["env"].get("num_actions", 12))
    if tuple(actions.shape) != (1, expected_actions):
        raise RuntimeError(f"Unexpected action shape: {tuple(actions.shape)}")

    print(f"[deploy] JIT inference passed: {tuple(actions.shape)}")

    if args.check_only:
        return

    import lcm

    from go2_gym_deploy.utils.deployment_runner import DeploymentRunner
    from go2_gym_deploy.envs.lcm_agent import LCMAgent
    from go2_gym_deploy.utils.cheetah_state_estimator import StateEstimator
    from go2_gym_deploy.utils.command_profile import RCControllerProfile
    from go2_gym_deploy.envs.history_wrapper import HistoryWrapper

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
    state_estimator = StateEstimator(lc)

    command_profile = RCControllerProfile(
        dt=args.control_dt,
        state_estimator=state_estimator,
        x_scale=args.max_vel,
        y_scale=args.y_scale,
        yaw_scale=args.max_yaw_vel,
    )

    hardware_agent = LCMAgent(
        cfg, state_estimator, command_profile
    )
    state_estimator.spin()
    hardware_agent = HistoryWrapper(hardware_agent)

    log_root = PROJECT_ROOT / "logs" / args.experiment_name
    log_root.mkdir(parents=True, exist_ok=True)

    runner = DeploymentRunner(
        experiment_name=args.experiment_name,
        se=None,
        log_root=str(log_root),
    )
    runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    runner.add_policy(policy)
    runner.add_command_profile(command_profile)

    print(
        "[deploy] --robot selects morphology only. The hardware bridge, "
        "joint mapping, and controller gains must match the actual robot."
    )
    runner.run(max_steps=args.max_steps, logging=True)


if __name__ == "__main__":
    main()
