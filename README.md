# MorFiC

**MorFiC: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer**  
Conference on Robot Learning (CoRL), 2026

Prakhar Mishra, Amir Hossain Raj, Xuesu Xiao, and Dinesh Manocha

[Paper](https://arxiv.org/abs/2603.14554) |
[Project page](https://prakharmishra27.github.io/MorFiC/)

## Overview

MorFiC studies critic miscalibration in cross-morphology reinforcement
learning. The main method concatenates a learned morphology representation
with the actor input and applies FiLM conditioning to the critic.

This repository contains training, simulation playback, evaluation scripts,
robot configurations, and a pretrained Go2-source MorFiC checkpoint.

Supported robot configurations:

- Unitree Go2
- Unitree Go1
- Unitree A1
- Unitree AlienGo
- Unitree B1
- Mini Cheetah
- ANYmal C

The paper includes physical deployment on Unitree Go1 and Go2.
The commands below cover simulation training, playback, and evaluation.

## Installation

The simulation checks were run with Python 3.8 and PyTorch 1.10.
An NVIDIA GPU, a compatible CUDA installation, and NVIDIA Isaac Gym
are required.

Install Isaac Gym and its Python bindings in your environment before
installing this repository. Isaac Gym is not installed by this package.

From the repository root:

```bash
python -m pip install -e .
```

For evaluation and representation plots:

```bash
python -m pip install -e ".[evaluation]"
```

For optional UMAP projections:

```bash
python -m pip install -e ".[evaluation,umap]"
```

Use an environment compatible with Isaac Gym. Installation of Isaac Gym
and PyTorch must be completed separately.

## Pretrained checkpoint

The released MorFiC checkpoint is located at:

```text
runs/gait-conditioned-agility/2026-05-15/train/003739.836224/
```

Playback expects:

```text
parameters.pkl
checkpoints/
    ac_weights_last.pt
```

Keep the accompanying JIT exports when distributing the pretrained run.
Keep the actuator-network assets required by the controller.

The checkpoint was trained using Go2 as the source robot. Selecting another
robot during playback uses that target robot's asset and morphology
descriptor without retraining the policy.

## Training

Run all commands from the repository root.

Train the main MorFiC model:

```bash
python scripts/train.py \
    --robot go2 \
    --variant morfic_film \
    --num-envs 4000 \
    --iterations 4000 \
    --headless
```

For a short execution check:

```bash
python scripts/train.py \
    --robot go2 \
    --variant morfic_film \
    --num-envs 256 \
    --iterations 20 \
    --save-interval 10 \
    --video-interval 0 \
    --headless
```

New runs are organized under `runs/morfic/` by robot, variant, seed,
and timestamp.

Available variants:

| Variant | Actor conditioning | Critic conditioning |
| --- | --- | --- |
| `vanilla` | None | None |
| `actor_concat` | Concatenation | None |
| `critic_concat` | None | Concatenation |
| `actor_critic_concat` | Concatenation | Concatenation |
| `morfic_film` | Concatenation | FiLM |
| `morfic_adaln` | Concatenation | AdaLN |
| `morfic_hypernet` | Concatenation | Hypernetwork |
| `actor_critic_film` | FiLM | FiLM |

Only the selected pretrained MorFiC checkpoint is included. Other variants
can be trained using the same training entry point.

### Morphology convention

The morphology descriptor has 11 entries:

```text
thigh mass, calf mass, hip mass, foot mass, base mass,
hip lower limit, hip upper limit,
thigh lower limit, thigh upper limit,
calf lower limit, calf upper limit
```

In this release, training samples morphology once per environment and retains
it across episode resets. The application path modifies body masses and
joint limits; it does not resize link geometry.

During evaluation, the selected target asset supplies the physical robot.
Its descriptor is normalized using the source training bounds. Out-of-range
descriptor values are not clipped.

## Playback

Play the released checkpoint on Go2:

```bash
python scripts/play.py --robot go2 --vx 1.0 --steps 800
```

Select a different target:

```bash
python scripts/play.py --robot go1 --vx 1.0 --steps 800
```

Available robot names are `go2`, `go1`, `a1`, `aliengo`, `b1`,
`mini_cheetah`, and `anymal_c`.

Use `--headless` to disable the viewer and `--plot` to display trajectory
plots.

To select another checkpoint:

```bash
python scripts/play.py \
    --run-dir /absolute/path/to/run \
    --robot go2 \
    --variant morfic_film
```

The run directory must contain `parameters.pkl` and `checkpoints/`.
For older checkpoints without saved variant metadata, specify `--variant`
when loading a variant other than the default MorFiC model.

## Evaluation

### Tracking and critic diagnostics

```bash
python -m scripts.evaluation.eval_vanilla_ppo_critic_metrics \
    --robot go2 --vx 1.0 --headless
```

Despite its historical filename, this script uses the shared model loader
and supports the configured policy variants.

Its return targets bootstrap from the critic at the rollout horizon.
Advantage-sign agreement uses centered returns as a reference proxy.

### Rollout export

```bash
python -m scripts.evaluation.export_rollout \
    --robot go2 --vx 1.0 --headless
```

Exports rewards, values, commands, and measured velocities to
`outputs/evaluation/`.

This script reports diagnostics against GAE-based targets. Its convention
cuts GAE and uses zero bootstrap at every done, including timeouts.

### Fixed-rollout critic comparison

```bash
python -m scripts.evaluation.eval_causal_advantage_update \
    --robot go2 --vx 1.0 --headless
```

Uses a fixed stochastic MorFiC teacher rollout to compare advantage signs
and actor-gradient directions. It does not perform optimizer updates.

Optional comparison checkpoints can be supplied with:

```bash
python -m scripts.evaluation.eval_causal_advantage_update \
    --robot go2 --headless \
    --critic actor_critic_concat=/absolute/path/to/concat/run
```

Comparison checkpoints must have compatible observation configurations
and morphology normalization bounds.

### Morphology representations

```bash
python -m scripts.evaluation.plot_morphology_representations \
    --rep encoder --headless
```

Saves representation data and PCA/t-SNE plots under
`outputs/representations/`.

Additional representations are `policy_hidden`, `critic_signature`, and
`action_signature`. Functional signatures can reuse a shared state batch:

```bash
python -m scripts.evaluation.plot_morphology_representations \
    --rep critic_signature \
    --state-file outputs/representations/shared_states.npz \
    --headless
```

These evaluation scripts use different return/reference conventions.
Their metrics should not be treated as interchangeable.

## Repository structure

```text
go2_gym/
    envs/
        base/          # Simulator and base configuration
        robots/        # Robot registry, configs, morphology, tracking env
        rewards/
        wrappers/
go2_gym_learn/
    ppo_cse/           # Shared actor-critic variants, PPO, rollout storage
resources/             # Robot and controller assets
scripts/
    train.py
    play.py
    evaluation/
runs/                  # Released pretrained run
LICENSES/              # Third-party notices
```

The Python package names retain the upstream Go2 naming.

## Deployment

For the upstream communication and hardware setup, see:

- [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways)
- [Walk These Ways Go2](https://github.com/Teddy-Liao/walk-these-ways-go2)

MorFiC additionally requires the target morphology descriptor, source
normalization bounds, morphology encoder, and matching actor input.
The upstream deployment instructions alone do not configure these
MorFiC-specific components.

## Citation

```bibtex
@inproceedings{mishra2026morfic,
  title={{MorFiC}: Fixing Value Miscalibration for Zero-Shot Quadruped Transfer},
  author={Mishra, Prakhar and Raj, Amir Hossain and Xiao, Xuesu and Manocha, Dinesh},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2026},
  url={https://arxiv.org/abs/2603.14554}
}
```

## Acknowledgments and license

This code builds on
[Walk These Ways](https://github.com/Improbable-AI/walk-these-ways)
and the
[Go2 adaptation by Dengting Liao](https://github.com/Teddy-Liao/walk-these-ways-go2),
with components derived from `legged_gym` and `rsl_rl`.

See the root `LICENSE` for the MIT license and `LICENSES/` for retained
third-party notices. Third-party assets and dependencies remain subject
to their respective licenses.