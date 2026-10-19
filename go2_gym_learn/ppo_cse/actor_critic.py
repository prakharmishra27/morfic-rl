"""Actor–critic models for MorFiC and internal PPO baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from params_proto import PrefixProto
from torch.distributions import Normal

from .conditioning import (
    ActorFiLM,
    CriticFiLM,
    CriticAdaLN,
    CriticHyperNet,
)


# Each entry specifies (actor conditioning, critic conditioning).
VARIANTS = {
    "vanilla": ("none", "none"),
    "actor_concat": ("concat", "none"),
    "critic_concat": ("none", "concat"),
    "actor_critic_concat": ("concat", "concat"),
    "morfic_film": ("concat", "film"),
    "morfic_adaln": ("concat", "adaln"),
    "morfic_hypernet": ("concat", "hypernet"),
    "actor_critic_film": ("film", "film"),
}


class AC_Args(PrefixProto, cli=False):
    variant = "morfic_film"

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    # critic_hidden_dims = [1024, 512, 256]  ### for expanded morfic
    activation = "elu"

    adaptation_module_branch_hidden_dims = [256, 128]
    student_latent_dropout = 0.2
    use_decoder = False

    film_scale = 0.1
    hypernet_rank = 16
    hypernet_scale = 0.1

    critic_num_morph_bins = 8
    critic_morph_bin_by = "norm"


def get_activation(name):
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name!r}")
    return activations[name]()


def build_mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    last = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(last, width), get_activation(activation)])
        last = width
    layers.append(nn.Linear(last, output_dim))
    
    return nn.Sequential(*layers)

def build_actor(conditioning, base_input_dim, morph_latent_dim, num_actions):
    """Build either an ordinary MLP actor or a FiLM actor."""
    if conditioning == "film":
        return ActorFiLM(
            input_dim=base_input_dim,
            hidden_dims=AC_Args.actor_hidden_dims,
            cond_dim=morph_latent_dim,
            num_actions=num_actions,
            activation=nn.ELU,
            film_scale=AC_Args.film_scale,
        )

    if conditioning not in ("none", "concat"):
        raise ValueError(f"Unknown actor conditioning: {conditioning}")

    input_dim = base_input_dim
    if conditioning == "concat":
        input_dim += morph_latent_dim

    return build_mlp(
        input_dim,
        AC_Args.actor_hidden_dims,
        num_actions,
        AC_Args.activation,
    )


def build_critic(conditioning, base_input_dim, morph_latent_dim):
    """Build the value network with the selected conditioning."""
    if conditioning in ("none", "concat"):
        input_dim = base_input_dim
        if conditioning == "concat":
            input_dim += morph_latent_dim

        return build_mlp(
            input_dim,
            AC_Args.critic_hidden_dims,
            1,
            AC_Args.activation,
        )

    if conditioning == "film":
        return CriticFiLM(
            input_dim=base_input_dim,
            hidden_dims=AC_Args.critic_hidden_dims,
            cond_dim=morph_latent_dim,
            activation=nn.ELU,
            film_scale=AC_Args.film_scale,
        )

    if conditioning == "adaln":
        return CriticAdaLN(
            input_dim=base_input_dim,
            hidden_dims=AC_Args.critic_hidden_dims,
            cond_dim=morph_latent_dim,
            activation=nn.ELU,
            film_scale=AC_Args.film_scale,
        )

    if conditioning == "hypernet":
        return CriticHyperNet(
            input_dim=base_input_dim,
            hidden_dims=AC_Args.critic_hidden_dims,
            cond_dim=morph_latent_dim,
            activation=nn.ELU,
            rank=AC_Args.hypernet_rank,
            hyper_scale=AC_Args.hypernet_scale,
            use_layernorm=True,
        )

    raise ValueError(f"Unknown critic conditioning: {conditioning}")


class MorphologyEncoder(nn.Module):
    def __init__(self, morphology_dim, morphology_latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(morphology_dim, 128),
            nn.ELU(),
            nn.Linear(128, morphology_latent_dim),
            nn.ELU(),
        )

    def forward(self, morphology_vector):
        return self.net(morphology_vector)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_obs,
        num_privileged_obs,
        num_obs_history,
        num_actions,
        morphology_dim,
        morphology_latent_dim=64,
        variant=None,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(
                "ActorCritic: ignoring unexpected arguments:",
                list(kwargs),
            )

        self.variant = AC_Args.variant if variant is None else variant
        if self.variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant {self.variant!r}. "
                f"Choose from: {', '.join(VARIANTS)}"
            )

        self.actor_conditioning, self.critic_conditioning = (
            VARIANTS[self.variant]
        )

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        self.morphology_dim = morphology_dim
        self.student_latent_dropout = AC_Args.student_latent_dropout
        self.decoder = AC_Args.use_decoder

        if self.decoder:
            raise ValueError(
                "Decoder support is not implemented in this actor–critic."
            )

        self.env_factor_encoder = nn.Identity()

        # Vanilla has no morphology encoder, matching the original baseline.
        self.morph_encoder = None
        if self.variant != "vanilla":
            self.morph_encoder = MorphologyEncoder(
                morphology_dim, morphology_latent_dim
            )

        self.adaptation_module = build_mlp(
            num_obs_history,
            AC_Args.adaptation_module_branch_hidden_dims,
            num_privileged_obs,
            AC_Args.activation,
        )

        base_input_dim = num_obs_history + num_privileged_obs

        self.actor_body = build_actor(
            self.actor_conditioning,
            base_input_dim,
            morphology_latent_dim,
            num_actions,
        )

        self.critic_body = build_critic(
            self.critic_conditioning,
            base_input_dim,
            morphology_latent_dim,
        )

        self.std = nn.Parameter(
            AC_Args.init_noise_std * torch.ones(num_actions)
        )
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"ActorCritic variant: {self.variant}")
        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor network: {self.actor_body}")
        print(f"Critic network: {self.critic_body}")

    @staticmethod
    def init_weights(sequential, scales):
        linear_layers = [
            module for module in sequential
            if isinstance(module, nn.Linear)
        ]
        for module, scale in zip(linear_layers, scales):
            nn.init.orthogonal_(module.weight, gain=scale)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _prepare_inputs(self, obs_history, privileged_obs):
        device = self.std.device
        return obs_history.to(device), privileged_obs.to(device)

    def _encode_morphology(self, morphology_vector, batch_size):
        if self.morph_encoder is None:
            return None

        if morphology_vector is None:
            raise ValueError(
                f"{self.variant} requires a morphology descriptor."
            )

        morphology_vector = morphology_vector.to(
            device=self.std.device, dtype=self.std.dtype
        )
        if morphology_vector.ndim == 1:
            morphology_vector = morphology_vector.unsqueeze(0)

        if (
            morphology_vector.ndim != 2
            or morphology_vector.shape[-1] != self.morphology_dim
        ):
            raise ValueError(
                "Expected morphology shape "
                f"[batch, {self.morphology_dim}], "
                f"got {tuple(morphology_vector.shape)}"
            )

        if morphology_vector.shape[0] == 1:
            morphology_vector = morphology_vector.expand(batch_size, -1)
        elif morphology_vector.shape[0] != batch_size:
            raise ValueError(
                "Morphology batch size must be 1 or match observations."
            )

        return self.morph_encoder(morphology_vector)

    def _actor_mean(self, obs_history, latent, morphology_vector):
        actor_input = torch.cat((obs_history, latent), dim=-1)

        if self.actor_conditioning == "none":
            return self.actor_body(actor_input)

        z_morph = self._encode_morphology(
            morphology_vector, obs_history.shape[0]
        )

        if self.actor_conditioning == "concat":
            return self.actor_body(
                torch.cat((actor_input, z_morph), dim=-1)
            )

        return self.actor_body(actor_input, z_morph)

    def update_distribution(
        self, obs_history, privileged_obs, morphology_vector, **kwargs
    ):
        obs_history, privileged_obs = self._prepare_inputs(
            obs_history, privileged_obs
        )
        mean = self._actor_mean(
            obs_history, privileged_obs, morphology_vector
        )
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(
        self, obs_history, privileged_obs, morphology_vector, **kwargs
    ):
        self.update_distribution(
            obs_history, privileged_obs, morphology_vector, **kwargs
        )
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info=None):
        return self.act_teacher(
            ob["obs_history"],
            ob["privileged_obs"],
            ob["morphology_vector"],
            policy_info=policy_info,
        )

    def act_inference(self, ob, policy_info=None):
        if policy_info is None:
            policy_info = {}

        if ob.get("privileged_obs") is not None:
            gt_latent = self.env_factor_encoder(ob["privileged_obs"])
            policy_info["gt_latents"] = (
                gt_latent.detach().cpu().numpy()
            )

        return self.act_student(
            ob["obs"],
            ob["obs_history"],
            ob["morphology_vector"],
            policy_info=policy_info,
        )

    def act_student(
        self,
        observations,
        observation_history,
        morphology_vector,
        policy_info=None,
    ):
        observation_history = observation_history.to(self.std.device)
        latent = self.adaptation_module(observation_history)
        latent = F.dropout(
            latent,
            p=self.student_latent_dropout,
            training=self.training,
        )

        mean = self._actor_mean(
            observation_history, latent, morphology_vector
        )

        if policy_info is not None:
            policy_info["latents"] = latent.detach().cpu().numpy()
        return mean

    def act_teacher(
        self,
        observation_history,
        privileged_info,
        morphology_vector,
        policy_info=None,
    ):
        observation_history, privileged_info = self._prepare_inputs(
            observation_history, privileged_info
        )
        latent = self.env_factor_encoder(privileged_info)
        mean = self._actor_mean(
            observation_history, latent, morphology_vector
        )

        if policy_info is not None:
            policy_info["latents"] = latent.detach().cpu().numpy()
        return mean

    def evaluate(
        self, obs_history, privileged_obs, morphology_vector, **kwargs
    ):
        obs_history, privileged_obs = self._prepare_inputs(
            obs_history, privileged_obs
        )
        critic_input = torch.cat(
            (obs_history, privileged_obs), dim=-1
        )

        if self.critic_conditioning == "none":
            return self.critic_body(critic_input)

        z_morph = self._encode_morphology(
            morphology_vector, obs_history.shape[0]
        )

        if self.critic_conditioning == "concat":
            return self.critic_body(
                torch.cat((critic_input, z_morph), dim=-1)
            )

        return self.critic_body(critic_input, z_morph)

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)