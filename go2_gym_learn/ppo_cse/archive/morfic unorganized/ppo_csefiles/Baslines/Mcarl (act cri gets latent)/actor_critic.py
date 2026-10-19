import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal
import torch.nn.functional as F


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False
    critic_num_morph_bins = 8
    critic_morph_bin_by = "norm"


# class FiLM(nn.Module):
#     """
#     Produces (gamma, beta) for a given hidden size.
#     We use residual FiLM: h <- h * (1 + gamma) + beta
#     and initialize gamma,beta ~ 0 so it starts as identity.
#     """
#     def __init__(self, cond_dim: int, hidden_dim: int, scale: float = 0.1):
#         super().__init__()
#         self.scale = scale
#         self.net = nn.Sequential(
#             nn.Linear(cond_dim, 128),
#             nn.ELU(),
#             nn.Linear(128, 2 * hidden_dim),
#         )
#         # init last layer near-zero => identity at start
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def forward(self, cond):
#         gb = self.net(cond) * self.scale
#         gamma, beta = gb.chunk(2, dim=-1)
#         return gamma, beta


# class CriticFiLM(nn.Module):
#     """
#     Critic trunk with FiLM modulation at each hidden layer.
#     """
#     def __init__(self, input_dim: int, hidden_dims, cond_dim: int, activation=nn.ELU, film_scale=0.1):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.films  = nn.ModuleList()
#         last = input_dim
#         for h in hidden_dims:
#             self.layers.append(nn.Linear(last, h))
#             self.films.append(FiLM(cond_dim, h, scale=film_scale))
#             last = h
#         self.act = activation()
#         self.v_head = nn.Linear(last, 1)

#     def forward(self, x, cond):
#         h = x
#         for lin, film in zip(self.layers, self.films):
#             h = lin(h)
#             gamma, beta = film(cond)
#             h = h * (1.0 + gamma) + beta
#             h = self.act(h)
#         return self.v_head(h)

    
class MorphologyEncoder(nn.Module):
    def __init__(self, morphology_dim, morphology_latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(morphology_dim, 128),
            nn.ELU(),
            nn.Linear(128, morphology_latent_dim),
            nn.ELU(),
        )

    def forward(self, morphology_vector: torch.Tensor) -> torch.Tensor:
        # Expect shape [B, morphology_dim]
        return self.net(morphology_vector)
    

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 morphology_dim,
                 morphology_latent_dim =64,                
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)
        self.env_factor_encoder = nn.Identity()

         ## morphology module
        self.morph_encoder = MorphologyEncoder (
            morphology_dim = morphology_dim,
            morphology_latent_dim = morphology_latent_dim 
        )
        import math

        

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history  + morphology_latent_dim, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history  + morphology_latent_dim, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)     


        ###morphic stuff
        # critic_input_dim = self.num_privileged_obs + self.num_obs_history  # NOTE: no z_morph concat
        # cond_dim = morphology_latent_dim  # we condition on z_morph

        # # print("condition dim;", {cond_dim})

        # self.critic_body = CriticFiLM(
        #     input_dim=critic_input_dim,
        #     hidden_dims=AC_Args.critic_hidden_dims,
        #     cond_dim=cond_dim,
        #     activation=nn.ELU,
        #     film_scale=0.1
        # )



        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

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
    
    # def update_distribution(self, observation_history):
    #     latent = self.adaptation_module(observation_history)
    #     mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
    #     self.distribution = Normal(mean, mean * 0. + self.std)

    def update_distribution(self, obs_history, privileged_obs, morphology_vector, **kwargs):
        device = next(self.parameters()).device

        obs_history = obs_history.to(device)
        privileged_obs = privileged_obs.to(device)

        # Teacher latent = privileged obs (matches PPO adaptation target)
        latent = privileged_obs

        morphology_vector = morphology_vector.to(device)
        if morphology_vector.dim() == 1:
            morphology_vector = morphology_vector.unsqueeze(0)  # [1, morph_dim]

        z_morph = self.morph_encoder(morphology_vector)

        # If a single morph was provided for the whole batch, expand it
        if z_morph.shape[0] == 1 and obs_history.shape[0] > 1:
            z_morph = z_morph.expand(obs_history.shape[0], -1)

        mean = self.actor_body(torch.cat((obs_history, latent, z_morph), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)


    # def act(self, observations,privileged_observations, morphology_vector, **kwargs):
    #     self.update_distribution(observations, privileged_observations, morphology_vector)
    #     return self.distribution.sample()
    def act(self, obs_history, privileged_obs, morphology_vector, **kwargs):
        self.update_distribution(obs_history, privileged_obs, morphology_vector, **kwargs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # def act_expert(self, ob, policy_info={}):
    #     return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(
                    ob["obs_history"],
                    ob["privileged_obs"],
                    ob["morphology_vector"],
                    policy_info=policy_info,
                )


    # def act_inference(self, ob, policy_info={}):
    #     return self.act_student(ob["obs_history"], policy_info=policy_info)
    def act_inference(self, ob, policy_info={}):
        if ob["privileged_obs"] is not None:
            gt_latent = self.env_factor_encoder(ob["privileged_obs"])
            policy_info["gt_latents"] = gt_latent.detach().cpu().numpy()
        return  self.act_student(ob["obs"], ob["obs_history"], ob["morphology_vector"])
    

    # def act_student(self, observation_history, policy_info={}):
    #     latent = self.adaptation_module(observation_history)
    #     actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
    #     policy_info["latents"] = latent.detach().cpu().numpy()
    #     return actions_mean

    def act_student(self, observations, observation_history, morphology_vector, policy_info={}):

        if not hasattr(self, "_morph_dbg_printed"):
            self._morph_dbg_printed = True
            mv = morphology_vector
            print("[policy] morphology_vector type:", type(mv))
            if mv is None:
                print("[policy] morphology_vector is None")
            elif torch.is_tensor(mv):
                print("[policy] morphology_vector shape:", tuple(mv.shape),
                    "device:", mv.device, "dtype:", mv.dtype)
                # stats (guard against empty)
                if mv.numel() > 0:
                    print("[policy] morphology_vector stats: min/max/mean:",
                        mv.min().item(), mv.max().item(), mv.mean().item())
                    print("[policy] morphology_vector has NaN?", torch.isnan(mv).any().item())
            else:
                print("[policy] morphology_vector is NOT a tensor:", mv)
                
        latent = self.adaptation_module(observation_history)
        latent = nn.functional.dropout(latent, p=0.2, training=self.training)

        device = next(self.parameters()).device
        z_morphology = self.morph_encoder(morphology_vector.to(device))

        if z_morphology.dim() == 1:
            z_morphology = z_morphology.unsqueeze(0)

        # make morph batch match the *history* batch
        if z_morphology.shape[0] != observation_history.shape[0]:
            z_morphology = z_morphology.expand(observation_history.shape[0], -1)

        # print("obs_hist:", observation_history.shape)
        # print("latent:", latent.shape)
        # print("z_morph:", z_morphology.shape)

        # IMPORTANT: use observation_history, not observations
        actions_mean = self.actor_body(
            torch.cat((observation_history, latent, z_morphology), dim=-1)
        )

        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean


    # def act_teacher(self, observation_history, privileged_info, policy_info={}):
    #     actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
    #     policy_info["latents"] = privileged_info
    #     return actions_mean

    def act_teacher(self, observation_history, privileged_info, morphology_vector, policy_info={}):
        latent = self.env_factor_encoder(privileged_info)
        device = next(self.parameters()).device
        z_morphology = self.morph_encoder(morphology_vector.to(device))

        if z_morphology.dim() == 1:
            z_morphology = z_morphology.unsqueeze(0)
        if z_morphology.shape[0] != observation_history.shape[0]:
            z_morphology = z_morphology.expand(observation_history.shape[0], -1)

        actions_mean = self.actor_body(
            torch.cat((observation_history, latent, z_morphology), dim=-1)
        )
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean


    # def evaluate(self, observation_history, privileged_observations, **kwargs):
    #     value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
    #     return value
    # def evaluate(self, obs_history, privileged_obs, morphology_vector, **kwargs):
    #     device = next(self.parameters()).device
    #     obs_history = obs_history.to(device)
    #     privileged_obs = privileged_obs.to(device)

    #     latent = privileged_obs

    #     morphology_vector = morphology_vector.to(device)
    #     if morphology_vector.dim() == 1:
    #         morphology_vector = morphology_vector.unsqueeze(0)

    #     z_morph = self.morph_encoder(morphology_vector)
    #     if z_morph.shape[0] == 1 and obs_history.shape[0] > 1:
    #         z_morph = z_morph.expand(obs_history.shape[0], -1)

    #     value = self.critic_body(torch.cat((obs_history, latent, z_morph), dim=-1))
    #     return value

    def evaluate(self, obs_history, privileged_obs, morphology_vector, **kwargs):
        device = next(self.parameters()).device
        obs_history = obs_history.to(device)
        privileged_obs = privileged_obs.to(device)

        latent = privileged_obs

        morphology_vector = morphology_vector.to(device)
        if morphology_vector.dim() == 1:
            morphology_vector = morphology_vector.unsqueeze(0)

        # z_morph = self.morph_encoder(morphology_vector)
        z_morph = self.morph_encoder(morphology_vector)

        if z_morph.shape[0] == 1 and obs_history.shape[0] > 1:
            z_morph = z_morph.expand(obs_history.shape[0], -1)

        critic_in = torch.cat((obs_history, latent, z_morph), dim=-1)

        value = self.critic_body(torch.cat((obs_history, latent, z_morph), dim=-1))

        # critic_in = torch.cat((obs_history, latent), dim=-1)  # no z_morph here
        # value = self.critic_body(critic_in, z_morph)    

        # print("zmorph evla dim;", {z_morph})

        return value



    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
