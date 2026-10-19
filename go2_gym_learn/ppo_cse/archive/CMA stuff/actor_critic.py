import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False



class MorphologyEncoder(nn.Module):
    def __init__(self, morphology_dim, morphology_latent_dim =64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(morphology_dim, 128),
            nn.ELU(),
            nn.Linear(128, morphology_latent_dim),
            nn.ELU(),
        )

    def forward(self, morphology_vector):
        ## shape [batch_size, ]
        return self.net(morphology_vector)
    

class BehaviorEncoder(nn.Module):
    def __init__(self, behavior_dim, behavior_latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(behavior_dim, 128),
            nn.ELU(),
            nn.Linear(128, behavior_latent_dim),
            nn.ELU(),
        )

    def forward(self, behavior_vector):
        # behavior_vector: [batch_size, behavior_dim]
        return self.net(behavior_vector)

    


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 morphology_dim,
                 morphology_latent_dim =64,
                 behavior_dim=8,             
                 behavior_latent_dim=64,     
                 cma_latent_dim=64, 
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

         # --- Behavior module (for CMA), if provided ---
        if behavior_dim is not None:
            if behavior_latent_dim is None:
                behavior_latent_dim = morphology_latent_dim
            if cma_latent_dim is None:
                cma_latent_dim = morphology_latent_dim

            self.behavior_encoder = BehaviorEncoder(
                behavior_dim=behavior_dim,
                behavior_latent_dim=behavior_latent_dim,
            )

            # projection heads into a shared CMA latent space
            self.morph_head = nn.Sequential(
                nn.Linear(morphology_latent_dim, cma_latent_dim),
                nn.ELU(),
            )
            self.behavior_head = nn.Sequential(
                nn.Linear(behavior_latent_dim, cma_latent_dim),
                nn.ELU(),
            )

            self.cma_latent_dim = cma_latent_dim
        else:
            self.behavior_encoder = None
            self.morph_head = None
            self.behavior_head = None
            self.cma_latent_dim = None


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
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history + morphology_latent_dim, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history + morphology_latent_dim, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        # print(f"Adaptation Module: {self.adaptation_module}")
        # print(f"Actor MLP: {self.actor_body}")
        # print(f"Critic MLP: {self.critic_body}")

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


    def update_distribution(self, observations, privileged_observations, morphology_vector):
        # print(f"Observations shape: {observations.shape}")
        # print(f"Privileged observations shape: {privileged_observations.shape}")

        latent = self.env_factor_encoder(privileged_observations)
        # print(f"Latent shape: {latent.shape}")

        device = next(self.parameters()).device

        z_morphology = self.morph_encoder(morphology_vector.to(device))

        if z_morphology.dim() == 1:
            z_morphology = z_morphology.unsqueeze(0)

        if z_morphology.shape[0] != observations.shape[0]:
            z_morphology = z_morphology.expand(observations.shape[0], -1)


        # print("obs:", observations.shape)
        # print("latent:", latent.shape)
        # print("z_morph:", z_morphology.shape)



        mean = self.actor_body(torch.cat((observations, latent,z_morphology), dim=-1))
        # print(f"Actor input shape: {torch.cat((observations, latent), dim=-1).shape}")
        self.distribution = Normal(mean, mean * 0. + self.std)


    def act(self, observations, privileged_observations, morphology_vector, **kwargs):
        self.update_distribution(observations, privileged_observations, morphology_vector)
        # print("actor critic act morph vect :", morphology_vector)
        return self.distribution.sample()
    
    

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(
            ob["obs"],               # or ob["obs_history"], depending on what you want teacher to use
            ob["privileged_obs"],
            ob["morphology_vector"],
            policy_info=policy_info,
        )


    # def act_inference(self, ob, policy_info={}):
    #     return self.act_student(ob["obs_history"], ob["morphology_vector"], policy_info=policy_info)
    def act_inference(self, ob, policy_info={}):
        if ob["privileged_obs"] is not None:
            gt_latent = self.env_factor_encoder(ob["privileged_obs"])
            policy_info["gt_latents"] = gt_latent.detach().cpu().numpy()
        return  self.act_student(ob["obs"], ob["obs_history"], ob["morphology_vector"])
    


    # def act_student(self, observations, observation_history, morphology_vector, policy_info={}):
    #     # print(f"Student Observations shape: {observations.shape}")
    #     # print(f"Observation history shape: {observation_history.shape}")
    #     latent = self.adaptation_module(observation_history)
    #     latent = nn.functional.dropout(latent, p=0.2, training=self.training)  ### for robust deployment
    #     # print(f"Adaptation module latent shape: {latent.shape}")

    #     device = next(self.parameters()).device

    #     z_morphology = self.morph_encoder(morphology_vector.to(device))

    #     if z_morphology.dim() == 1:
    #         z_morphology = z_morphology.unsqueeze(0)

    #     if z_morphology.shape[0] != observations.shape[0]:
    #         z_morphology = z_morphology.expand(observations.shape[0], -1)


    #     print("obs:", observations.shape)
    #     print("latent:", latent.shape)
    #     print("z_morph:", z_morphology.shape)

        

    #     actions_mean = self.actor_body(torch.cat((observations, latent, z_morphology), dim=-1))
    #     # print(f"Actor input shape: {torch.cat((observations, latent), dim=-1).shape}")
        
    #     policy_info["latents"] = latent.detach().cpu().numpy()
    #     return actions_mean

    def act_student(self, observations, observation_history, morphology_vector, policy_info={}):
        latent = self.adaptation_module(observation_history)
        latent = nn.functional.dropout(latent, p=0.2, training=self.training)

        device = next(self.parameters()).device
        z_morphology = self.morph_encoder(morphology_vector.to(device))

        if z_morphology.dim() == 1:
            z_morphology = z_morphology.unsqueeze(0)

        # make morph batch match the *history* batch
        if z_morphology.shape[0] != observation_history.shape[0]:
            z_morphology = z_morphology.expand(observation_history.shape[0], -1)

        print("obs_hist:", observation_history.shape)
        print("latent:", latent.shape)
        print("z_morph:", z_morphology.shape)

        # IMPORTANT: use observation_history, not observations
        actions_mean = self.actor_body(
            torch.cat((observation_history, latent, z_morphology), dim=-1)
        )

        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean



    # def act_teacher(self, observations, privileged_info, morphology_vector, policy_info={}):
    #     latent = self.env_factor_encoder(privileged_info)
    #     device = next(self.parameters()).device
    #     z_morphology = self.morph_encoder(morphology_vector.to(device))
    #     if z_morphology.dim() == 1:
    #         z_morphology = z_morphology.unsqueeze(0)
    #     if z_morphology.shape[0] != observations.shape[0]:
    #         z_morphology = z_morphology.expand(observations.shape[0], -1)
    #     actions_mean = self.actor_body(torch.cat((observations, latent, z_morphology), dim=-1))
    #     policy_info["latents"] = latent.detach().cpu().numpy()
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



    def evaluate(self, critic_observations, privileged_observations, morphology_vector, **kwargs):
        latent = self.env_factor_encoder(privileged_observations)
        device = next(self.parameters()).device
        z_morphology = self.morph_encoder(morphology_vector.to(device))

        if z_morphology.dim() == 1:
            z_morphology = z_morphology.unsqueeze(0)

        if z_morphology.shape[0] != critic_observations.shape[0]:
            z_morphology = z_morphology.expand(critic_observations.shape[0], -1)


        print("critic_observations:", critic_observations.shape)
        # print("latent:", latent.shape)
        # print("z_morph:", z_morphology.shape)
        value = self.critic_body(torch.cat((critic_observations, latent, z_morphology), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)
    
        # ---------- CMA helpers ----------

    def encode_morph_for_cma(self, morphology_vector, detach: bool = True):
        """
        morphology_vector: [N, morphology_dim] or [morphology_dim]
        returns: [N, cma_latent_dim]
        """
        if self.morph_head is None:
            raise RuntimeError("Morph CMA head not initialized (behavior_dim was None).")

        device = next(self.parameters()).device
        z_m = self.morph_encoder(morphology_vector.to(device))
        if z_m.dim() == 1:
            z_m = z_m.unsqueeze(0)
        z_m = self.morph_head(z_m)
        return z_m.detach() if detach else z_m

    def encode_behavior_for_cma(self, behavior_vector, detach: bool = True):
        """
        behavior_vector: [N, behavior_dim] or [behavior_dim]
        returns: [N, cma_latent_dim]
        """
        if self.behavior_encoder is None or self.behavior_head is None:
            raise RuntimeError("Behavior CMA head not initialized (behavior_dim was None).")

        device = next(self.parameters()).device
        z_b = self.behavior_encoder(behavior_vector.to(device))
        if z_b.dim() == 1:
            z_b = z_b.unsqueeze(0)
        z_b = self.behavior_head(z_b)
        return z_b.detach() if detach else z_b


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
