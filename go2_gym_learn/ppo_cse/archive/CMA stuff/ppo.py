import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go2_gym_learn.ppo_cse import ActorCritic
from go2_gym_learn.ppo_cse import RolloutStorage
from go2_gym_learn.ppo_cse import caches


class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.
    cma_coef = 0.1

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device
        self.global_cma_step = 0

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        # self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
        #                                               lr=PPO_Args.adaptation_module_learning_rate)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)

        self.adaptation_module_optimizer = optim.Adam(
            self.actor_critic.adaptation_module.parameters(),
            lr=PPO_Args.adaptation_module_learning_rate,
        )

        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate
        self.beh_episode_fifo = []  ## for beh to pass

    def push_behavior_episodes(self, env_ids, b):  # b: [k, 8]
        self.beh_episode_fifo.append((env_ids, b))

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape, morphology_vector):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, morphology_vector, self.device, behavior_dim=8)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    # def act(self, obs, privileged_obs, obs_history, morphology_vector = None):
    #     # Compute the actions and values
    #     self.transition.actions = self.actor_critic.act(obs_history, morphology_vector).detach()
    #     self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs, morphology_vector).detach()
    #     self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
    #     self.transition.action_mean = self.actor_critic.action_mean.detach()
    #     self.transition.action_sigma = self.actor_critic.action_std.detach()
    #     # need to record obs and critic_obs before env.step()
    #     self.transition.observations = obs
    #     self.transition.critic_observations = obs
    #     self.transition.privileged_observations = privileged_obs
    #     self.transition.observation_histories = obs_history
    #     self.transition.morphology_vector = morphology_vector.detach()
    #     print("PPO Morph vector:", morphology_vector)  # print to check
    #     print("PPO Morph vector shape:", morphology_vector.shape)
    #     return self.transition.actions

    def act(self, obs, privileged_obs, obs_history, morphology_vector=None):
        # Compute the actions and values using TEACHER policy
        # observations     = obs_history
        # privileged_obs   = privileged_obs
        # morphology_vector= morphology_vector

        self.transition.actions = self.actor_critic.act(
            obs_history,
            privileged_obs,
            morphology_vector,
        ).detach()

        self.transition.values = self.actor_critic.evaluate(
            obs_history,
            privileged_obs,
            morphology_vector,
        ).detach()

        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # record stuff into storage
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.morphology_vector = morphology_vector.detach()

        # print("PPO Morph vector:", morphology_vector)
        # print("PPO Morph vector shape:", morphology_vector.shape)
        return self.transition.actions


    def process_env_step(self, rewards, dones, infos, behavior_vector=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        if behavior_vector is not None:                                       
            self.transition.behavior_vector = behavior_vector.detach()   
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs,last_morph_vector):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs,last_morph_vector).detach()
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)


    # def compute_cma_loss(self, morphology_vector_batch, behavior_vector_batch):
    #     N = morphology_vector_batch.shape[0]
    #     max_cma_batch = 256  # even 256 is fine to start

    #     # ---- subsample batch for CMA (alignment) ----
    #     if N > max_cma_batch:
    #         idx = torch.randperm(N, device=morphology_vector_batch.device)[:max_cma_batch]
    #         morph_cma = morphology_vector_batch[idx]
    #         behav_cma = behavior_vector_batch[idx]
    #     else:
    #         morph_cma = morphology_vector_batch
    #         behav_cma = behavior_vector_batch

    #     # encode
    #     z_m = self.actor_critic.encode_morph_for_cma(morph_cma, detach=False)
    #     z_b = self.actor_critic.encode_behavior_for_cma(behav_cma, detach=False)

    #     z_m = F.normalize(z_m, dim=-1)
    #     z_b = F.normalize(z_b, dim=-1)

    #     # alignment
    #     cos_sim = (z_m * z_b).sum(dim=-1)
    #     cma_loss = (1.0 - cos_sim).mean()

    #     # ---- subsample again for uniformity ----
    #     with torch.no_grad():
    #         z_for_uni = z_m
    #         max_uni = 128  # much smaller
    #         if z_for_uni.shape[0] > max_uni:
    #             idx_u = torch.randperm(z_for_uni.shape[0], device=z_for_uni.device)[:max_uni]
    #             z_for_uni = z_for_uni[idx_u]

    #         sim_mm = z_for_uni @ z_for_uni.t()
    #         sim_mm.fill_diagonal_(0.0)
    #         spread = (sim_mm ** 2).mean()

    #     return cma_loss + 0.02 * spread


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, morphology_vector_batch,  behavior_vector_batch,  actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in (generator): ##morph

            # self.actor_critic.act(obs_history_batch, morphology_vector_batch, masks=masks_batch) ##morph
            # actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch,morphology_vector_batch, masks=masks_batch) ##morph
            self.actor_critic.act(
                obs_history_batch,
                privileged_obs_batch,
                morphology_vector_batch,
                masks=masks_batch,
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

            value_batch = self.actor_critic.evaluate(
                obs_history_batch,
                privileged_obs_batch,
                morphology_vector_batch,
                masks=masks_batch,
            )

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPO_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = surrogate_loss + PPO_Args.value_loss_coef * value_loss - PPO_Args.entropy_coef * entropy_batch.mean()



            # cma_loss = 0.0

            # only do CMA sometimes, e.g.:
            # --- CMA alignment (morph ↔ behavior) ---

            z_m = self.actor_critic.encode_morph_for_cma(
                morphology_vector_batch,
                detach=False,        # we want grads here
            )
            z_b = self.actor_critic.encode_behavior_for_cma(
                behavior_vector_batch,
                detach=False,
            )

            z_m = F.normalize(z_m, dim=-1)
            z_b = F.normalize(z_b, dim=-1)

            # alignment: we DO want gradients
            cos_sim = (z_m * z_b).sum(dim=-1)      # [N]
            cma_loss = (1.0 - cos_sim).mean()

            # --- tiny uniformity penalty (NO gradients, to save memory) ---
            # with torch.no_grad():
                # (optional but smart) subsample to avoid 6000x6000 when batch is big
            with torch.no_grad():
                z_for_uni = z_m
                max_uni = 1024
                if z_for_uni.shape[0] > max_uni:
                    idx = torch.randperm(z_for_uni.shape[0], device=z_for_uni.device)[:max_uni]
                    z_for_uni = z_for_uni[idx]
                sim_mm = z_for_uni @ z_for_uni.t()
                sim_mm.fill_diagonal_(0.0)
                spread = (sim_mm ** 2).mean()
              


            # final loss
            # loss = ppo_loss + PPO_Args.cma_coef * cma_loss


            loss = ppo_loss + PPO_Args.cma_coef * cma_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            

            ###cma loss end###

            # # Gradient step
            # self.optimizer.zero_grad()
            # loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            # self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step

            for epoch in range(PPO_Args.num_adaptation_module_substeps):

                adaptation_pred = self.actor_critic.adaptation_module(obs_history_batch)
                with torch.no_grad():
                    adaptation_target = privileged_obs_batch
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
                if PPO_Args.selective_adaptation_module_loss:
                    # mask out indices corresponding to swing feet
                    selection_indices = 0

                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])



                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward()
                self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()

        # self.global_cma_step += 1

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
