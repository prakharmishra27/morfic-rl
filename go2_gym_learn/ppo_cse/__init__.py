import time
from collections import deque
import copy
import os

import torch
from ml_logger import logger
from params_proto import PrefixProto

from .actor_critic import ActorCritic, AC_Args
from .rollout_storage import RolloutStorage



def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from .metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 5  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env
        
        ### model construction
        self.model_config = {
            "num_obs": self.env.num_obs,
            "num_privileged_obs": self.env.num_privileged_obs,
            "num_obs_history": self.env.num_obs_history,
            "num_actions": self.env.num_actions,
            "morphology_dim": self.env.morphology_vector.shape[-1],
            "morphology_latent_dim": 64,
            "variant": AC_Args.variant,
        }

        # Snapshot settings now, rather than reading mutable globals at save time.
        self.architecture_config = {
            "actor_hidden_dims": list(AC_Args.actor_hidden_dims),
            "critic_hidden_dims": list(AC_Args.critic_hidden_dims),
            "adaptation_module_branch_hidden_dims": list(
                AC_Args.adaptation_module_branch_hidden_dims
            ),
            "activation": AC_Args.activation,
            "init_noise_std": AC_Args.init_noise_std,
            "student_latent_dropout": AC_Args.student_latent_dropout,
            "use_decoder": AC_Args.use_decoder,
            "film_scale": AC_Args.film_scale,
            "hypernet_rank": AC_Args.hypernet_rank,
            "hypernet_scale": AC_Args.hypernet_scale,
        }

        actor_critic = ActorCritic(
            **self.model_config
        ).to(self.device)

        if RunnerArgs.resume:
            # load pretrained weights from resume_path
            from ml_logger import ML_Logger
            loader = ML_Logger(root="http://escher.csail.mit.edu:8080",
                               prefix=RunnerArgs.resume_path)
            weights = loader.load_torch("checkpoints/ac_weights_last.pt")
            actor_critic.load_state_dict(state_dict=weights)

            if hasattr(self.env, "curricula") and RunnerArgs.resume_curriculum:
                # load curriculum state
                distributions = loader.load_pkl("curriculum/distribution.pkl")
                distribution_last = distributions[-1]["distribution"]
                gait_names = [key[8:] if key.startswith("weights_") else None for key in distribution_last.keys()]
                for gait_id, gait_name in enumerate(self.env.category_names):
                    self.env.curricula[gait_id].weights = distribution_last[f"weights_{gait_name}"]
                    print(gait_name)

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions],self.env.morphology_vector[:self.env.num_train_envs])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = 0

        self.env.reset()

    def save_checkpoint(self, iteration):
        """Save weights, reconstruction metadata, and inference components."""
        ac = self.alg.actor_critic

        metadata = {
            "iteration": int(iteration),
            "model_config": dict(self.model_config),
            "architecture_config": dict(self.architecture_config),
            "robot_name": self.env.cfg.Morphology.robot_name,
            "morphology_mode": self.env.morphology_mode,
            "morphology_min": self.env.morph_min.detach().cpu().tolist(),
            "morphology_max": self.env.morph_max.detach().cpu().tolist(),
            "actor_conditioning": ac.actor_conditioning,
            "critic_conditioning": ac.critic_conditioning,
        }

        with logger.Sync():
            weights_path = (
                f"checkpoints/ac_weights_{iteration:06d}.pt"
            )
            logger.torch_save(ac.state_dict(), weights_path)
            logger.duplicate(
                weights_path, "checkpoints/ac_weights_last.pt"
            )

            metadata_path = (
                f"checkpoints/model_config_{iteration:06d}.pkl"
            )
            logger.save_pkl(metadata, metadata_path)
            logger.duplicate(
                metadata_path, "checkpoints/model_config_last.pkl"
            )

            # A separate temporary directory avoids collisions between runs.
            import tempfile

            with tempfile.TemporaryDirectory(
                prefix="morfic_export_"
            ) as directory:
                components = {
                    "adaptation_module_latest.jit": ac.adaptation_module,
                    "body_latest.jit": ac.actor_body,
                }

                if ac.morph_encoder is not None:
                    components["morph_encoder_latest.jit"] = (
                        ac.morph_encoder
                    )

                for filename, module in components.items():
                    export_module = copy.deepcopy(module).cpu().eval()
                    scripted = torch.jit.script(export_module)
                    local_path = os.path.join(directory, filename)
                    scripted.save(local_path)
                    logger.upload_file(
                        file_path=local_path,
                        target_path="checkpoints/",
                    )

    def learn(
        self,
        num_learning_iterations,
        init_at_random_ep_len=False,
        eval_freq=100,
        curriculum_dump_freq=500,
        eval_expert=False,
    ):
        from ml_logger import logger

        assert logger.prefix, "Set a run-specific logger prefix first."

        if num_learning_iterations < 0:
            raise ValueError("num_learning_iterations must be nonnegative.")
        if num_learning_iterations == 0:
            return
        if self.num_steps_per_env <= 0:
            raise ValueError("num_steps_per_env must be positive.")

        logger.start("start", "epoch", "episode", "run", "step")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        num_train_envs = self.env.num_train_envs
        num_envs = self.env.num_envs
        num_eval_envs = num_envs - num_train_envs

        if not 0 < num_train_envs <= num_envs:
            raise ValueError("Invalid training/evaluation environment split.")

        print(
            f"[runner] train={num_train_envs}, eval={num_eval_envs}, "
            f"variant={self.alg.actor_critic.variant}"
        )

        def unpack_observations(obs_dict):
            obs = obs_dict["obs"].to(self.device)
            privileged_obs = obs_dict["privileged_obs"].to(self.device)

            history = obs_dict.get("obs_history")
            if history is None:
                history = self.env.obs_history
            history = history.to(self.device)

            morph = obs_dict.get("morphology_vector")
            if morph is None or morph.numel() == 0:
                morph = self.env.morphology_vector

            if (
                morph is None
                or morph.ndim != 2
                or morph.shape[0] != num_envs
                or morph.shape[1] != self.alg.actor_critic.morphology_dim
            ):
                raise RuntimeError(
                    "Morphology must have shape "
                    f"({num_envs}, "
                    f"{self.alg.actor_critic.morphology_dim})."
                )

            return obs, privileged_obs, history, morph.to(self.device)

        obs, privileged_obs, obs_history, morphology_vector = (
            unpack_observations(self.env.get_observations())
        )

        ac = self.alg.actor_critic
        ac.train()

        first_iteration = self.current_learning_iteration
        final_iteration = first_iteration + num_learning_iterations
        last_saved_iteration = None

        for it in range(first_iteration, final_iteration):
            collection_start = time.time()
            infos = {}

            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    actions_train = self.alg.act(
                        obs[:num_train_envs],
                        privileged_obs[:num_train_envs],
                        obs_history[:num_train_envs],
                        morphology_vector[:num_train_envs],
                    )

                    actions = actions_train

                    # Do not call inference methods with an empty eval batch.
                    if num_eval_envs > 0:
                        morph_eval = morphology_vector[num_train_envs:]

                        # Disable student dropout during evaluation only.
                        was_training = ac.training
                        ac.eval()
                        try:
                            if eval_expert:
                                actions_eval = ac.act_teacher(
                                    obs_history[num_train_envs:],
                                    privileged_obs[num_train_envs:],
                                    morph_eval,
                                )
                            else:
                                actions_eval = ac.act_student(
                                    obs[num_train_envs:],
                                    obs_history[num_train_envs:],
                                    morph_eval,
                                )
                        finally:
                            ac.train(was_training)

                        actions = torch.cat(
                            (actions_train, actions_eval), dim=0
                        )

                    obs_dict, rewards, dones, infos = self.env.step(actions)

                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(
                        rewards[:num_train_envs],
                        dones[:num_train_envs],
                        infos,
                    )

                    obs, privileged_obs, obs_history, morphology_vector = (
                        unpack_observations(obs_dict)
                    )

                    for group in ("train/episode", "eval/episode"):
                        if group in infos:
                            with logger.Prefix(metrics=group):
                                logger.store_metrics(**infos[group])

                collection_time = time.time() - collection_start

                self.alg.compute_returns(
                    obs_history[:num_train_envs],
                    privileged_obs[:num_train_envs],
                    morphology_vector[:num_train_envs],
                )

                if (
                    curriculum_dump_freq
                    and it % curriculum_dump_freq == 0
                ):
                    logger.save_pkl(
                        {
                            "iteration": it,
                            **caches.slot_cache.get_summary(),
                            **caches.dist_cache.get_summary(),
                        },
                        path="curriculum/info.pkl",
                        append=True,
                    )

                    if "curriculum/distribution" in infos:
                        logger.save_pkl(
                            {
                                "iteration": it,
                                "distribution": infos[
                                    "curriculum/distribution"
                                ],
                            },
                            path="curriculum/distribution.pkl",
                            append=True,
                        )

            update_start = time.time()

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_adaptation_module_loss,
                mean_decoder_loss,
                mean_decoder_loss_student,
                mean_adaptation_module_test_loss,
                mean_decoder_test_loss,
                mean_decoder_test_loss_student,
            ) = self.alg.update()

            learn_time = time.time() - update_start

            logger.store_metrics(
                time_elapsed=logger.since("start"),
                time_iter=logger.split("epoch"),
                collection_time=collection_time,
                learn_time=learn_time,
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss,
                mean_decoder_loss=mean_decoder_loss,
                mean_decoder_loss_student=mean_decoder_loss_student,
                mean_decoder_test_loss=mean_decoder_test_loss,
                mean_decoder_test_loss_student=(
                    mean_decoder_test_loss_student
                ),
                mean_adaptation_module_test_loss=(
                    mean_adaptation_module_test_loss
                ),
            )

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            # Count transitions used for PPO, excluding evaluation rollouts.
            self.tot_timesteps += (
                self.num_steps_per_env * num_train_envs
            )
            self.tot_time += collection_time + learn_time
            self.current_learning_iteration = it + 1

            if (
                RunnerArgs.log_freq
                and logger.every(
                    RunnerArgs.log_freq, "iteration", start_on=1
                )
            ):
                logger.log_metrics_summary(
                    key_values={
                        "timesteps": self.tot_timesteps,
                        "iterations": it,
                    }
                )
                logger.job_running()

            if (
                RunnerArgs.save_interval
                and it % RunnerArgs.save_interval == 0
            ):
                self.save_checkpoint(it)
                last_saved_iteration = it

        last_iteration = self.current_learning_iteration - 1
        if last_saved_iteration != last_iteration:
            self.save_checkpoint(last_iteration)

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            if self.env.num_eval_envs > 0:
                self.env.start_recording_eval()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)

        if self.env.num_eval_envs > 0:
            frames = self.env.get_complete_frames_eval()
            if len(frames) > 0:
                self.env.pause_recording_eval()
                print("LOGGING EVAL VIDEO")
                logger.save_video(frames, f"videos/{it:05d}_eval.mp4", fps=1 / self.env.dt)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
