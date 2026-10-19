from isaacgym import gymutil, gymapi
import torch
from go2_gym.envs.robots.morphology import (
    normalize_morphology,
    sample_morphologies,
)

from go2_gym.envs.base.legged_robot import LeggedRobot
from go2_gym.envs.base.legged_robot_config import Cfg



class VelocityTrackingEasyEnv(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX", morphology_mode=None,):
        
        if morphology_mode not in ("train", "eval"):
            raise ValueError(
                "Set morphology_mode='train' or morphology_mode='eval'."
            )
        if deploy and morphology_mode == "train":
            raise ValueError("Deployment requires morphology_mode='eval'.")

        if num_envs is not None:
            cfg.env.num_envs = num_envs
        if prone:
            cfg.init_state.rot = [0.0, 1.0, 0.0, 0.0]
            cfg.init_state.pos = [0.0, 0.0, 0.15]
            cfg.asset.fix_base_link = True
        if deploy:  # turn off randomization, fix terrain
            terrain_level = 1
            cfg.terrain.num_rows = 10
            cfg.terrain.num_cols = 1
            cfg.terrain.curriculum = True
            cfg.terrain.min_init_terrain_level = terrain_level
            cfg.terrain.max_init_terrain_level = terrain_level
            cfg.noise.add_noise = False
            cfg.domain_rand.push_robots = False
            cfg.domain_rand.randomize_friction = False
            cfg.env.episode_length_s = 100
            cfg.commands.ranges.lin_vel_x = [0, 0]
            cfg.commands.ranges.lin_vel_y = [0, 0]
            cfg.commands.ranges.ang_vel_yaw = [0, 0]
            cfg.commands.ranges.heading = [0, 0]
            cfg.commands.heading_command = False
            # cfg.Morphology.morphology_vector = morphology_vector
        
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)

        self.morphology_mode = morphology_mode

        # Always normalize with the source training bounds.
        self.morph_min = cfg.Morphology.morphology_min.to(
            device=self.device, dtype=torch.float32
        )
        self.morph_max = cfg.Morphology.morphology_max.to(
            device=self.device, dtype=torch.float32
        )

        all_env_ids = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        )

        if morphology_mode == "train":
            # Sample once per environment and retain across episode resets.
            self.morphology_vector = sample_morphologies(
                self.num_envs, self.morph_min, self.morph_max
            )

            # Apply masses and joint limits to the source robot template.
            self.apply_morphology(all_env_ids)

        else:
            # The selected target asset supplies the physical morphology.
            descriptor = cfg.Morphology.nominal_descriptor
            normalized = normalize_morphology(
                descriptor, self.morph_min, self.morph_max
            )
            if normalized.shape != (11,):
                raise ValueError(
                    "nominal_descriptor must contain exactly 11 values."
                )
            self.morphology_vector = normalized.unsqueeze(0).repeat(
                self.num_envs, 1
            )

        # Preserve the existing contact-property refresh.
        self._call_train_eval(
            self.refresh_actor_rigid_shape_props, all_env_ids
        )

        print(
            "[VTEnv] robot:", cfg.Morphology.robot_name,
            "| morphology mode:", morphology_mode,
            "| shape:", tuple(self.morphology_vector.shape),
        )


   
    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                               0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })
        self.extras["morphology_vector"] = self.morphology_vector


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs
    
    def get_observations(self):  ### changes for morph
        return {
            "obs": self.obs_buf,
            "privileged_obs": self.privileged_obs_buf,
            # "obs_history": self.obs_history,
            "morphology_vector": self.morphology_vector
        }
    


