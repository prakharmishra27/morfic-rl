"""ANYmal-C configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    0.74,
    0.33742,
    0.74,
    0.25,
    18.21,
    -0.72,
    0.72,
    -1.65,
    4.05,
    -2.72,
    -0.78,
)


def config_anymal_c(cfg):
    config_shared(cfg)

    cfg.init_state.pos = [0.0, 0.0, 0.55]
    cfg.init_state.default_joint_angles = {
        "FL_hip_joint": 0.1,
        "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "FR_thigh_joint": 0.8,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }

    cfg.control.stiffness = {"joint": 80.0}
    cfg.control.damping = {"joint": 2.0}

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/"
        "anymal_c_simple_description/urdf/anymal_c.urdf"
    )
    cfg.asset.foot_name = [
        "FL_foot", "FR_foot", "RL_foot", "RR_foot"
    ]
    cfg.asset.penalize_contacts_on = []
    cfg.asset.terminate_after_contacts_on = ["trunk"]
    cfg.asset.collapse_fixed_joints = True
    cfg.asset.flip_visual_attachments = True

    cfg.rewards.base_height_target = 0.55

    # Additional reward settings from the original ANYmal-C block.
    cfg.reward_scales.dof_vel = -0.0002
    cfg.reward_scales.feet_air_time = 1.0
    cfg.reward_scales.collision = -1.0
    cfg.reward_scales.stumble = 0.0
    cfg.reward_scales.stand_still = 0.0

    cfg.env.num_envs = 2000
    cfg.env.episode_length_s = 20

    cfg.domain_rand.push_robots = True
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.friction_range = [0.5, 1.25]
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]

    cfg.Morphology.robot_name = "anymal_c"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)