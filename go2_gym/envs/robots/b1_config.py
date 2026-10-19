"""Unitree B1 configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    4.3,
    1.05,
    2.15,
    0.05,
    29.45,
    -0.75,
    0.75,
    -1.0,
    3.5,
    -2.6,
    -0.6,
)


def config_b1(cfg):
    config_shared(cfg)

    cfg.init_state.pos = [0.0, 0.0, 0.60]
    cfg.init_state.default_joint_angles = {
        "FL_hip_joint": 0.1,
        "RL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "RL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RR_thigh_joint": 0.8,
        "FL_calf_joint": -1.6,
        "RL_calf_joint": -1.6,
        "FR_calf_joint": -1.6,
        "RR_calf_joint": -1.6,
    }

    cfg.control.stiffness = {"joint": 90.0}
    cfg.control.damping = {"joint": 2.5}
    cfg.control.action_scale = 0.35

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/b1_description/urdf/"
        "b1.urdf"
    )
    cfg.asset.foot_name = [
        "FL_foot", "FR_foot", "RL_foot", "RR_foot"
    ]
    cfg.asset.penalize_contacts_on = ["thigh", "calf"]
    cfg.asset.terminate_after_contacts_on = ["trunk"]
    cfg.asset.flip_visual_attachments = True

    cfg.rewards.base_height_target = 0.56
    cfg.reward_scales.orientation = -3.0
    cfg.reward_scales.base_height = -15.0

    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.friction_range = [0.05, 4.5]
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]

    cfg.Morphology.robot_name = "b1"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)