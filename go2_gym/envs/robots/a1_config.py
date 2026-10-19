"""Unitree A1 configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    1.013,
    0.166,
    0.696,
    0.06,
    6.0,
    -0.802851,
    0.802851,
    -1.047198,
    4.188790,
    -2.696534,
    -0.916298,
)


def config_a1(cfg):
    config_shared(cfg)

    cfg.init_state.pos = [0.0, 0.0, 0.32]
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

    cfg.control.stiffness = {"joint": 30.0}
    cfg.control.damping = {"joint": 0.7}
    cfg.control.hip_scale_reduction = 1.0

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/"
        "a1_description/urdf/a1.urdf"
    )
    cfg.asset.foot_name = [
        "FL_foot", "FR_foot", "RL_foot", "RR_foot"
    ]
    cfg.asset.penalize_contacts_on = ["thigh", "calf"]
    cfg.asset.terminate_after_contacts_on = ["base"]
    cfg.asset.flip_visual_attachments = True

    # From the full A1 block; historical experiment value unverified.
    cfg.rewards.base_height_target = 0.30
    cfg.reward_scales.orientation = -3.0
    cfg.reward_scales.base_height = -15.0

    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.friction_range = [0.05, 4.5]
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]

    cfg.Morphology.robot_name = "a1"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)