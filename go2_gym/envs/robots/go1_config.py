"""Unitree Go1 configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    0.898919,
    0.158015,
    0.510299,
    0.06,
    4.8,
    -0.8028514559,
    0.8028514559,
    -1.0471975512,
    4.1887902048,
    -2.6965336943,
    -0.9162978573,
)


def config_go1(cfg):
    config_shared(cfg)

    cfg.init_state.pos = [0.0, 0.0, 0.34]
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

    cfg.control.stiffness = {"joint": 20.0}
    cfg.control.damping = {"joint": 0.5}

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf"
    )
    cfg.asset.foot_name = "foot"
    cfg.asset.penalize_contacts_on = ["thigh", "calf"]
    cfg.asset.terminate_after_contacts_on = ["base"]
    cfg.asset.flip_visual_attachments = False

    cfg.rewards.base_height_target = 0.34

    # Overrides preserved from the original Go1 config block.
    cfg.domain_rand.randomize_base_mass = True
    cfg.domain_rand.randomize_com_displacement = True
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]

    cfg.Morphology.robot_name = "go1"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)