"""Mini Cheetah configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    0.634,
    0.064,
    0.54,
    0.06,
    3.3,
    -1.6,
    1.6,
    -2.6,
    2.6,
    -2.6,
    -0.8,
)


def config_mini_cheetah(cfg):
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

    cfg.control.stiffness = {"joint": 20.0}
    cfg.control.damping = {"joint": 0.5}

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/"
        "mini_cheetah/urdf/cheetah.urdf"
    )
    cfg.asset.foot_name = [
        "FL_foot", "FR_foot", "RL_foot", "RR_foot"
    ]
    cfg.asset.penalize_contacts_on = ["thigh"]
    cfg.asset.terminate_after_contacts_on = ["trunk"]
    cfg.asset.flip_visual_attachments = False

    cfg.rewards.base_height_target = 0.30
    cfg.reward_scales.torques = -0.0002

    cfg.domain_rand.randomize_base_mass = True
    cfg.domain_rand.added_mass_range = [0, 0]
    cfg.domain_rand.friction_range = [0.05, 4.5]
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]

    cfg.Morphology.robot_name = "mini_cheetah"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)