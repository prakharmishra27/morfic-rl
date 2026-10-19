"""AlienGo configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    0.639,
    0.207,
    1.0,
    0.06,
    11.644,
    -1.2217,
    1.2217,
    -3.1415,
    3.1415,
    -2.775,
    -0.6457,
)


def config_aliengo(cfg):
    config_shared(cfg)

    cfg.init_state.pos = [0.0, 0.0, 0.36]
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

    cfg.control.stiffness = {"joint": 40.0}
    cfg.control.damping = {"joint": 1.0}
    cfg.control.action_scale = 0.30

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/"
        "aliengo_description/urdf/aliengo.urdf"
    )
    cfg.asset.foot_name = [
        "FL_foot", "FR_foot", "RL_foot", "RR_foot"
    ]
    cfg.asset.penalize_contacts_on = ["thigh", "calf"]
    cfg.asset.terminate_after_contacts_on = ["trunk"]
    cfg.asset.flip_visual_attachments = True

    cfg.rewards.base_height_target = 0.36
    cfg.reward_scales.base_height = -10.0

    cfg.terrain.teleport_robots = False

    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.friction_range = [0.05, 4.5]
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]

    cfg.Morphology.robot_name = "aliengo"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)