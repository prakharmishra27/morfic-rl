"""Unitree Go2 configuration and nominal morphology descriptor."""

from .config_shared import config_shared


# Order: thigh, calf, hip, foot, base masses;
# then hip, thigh, calf lower/upper joint limits.
MORPHOLOGY_DESCRIPTOR = (
    1.152,
    0.154,
    0.678,
    0.04,
    6.921,
    -1.0472,
    1.0472,
    -1.5708,
    3.4907,
    -2.7227,
    -0.83776,
)


def config_go2(cfg):
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

    cfg.control.stiffness = {"joint": 25.0}
    cfg.control.damping = {"joint": 0.6}

    cfg.asset.file = (
        "{MINI_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
    )
    cfg.asset.foot_name = "foot"
    cfg.asset.penalize_contacts_on = ["thigh", "calf"]
    cfg.asset.terminate_after_contacts_on = ["base"]
    cfg.asset.flip_visual_attachments = True

    cfg.rewards.base_height_target = 0.34

    cfg.Morphology.robot_name = "go2"
    cfg.Morphology.nominal_descriptor = list(MORPHOLOGY_DESCRIPTOR)