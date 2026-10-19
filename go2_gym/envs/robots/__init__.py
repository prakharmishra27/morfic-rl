"""Robot configuration registry."""

from importlib import import_module

ROBOT_CONFIGS = {
    "go2": (".go2_config", "config_go2"),
    "go1": (".go1_config", "config_go1"),
    "a1": (".a1_config", "config_a1"),
    "aliengo": (".aliengo_config", "config_aliengo"),
    "b1": (".b1_config", "config_b1"),
    "mini_cheetah": (".mini_cheetah_config", "config_mini_cheetah"),
    "anymal_c": (".anymal_c_config", "config_anymal_c"),
}


def configure_robot(cfg, robot_name):
    """Apply the selected robot configuration to cfg."""
    name = robot_name.strip().lower()

    if name not in ROBOT_CONFIGS:
        choices = ", ".join(ROBOT_CONFIGS)
        raise ValueError(
            f"Unknown robot {robot_name!r}. Choose from: {choices}"
        )

    module_name, function_name = ROBOT_CONFIGS[name]
    module = import_module(module_name, package=__name__)
    configure = getattr(module, function_name)
    configure(cfg)

    return cfg