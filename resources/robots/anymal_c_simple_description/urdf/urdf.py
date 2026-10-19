import os

# your docker-ish template path (copy/paste exactly)
ANYMAL_URDF_TEMPLATE = "{MINI_GYM_ROOT_DIR}/resources/robots/anymal_c_simple_description/urdf/anymal.urdf"

def resolve_template_path(path_template: str) -> str:
    """
    Resolves {VARNAME} placeholders using environment variables.
    Example: "{MINI_GYM_ROOT_DIR}/x/y" -> "/abs/root/x/y"
    """
    import re
    def repl(m):
        var = m.group(1)
        val = os.environ.get(var)
        if not val:
            raise RuntimeError(f"Environment variable {var} is not set, can't resolve: {path_template}")
        return val
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, path_template)

# Use resolved path
input_urdf = resolve_template_path(ANYMAL_URDF_TEMPLATE)

# output next to input
output_urdf = os.path.join(os.path.dirname(input_urdf), "anymal_clean.urdf")

print("Input:", input_urdf)
print("Output:", output_urdf)
