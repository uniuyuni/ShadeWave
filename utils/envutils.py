import os


TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag(name, default=False, environ=None):
    source = os.environ if environ is None else environ
    value = source.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in TRUE_ENV_VALUES
