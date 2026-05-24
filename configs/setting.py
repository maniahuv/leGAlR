import yaml
from types import SimpleNamespace
from pathlib import Path


def _dict_to_namespace(d: dict):
    return SimpleNamespace(
        **{
            k: _dict_to_namespace(v) if isinstance(v, dict) else v
            for k, v in d.items()
        }
    )


def load_config(path: str | None = None):
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    else:
        path = Path(path)

    print("Loading config from:", path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _dict_to_namespace(data)


config = load_config()