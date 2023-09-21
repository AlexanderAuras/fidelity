import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
import torch.backends.cudnn
import yaml


class _Reference:
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path.split(".")


@lru_cache
def load_config(path: Union[str, Path], _toplevel: bool = True) -> Any:
    if isinstance(path, str):
        path = Path(path)

    class ConfigLoader(yaml.FullLoader):
        pass

    ConfigLoader.add_constructor("!complex", lambda l, n: complex(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n))).replace("i", "j")))
    ConfigLoader.add_implicit_resolver("!complex", re.compile(r"^$"), None)
    ConfigLoader.add_constructor("!path", lambda l, n: Path(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))
    ConfigLoader.add_implicit_resolver("!path", re.compile(r"^file://"), None)
    ConfigLoader.add_constructor("!include", lambda l, n: load_config(Path(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))).resolve(), False) if Path(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))).is_absolute() else load_config(path.parent.joinpath(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))).resolve(), False))
    ConfigLoader.add_constructor("!eval", lambda l, n: eval(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))  # pylint: disable=W0123
    ConfigLoader.add_constructor("!ref", lambda l, n: _Reference(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))
    # BUG pyyaml does not recognize all float literals correctly
    ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile("""^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\\.[0-9_]+(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$""", re.X), list("-+0123456789."))

    def resolve_references(element: Any, _full_config: Optional[Dict[str, Any]] = None) -> Any:
        if _full_config is None:
            _full_config = element
        if isinstance(element, _Reference):
            value = _full_config
            for path_part in element.path:
                value = cast(Dict[str, Any], value)[path_part]
            return value
        elif isinstance(element, Dict):
            return {k: resolve_references(v, _full_config) for k, v in element.items()}
        elif isinstance(element, List):
            return [resolve_references(x, _full_config) for x in element]
        return element

    with open(path, "r", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, ConfigLoader)
        if not isinstance(config, Dict):
            raise RuntimeError("Configuration root element must be a dictionary/mapping")
        if _toplevel:
            config = resolve_references(config)
        return config


def setup_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng_states() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def load_rng_states(states: Dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
