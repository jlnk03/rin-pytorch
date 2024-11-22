import torch
import torch_optimizer

from .lamb import Lamb


def get_optimizer(name: str, params, lr: float, **kwargs) -> torch.optim.Optimizer:
    name = name.lower()

    if name == "lamb":
        return Lamb(params=params, lr=lr, **kwargs)

    optimizer_cls = getattr(torch.optim, name, None)
    if optimizer_cls is None:
        optimizer_cls = torch_optimizer.get(name)

    return optimizer_cls(params=params, lr=lr, **kwargs)


def build_parameters_mapping(model: torch.nn.Module) -> dict[int, str]:
    """Build a mapping of parameter IDs to their names in the model."""
    mapping = {}
    
    for name, param in model.named_parameters():
        mapping[id(param)] = name
        
    return mapping


def override_config_for_names(
    parameters,
    names: list[str],
    config_override: dict,
    name_mapping: dict[int, str],
) -> list[dict]:
    """Create parameter groups with different configs based on parameter names."""
    group_default = {"params": []}
    group_override = {"params": [], **config_override}

    for param in parameters:
        if not param.requires_grad:
            continue

        if id(param) in name_mapping and any(name in name_mapping[id(param)] for name in names):
            group_override["params"].append(param)
        else:
            group_default["params"].append(param)

    if len(group_default["params"]) == 0:
        return [group_override]
    elif len(group_override["params"]) == 0:
        return [group_default]
    else:
        return [group_default, group_override]
