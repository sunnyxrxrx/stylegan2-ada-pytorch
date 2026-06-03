import copy
from typing import Any

import torch
import dnnlib

import legacy
from torch_utils import misc


_STATE_DICT_FORMAT_VERSION = 1


def _module_class_name(module: torch.nn.Module) -> str:
    module_type = type(module)
    if module_type.__module__ == "torch_utils.persistence" and len(module_type.__mro__) > 1:
        orig_type = module_type.__mro__[1]
        return f"{orig_type.__module__}.{orig_type.__name__}"
    return f"{module_type.__module__}.{module_type.__name__}"


def _normalize_class_name(class_name: str) -> str:
    if class_name.startswith("torch_utils.persistence."):
        short_name = class_name.rsplit(".", 1)[-1]
        mapping = {
            "Generator": "training.networks.Generator",
            "Discriminator": "training.networks.Discriminator",
            "AugmentPipe": "training.augment.AugmentPipe",
        }
        return mapping.get(short_name, class_name)
    return class_name


def _serialize_module(module: torch.nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    return {
        "class_name": _module_class_name(module),
        "init_kwargs": copy.deepcopy(dict(module.init_kwargs)),
        "state_dict": {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()},
    }


def _deserialize_module(spec: dict[str, Any] | None, force_fp16: bool = False) -> torch.nn.Module | None:
    if spec is None:
        return None

    class_name = _normalize_class_name(spec["class_name"])
    module = dnnlib.util.construct_class_by_name(class_name=class_name, **spec["init_kwargs"]).eval().requires_grad_(False)
    module.load_state_dict(spec["state_dict"])

    if force_fp16:
        kwargs = copy.deepcopy(module.init_kwargs)
        if class_name.endswith(".Generator"):
            kwargs = dnnlib.EasyDict(kwargs)
            kwargs.synthesis_kwargs = dnnlib.EasyDict(kwargs.get("synthesis_kwargs", {}))
            kwargs.synthesis_kwargs.num_fp16_res = 4
            kwargs.synthesis_kwargs.conv_clamp = 256
        if class_name.endswith(".Discriminator"):
            kwargs = dnnlib.EasyDict(kwargs)
            kwargs.num_fp16_res = 4
            kwargs.conv_clamp = 256
        if kwargs != module.init_kwargs:
            fp16_module = dnnlib.util.construct_class_by_name(class_name=class_name, **kwargs).eval().requires_grad_(False)
            misc.copy_params_and_buffers(module, fp16_module, require_all=True)
            module = fp16_module

    return module


def is_state_dict_checkpoint(data: Any) -> bool:
    return isinstance(data, dict) and data.get("format") == "stylegan2-ada-pytorch-state-dict"


def save_network_checkpoint(path: str, snapshot_data: dict[str, Any]) -> None:
    payload = {
        "format": "stylegan2-ada-pytorch-state-dict",
        "format_version": _STATE_DICT_FORMAT_VERSION,
        "training_set_kwargs": copy.deepcopy(snapshot_data.get("training_set_kwargs")),
        "modules": {
            "G": _serialize_module(snapshot_data.get("G")),
            "D": _serialize_module(snapshot_data.get("D")),
            "G_ema": _serialize_module(snapshot_data.get("G_ema")),
            "augment_pipe": _serialize_module(snapshot_data.get("augment_pipe")),
        },
    }
    torch.save(payload, path)


def load_network_checkpoint(path_or_url: str, force_fp16: bool = False) -> dict[str, Any]:
    with dnnlib.util.open_url(path_or_url) as f:
        if str(path_or_url).lower().endswith(".pkl"):
            return legacy.load_network_pkl(f, force_fp16=force_fp16)
        data = torch.load(f, map_location="cpu")

    if not is_state_dict_checkpoint(data):
        raise ValueError(f'Unsupported checkpoint format: "{path_or_url}"')

    modules = data["modules"]
    return {
        "G": _deserialize_module(modules.get("G"), force_fp16=force_fp16),
        "D": _deserialize_module(modules.get("D"), force_fp16=force_fp16),
        "G_ema": _deserialize_module(modules.get("G_ema"), force_fp16=force_fp16),
        "augment_pipe": _deserialize_module(modules.get("augment_pipe"), force_fp16=False),
        "training_set_kwargs": copy.deepcopy(data.get("training_set_kwargs")),
    }
