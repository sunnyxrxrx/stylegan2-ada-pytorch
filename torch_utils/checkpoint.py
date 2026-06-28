import copy
import io
import os
import pickle
from typing import Any

import torch
import dnnlib

import legacy
from torch_utils import misc


_STATE_DICT_FORMAT_VERSION = 1
_TRAINING_STATE_FORMAT_VERSION = 1
_NETWORK_CHECKPOINT_FORMAT = "stylegan2-ada-pytorch-state-dict"
_TRAINING_STATE_FORMAT = "stylegan2-ada-pytorch-training-state"


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


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _resolve_training_resume_path(path_or_url: str) -> str:
    if not isinstance(path_or_url, str):
        return path_or_url
    if os.path.isdir(path_or_url):
        latest_path = os.path.join(path_or_url, 'latest.pt')
        if os.path.isfile(latest_path):
            return latest_path
        legacy_latest_path = os.path.join(path_or_url, 'training-state-latest.pt')
        if os.path.isfile(legacy_latest_path):
            return legacy_latest_path
        return latest_path
    if str(path_or_url).lower().endswith('.pkl') or not os.path.isfile(path_or_url):
        return path_or_url
    return path_or_url


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
    return isinstance(data, dict) and data.get("format") == _NETWORK_CHECKPOINT_FORMAT


def is_training_state_checkpoint(data: Any) -> bool:
    return isinstance(data, dict) and data.get("format") == _TRAINING_STATE_FORMAT


def _deserialize_network_payload(data: dict[str, Any], force_fp16: bool = False) -> dict[str, Any]:
    modules = data["modules"]
    return {
        "G": _deserialize_module(modules.get("G"), force_fp16=force_fp16),
        "D": _deserialize_module(modules.get("D"), force_fp16=force_fp16),
        "G_ema": _deserialize_module(modules.get("G_ema"), force_fp16=force_fp16),
        "augment_pipe": _deserialize_module(modules.get("augment_pipe"), force_fp16=False),
        "training_set_kwargs": copy.deepcopy(data.get("training_set_kwargs")),
    }


def save_network_checkpoint(path: str, snapshot_data: dict[str, Any]) -> None:
    payload = {
        "format": _NETWORK_CHECKPOINT_FORMAT,
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


def save_training_checkpoint(path: str, training_state: dict[str, Any]) -> None:
    payload = {
        "format": _TRAINING_STATE_FORMAT,
        "format_version": _TRAINING_STATE_FORMAT_VERSION,
        "training_set_kwargs": copy.deepcopy(training_state.get("training_set_kwargs")),
        "modules": {
            "G": _serialize_module(training_state.get("G")),
            "D": _serialize_module(training_state.get("D")),
            "G_ema": _serialize_module(training_state.get("G_ema")),
            "augment_pipe": _serialize_module(training_state.get("augment_pipe")),
        },
        "optimizers": _to_cpu(training_state.get("optimizers", {})),
        "progress": _to_cpu(training_state.get("progress", {})),
        "sampler_states": _to_cpu(training_state.get("sampler_states", [])),
        "rng_states": _to_cpu(training_state.get("rng_states", [])),
        "snapshot_grid": _to_cpu(training_state.get("snapshot_grid")),
        "stats_metrics": _to_cpu(training_state.get("stats_metrics", {})),
    }
    torch.save(payload, path)


def load_network_checkpoint(path_or_url: str, force_fp16: bool = False) -> dict[str, Any]:
    with dnnlib.util.open_url(path_or_url) as f:
        if str(path_or_url).lower().endswith(".pkl"):
            return legacy.load_network_pkl(f, force_fp16=force_fp16)
        raw = f.read()

    bio = io.BytesIO(raw)
    try:
        data = torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        bio.seek(0)
        data = torch.load(bio, map_location="cpu")
    except pickle.UnpicklingError:
        bio.seek(0)
        data = torch.load(bio, map_location="cpu", weights_only=False)

    if is_training_state_checkpoint(data):
        return _deserialize_network_payload(data, force_fp16=force_fp16)
    if not is_state_dict_checkpoint(data):
        raise ValueError(f'Unsupported checkpoint format: "{path_or_url}"')
    return _deserialize_network_payload(data, force_fp16=force_fp16)


def load_training_checkpoint(path_or_url: str, force_fp16: bool = False) -> dict[str, Any]:
    path_or_url = _resolve_training_resume_path(path_or_url)
    with dnnlib.util.open_url(path_or_url) as f:
        if str(path_or_url).lower().endswith(".pkl"):
            data = legacy.load_network_pkl(f, force_fp16=force_fp16)
            data["is_full_state"] = False
            return data
        raw = f.read()

    bio = io.BytesIO(raw)
    try:
        data = torch.load(bio, map_location="cpu", weights_only=False)
    except TypeError:
        bio.seek(0)
        data = torch.load(bio, map_location="cpu")
    except pickle.UnpicklingError:
        bio.seek(0)
        data = torch.load(bio, map_location="cpu", weights_only=False)

    if is_training_state_checkpoint(data):
        result = _deserialize_network_payload(data, force_fp16=force_fp16)
        result.update(
            is_full_state=True,
            optimizers=copy.deepcopy(data.get("optimizers", {})),
            progress=copy.deepcopy(data.get("progress", {})),
            sampler_states=copy.deepcopy(data.get("sampler_states", [])),
            rng_states=copy.deepcopy(data.get("rng_states", [])),
            snapshot_grid=copy.deepcopy(data.get("snapshot_grid")),
            stats_metrics=copy.deepcopy(data.get("stats_metrics", {})),
        )
        return result

    if is_state_dict_checkpoint(data):
        result = _deserialize_network_payload(data, force_fp16=force_fp16)
        result["is_full_state"] = False
        return result

    raise ValueError(f'Unsupported checkpoint format: "{path_or_url}"')
