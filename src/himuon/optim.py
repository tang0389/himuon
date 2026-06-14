import inspect
from torch.optim import Optimizer
from typing import Any, Optional, Type
from .optimizers import *


def get_optimizer(
    optimizer_name: str, model: Any, kwargs: Optional[dict | Any] = None
) -> Optimizer:
    if optimizer_name.lower() == "adamw":
        # get optimizer arguments
        optim_kwargs = get_optimizer_kwargs(AdamW, kwargs)
        # group parameters
        decay_params = [
            p for p in model.parameters() if p.dim() >= 2 and p.requires_grad
        ]
        nodecay_params = [
            p for p in model.parameters() if p.dim() < 2 and p.requires_grad
        ]
        param_groups = [
            {
                "params": decay_params,
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, **optim_kwargs)
    elif optimizer_name.lower() == "soap":
        optim_kwargs = get_optimizer_kwargs(SOAP, kwargs)
        decay_params = [
            p for p in model.parameters() if p.dim() >= 2 and p.requires_grad
        ]
        nodecay_params = [
            p for p in model.parameters() if p.dim() < 2 and p.requires_grad
        ]
        param_groups = [
            {
                "params": decay_params,
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
            },
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return SOAP(param_groups, **optim_kwargs)
    elif optimizer_name.lower() == "muon":
        optim_kwargs = get_optimizer_kwargs(Muon, kwargs)
        param_dict = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        muon_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw1d_params = [p for name, p in param_dict.items() if p.ndim < 2]
        adamw2d_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and ("embed_tokens" in name or "lm_head" in name)
        ]
        param_groups = [
            {
                "params": muon_params,
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": True,
            },
            {
                "params": adamw1d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "params": adamw2d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": False,
            },
        ]
        return Muon(param_groups, **optim_kwargs)
    elif optimizer_name.lower() == "muon-fsdp":
        optim_kwargs = get_optimizer_kwargs(MuonFsdp, kwargs)
        param_dict = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        muon_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw1d_params = [p for name, p in param_dict.items() if p.ndim < 2]
        adamw2d_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and ("embed_tokens" in name or "lm_head" in name)
        ]
        param_groups = [
            {
                "params": muon_params,
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": True,
            },
            {
                "params": adamw1d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "params": adamw2d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": False,
            },
        ]
        return MuonFsdp(param_groups, **optim_kwargs)
    elif optimizer_name.lower() == "himuon":
        optim_kwargs = get_optimizer_kwargs(HiMuon, kwargs)
        param_dict = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        muon_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw1d_params = [p for name, p in param_dict.items() if p.ndim < 2]
        adamw2d_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and ("embed_tokens" in name or "lm_head" in name)
        ]
        param_groups = [
            {
                "params": muon_params,
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": True,
            },
            {
                "params": adamw1d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "params": adamw2d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": False,
            },
        ]
        return HiMuon(param_groups, **optim_kwargs)
    elif optimizer_name.lower() == "himuon-legacy":
        optim_kwargs = get_optimizer_kwargs(HiMuonLegacy, kwargs)
        param_dict = {
            name: p for name, p in model.named_parameters() if p.requires_grad
        }
        muon_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw1d_params = [p for name, p in param_dict.items() if p.ndim < 2]
        adamw2d_params = [
            p
            for name, p in param_dict.items()
            if p.ndim >= 2 and ("embed_tokens" in name or "lm_head" in name)
        ]
        param_groups = [
            {
                "params": muon_params,
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": True,
            },
            {
                "params": adamw1d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "params": adamw2d_params,
                "lr": optim_kwargs.get("adamw_lr", 1e-3),
                "weight_decay": optim_kwargs.get("weight_decay", 0.1),
                "use_muon": False,
            },
        ]
        return HiMuonLegacy(param_groups, **optim_kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


def get_optimizer_kwargs(
    optimizer_class: Type[Optimizer], kwargs: Optional[dict | Any] = None
) -> dict:
    kwargs = vars(kwargs) if kwargs else {}
    optimizer_keys = inspect.signature(optimizer_class).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in optimizer_keys and v is not None}
