import torch

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop
}

CRITERION_REGISTRY = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "label_smoothing": torch.nn.CrossEntropyLoss,
}

def get_optimizer(model_params, cfg):
    name = cfg["optimizer_name"].lower()
    lr = cfg["lr"]
    wd = cfg["weight_decay"]

    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{name}' not found"
                         f"Available: {list(OPTIMIZER_REGISTRY.keys())}")

    opt_class = OPTIMIZER_REGISTRY[name]

    kwargs = {
        "params": model_params,
        "lr": lr,
        "weight_decay": wd
    }

    if name == "sgd":
        kwargs["momentum"] = cfg.get("momentum", 0.9)

    return opt_class(**kwargs)

def get_criterion(cfg):
    name = cfg["loss_name"].lower()

    if name not in CRITERION_REGISTRY:
        raise ValueError(f"Criterion '{name}' not found"
                         f"Available: {list(CRITERION_REGISTRY.keys())}")

    criterion_class = CRITERION_REGISTRY[name]

    kwargs = {}
    if name == "label_smoothing":
        kwargs["label_smoothing"] = cfg.get("label_smoothing_val", 0.1)

    return criterion_class(**kwargs)