# experiment_manager.py (Updated)
from src.models.model_factory import create_model
from src.models.optimizer_factory import get_optimizer, get_criterion

def get_experiment_setup(cfg, num_classes):
    # Initialize Model
    model = create_model(
        model_name=cfg["model_name"],
        num_classes=num_classes,
        dropout=cfg["dropout"],
        freeze_backbone=cfg["freeze_backbone"],
        unfreeze_last_n=cfg.get("unfreeze_last_n", 1)
    )

    # Filter Parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer & Loss
    optimizer = get_optimizer(trainable_params, cfg)
    criterion = get_criterion(cfg)

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "cfg": cfg
    }