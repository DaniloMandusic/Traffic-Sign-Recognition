# model_configs.py

CONFIGS = {
    # Experiment 1: The most basic "Feature Extraction"
    # Only the final output layer (Head) is trained.
    "exp001": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 64,
        "lr": 1e-3,
        "freeze_backbone": True,
        "unfreeze_last_n": 1,         # Only the classifier head
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },

    # Experiment 2: Moderate Fine-Tuning
    # Frozen backbone but unfreezing the last 3 blocks + Head to refine vision.
    "exp002": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 48,             # Slightly smaller batch as more gradients are stored
        "lr": 5e-4,                   # Lower LR to protect weights
        "freeze_backbone": True,
        "unfreeze_last_n": 2,         # Head + roughly 1 blocks
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },

    "exp003": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 48,             # Slightly smaller batch as more gradients are stored
        "lr": 5e-4,                   # Lower LR to protect weights
        "freeze_backbone": True,
        "unfreeze_last_n": 4,         # Head + roughly 3 blocks
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },

    "exp004": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 48,             # Slightly smaller batch as more gradients are stored
        "lr": 5e-4,                   # Lower LR to protect weights
        "freeze_backbone": True,
        "unfreeze_last_n": 6,         # Head + roughly 3 blocks
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },

    "exp005": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 48,
        "lr": 3e-4,
        "freeze_backbone": True,
        "unfreeze_last_n": 6,
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },

    "exp006": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 48,
        "lr": 3e-4,
        "freeze_backbone": True,
        "unfreeze_last_n": 6,
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.4,
        "weight_decay": 1e-4,
    },

    "exp007": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 48,
        "lr": 3e-4,
        "freeze_backbone": False,
        "unfreeze_last_n": 6,
        "optimizer_name": "adam",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-4,
    },

    "exp008": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 32,
        "lr": 3e-4,
        "freeze_backbone": True,
        "unfreeze_last_n": 6,
        "optimizer_name": "sgd",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-2,
        "momentum": 0.9,
    },

    "exp009": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 32,
        "lr": 1e-4,
        "freeze_backbone": True,
        "unfreeze_last_n": 6,
        "optimizer_name": "sgd",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-2,
        "momentum": 0.9,
    },

    "exp010": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 32,
        "lr": 5e-4,
        "freeze_backbone": True,
        "unfreeze_last_n": 6,
        "optimizer_name": "sgd",
        "loss_name": "cross_entropy",
        "dropout": 0.3,
        "weight_decay": 1e-2,
        "momentum": 0.9,
    }
}