# model_configs.py

CONFIGS = {
    # Experiment 1: The most basic "Feature Extraction"
    # Only the final output layer (Head) is trained.
    "exp001_b0_frozen_head": {
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
    "exp002_b0_partial_unfreeze": {
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

    # Experiment 3: Full Fine-Tuning
    # Unfreeze everything. Uses SGD for better generalization during fine-tuning.
    "exp003_b0_full_unfreeze": {
        "model_name": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 32,
        "lr": 1e-5,                   # Very low LR to avoid destroying pre-trained info
        "freeze_backbone": False,     # Logic in factory will ignore unfreeze_last_n
        "optimizer_name": "sgd",      # SGD is often preferred for late-stage fine-tuning
        "loss_name": "cross_entropy",
        "dropout": 0.4,
        "weight_decay": 1e-2,
        "momentum": 0.9,
    }
}