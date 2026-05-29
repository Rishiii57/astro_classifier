import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.dataset import get_dataloaders
from src.models.model import build_model, get_model_info
from src.models.trainer import train, final_evaluate

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

#  Data
train_loader, val_loader, test_loader, class_names = get_dataloaders(
    images_path='data/raw/galaxy10_images.npy',
    labels_path='data/raw/galaxy10_labels.npy',
    kaggle_base_dir='data/kaggle_space/space images',
    batch_size=32,
    kaggle_target_count=800,   # augment Nebula/Planets/Stars to 800 each
)

#  Phase 1: training classifier head only 
print("\n=== Phase 1: training classifier head ===")
model = build_model(num_classes=7, freeze_backbone=True)
model = model.to(device)
get_model_info(model)

config_phase1 = {
    'epochs': 10,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'checkpoint_dir': 'outputs/checkpoints/',
    'run_name': 'phase1_v2',   # v2 = combined 8-class dataset
    'use_wandb': False,
}

train(model, train_loader, val_loader, config_phase1, device, class_names)

#  Phase 2: fine-tuning full network 
print("\n=== Phase 2: fine-tuning full network ===")
model = build_model(num_classes=7, freeze_backbone=False)
model.load_state_dict(torch.load('outputs/checkpoints/phase1_v2_best.pth', map_location=device))
model = model.to(device)
get_model_info(model)

config_phase2 = {
    'epochs': 15,              
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'checkpoint_dir': 'outputs/checkpoints/',
    'run_name': 'phase2_v2',
    'use_wandb': False,
}

train(model, train_loader, val_loader, config_phase2, device, class_names)

#Final evaluation 
print("\n=== Final test set evaluation ===")
model.load_state_dict(torch.load('outputs/checkpoints/phase2_v2_best.pth', map_location=device))
model = model.to(device)
cm = final_evaluate(model, test_loader, device, class_names)
print("\nConfusion matrix:")
print(cm)