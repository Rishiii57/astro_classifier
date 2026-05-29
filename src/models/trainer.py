import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import wandb
import os


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, val_loader, config, device, class_names):
    """
    config = {
        'epochs': 10,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'checkpoint_dir': 'outputs/checkpoints/',
        'run_name': 'phase1',
        'use_wandb': True,
    }
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    if config.get('use_wandb'):
        wandb.init(project='astro-classifier', name=config['run_name'], config=config)

    best_val_acc = 0.0
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if config.get('use_wandb'):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0]
            })

        # Saving the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(config['checkpoint_dir'], f"{config['run_name']}_best.pth")
            torch.save(model.state_dict(), path)
            print(f"   New best model saved ({val_acc:.4f})")

    if config.get('use_wandb'):
        wandb.finish()

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return best_val_acc


def final_evaluate(model, test_loader, device, class_names):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}\n")
    print(classification_report(labels, preds, target_names=class_names))

    return confusion_matrix(labels, preds)