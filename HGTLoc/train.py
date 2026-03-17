import torch
import os
from torch import nn
from model import RSSITransformerModel
from dataloader1 import load_data_and_build_dataloaders
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from utils import set_seed, save_checkpoint, plot_loss_curve, ale, mse_loss, ale_loss
import numpy as np

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data['label'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_ale(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu())
            all_targets.append(data['label'].cpu())
    
    pred = torch.cat(all_preds, dim=0) * 30
    target = torch.cat(all_targets, dim=0) * 30
    return ale(pred, target)

def main():
    set_seed(42)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

   
    train_loader, val_loader, test_loader = load_data_and_build_dataloaders('train.csv', 'test.csv', batch_size=32)
    
    model = RSSITransformerModel().to(device)

    base_lr = 0.001
    min_lr = 0.0001
    warmup_epochs = 30
    total_epochs = 400
    early_stop_patience = 40

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler_const = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)

    
    criterion = mse_loss()

    best_val_ale = float('inf')
    train_losses = []
    val_ales = []
    no_improve_count = 0
 
    os.makedirs('./checkpoints/new', exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_ale = evaluate_ale(model, val_loader, device)

        train_losses.append(train_loss)
        val_ales.append(val_ale)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val ALE: {val_ale:.4f} | LR: {current_lr:.2e}")

        if epoch <= warmup_epochs:
            scheduler_const.step()
        else:
            scheduler_cosine.step()

        if val_ale < best_val_ale:
            best_val_ale = val_ale
            save_checkpoint(model, './checkpoints/new/best_model.pth')
            print("Saved best model.")
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. No improvement in {early_stop_patience} consecutive epochs.")
            break

    print("\nTraining completed.")
    print(f"Best Validation ALE: {best_val_ale:.4f}")

    plot_loss_curve(train_losses, val_ales)

if __name__ == '__main__':
    main()