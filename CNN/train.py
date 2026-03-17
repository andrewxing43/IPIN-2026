import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from CNN import RSSICNN
from brute_dataloader import load_dataset
from utils import set_seed, compute_rmse, compute_ale

def train(
    csv_path,
    selected_ap_index_path=None,
    num_epochs=300,
    batch_size=64,
    learning_rate=1e-3,
    loss_type="smoothl1",
    device="cuda:1" if torch.cuda.is_available() else "cpu",
    save_path="best_model.pth"
):
    set_seed(42)
    print(f"[Device] Using {device}")

    train_loader, val_loader, _= load_dataset(
        csv_path,
        selected_ap_index_path=selected_ap_index_path,
        batch_size=batch_size,
        seed=42
    )

    input_dim = next(iter(train_loader))[0].shape[1]
    print(f"[Model] Input feature dim = {input_dim}")

    model = RSSICNN(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "smoothl1":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported loss type")

    best_val_ale = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for rssi, coord in train_loader:
            rssi, coord = rssi.to(device), coord.to(device)
            optimizer.zero_grad()
            pred = model(rssi)
            loss = criterion(pred, coord)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * rssi.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for rssi, coord in val_loader:
                rssi, coord = rssi.to(device), coord.to(device)
                output = model(rssi)
                preds.append(output.cpu())
                targets.append(coord.cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        preds_real = preds * 30
        targets_real = targets * 30 

        val_rmse = compute_rmse(preds_real, targets_real)
        val_ale = compute_ale(preds_real, targets_real)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f} | ALE: {val_ale:.4f}")

        if val_ale < best_val_ale:
            best_val_ale = val_ale
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved at epoch {epoch}, ALE = {val_ale:.4f}")

    print(f"Training complete. Best ALE = {best_val_ale:.4f}")

if __name__ == "__main__":
    train(
        csv_path="train_final.csv",
        selected_ap_index_path=None,
        num_epochs=500,
        batch_size=16,
        learning_rate=0.0005,
        loss_type="mse",
        save_path="./checkpoints/CNN/best_model.pth"
    )
