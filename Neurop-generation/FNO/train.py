import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional
import os

def train_step(model, encoder, loss_fn, optimizer, batch, device):
    model.train()
    
    events = batch["events"].to(device)
    batch_idx = batch["batch_idx"].to(device)
    adj_true = batch["adjacency"].to(device)
    
    optimizer.zero_grad()

    B = batch["adjacency"].shape[0]
    spikes = encoder(events, batch_idx, B=B)
    out = model(spikes)
    loss = loss_fn(out["adj_logits"], adj_true)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def val_step(model, encoder, loss_fn, batch, device):
    model.eval()
    
    events = batch["events"].to(device)
    batch_idx = batch["batch_idx"].to(device)
    adj_true = batch["adjacency"].to(device)
    
    B = batch["adjacency"].shape[0]
    spikes = encoder(events, batch_idx, B=B)
    out = model(spikes)
    loss = loss_fn(out["adj_logits"], adj_true)
    
    return loss.item()

def train_epoch(model, encoder, loss_fn, optimizer, dataloader, device):
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Train", leave=False):
        loss = train_step(model, encoder, loss_fn, optimizer, batch, device)
        total_loss += loss
    
    return total_loss / len(dataloader)

@torch.no_grad()
def val_epoch(model, encoder, loss_fn, dataloader, device):
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Val", leave=False):
        loss = val_step(model, encoder, loss_fn, batch, device)
        total_loss += loss
    
    return total_loss / len(dataloader)

def train(
    model,
    encoder,
    loss_fn,
    train_loader,
    val_loader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    save_every: int = 10,
):
    model.to(device)
    encoder.to(device)
    loss_fn.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    best_val_loss = float("inf")
    train_loss = [0.0] * n_epochs
    val_loss = [0.0] * n_epochs
    
    for epoch in range(n_epochs):
        train_loss[epoch] = train_epoch(model, encoder, loss_fn, optimizer, train_loader, device)
        val_loss[epoch] = val_epoch(model, encoder, loss_fn, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss[epoch]:.4f} | Val: {val_loss[epoch]:.4f}")
        
        if save_dir and val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, os.path.join(save_dir, "best.pt"))
        
        if save_dir and (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "encoder": encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, f"checkpoint_{epoch+1}.pt"))
    
    return {"train_loss": train_loss, "val_loss": val_loss}


def load_checkpoint(path, model, encoder, optimizer=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model"])
    encoder.load_state_dict(checkpoint["encoder"])
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("val_loss", None)
