import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from model import adjacency_loss, HierarchicalNeuralOperator
from dataset import SpikeGraphDataset

def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for spikes, adj in tqdm(dataloader, desc="Train"):
        # spikes: (N, T_ds) -> (B, N, T_ds)
        spikes = spikes.to(device)
        adj = adj.to(device)

        # Add batch dimension if dataset returns single sample per item
        # (We can also set batch_size > 1 in DataLoader, then spikes: (B, N, T_ds))
        if spikes.dim() == 2:
            spikes = spikes.unsqueeze(0)
            adj = adj.unsqueeze(0)

        optimizer.zero_grad()
        A_hat = model(spikes)  # (B, N, N)

        loss = adjacency_loss(A_hat, adj)
        loss.backward()

        # --- gradient clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for spikes, adj in tqdm(dataloader, desc="Val"):
        spikes = spikes.to(device)
        adj = adj.to(device)
        if spikes.dim() == 2:
            spikes = spikes.unsqueeze(0)
            adj = adj.unsqueeze(0)

        A_hat = model(spikes)
        loss = adjacency_loss(A_hat, adj)
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--downsample_mode", type=str, default="sum")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")

    # hyperparameters
    parser.add_argument("--n_levels_time", type=int, default=3)
    parser.add_argument("--time_base_dim", type=int, default=64)
    parser.add_argument("--neuron_emb_dim", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--edge_hidden", type=int, default=128)

    # model saving
    parser.add_argument("--best_ckpt", type=str, default="best_model.pt")
    parser.add_argument("--last_ckpt", type=str, default="last_model.pt")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = SpikeGraphDataset(
        data_dir=args.data_dir + "/train/",
        downsample_factor=args.downsample_factor,
        downsample_mode=args.downsample_mode,
        default_T=10000,
        device=device,
    )
    val_ds = SpikeGraphDataset(
        data_dir=args.data_dir + "/test/",
        downsample_factor=args.downsample_factor,
        downsample_mode=args.downsample_mode,
        default_T=10000,
        device=device,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = HierarchicalNeuralOperator(
        n_levels_time=args.n_levels_time,
        time_base_dim=args.time_base_dim,
        neuron_emb_dim=args.neuron_emb_dim,
        group_size=args.group_size,
        n_heads=args.n_heads,
        edge_hidden=args.edge_hidden,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,   # halve LR when plateau
        patience=3,   # wait 3 epochs with no improvement
        verbose=True,
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- NEW: save best-val checkpoint ---
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(best_state, "best_model.pt")
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        # --- NEW: step LR scheduler on validation loss ---
        scheduler.step(val_loss)

    # Optional: save last epoch as well
    torch.save(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        "last_model.pt",
    )

    print(f"Training complete. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
