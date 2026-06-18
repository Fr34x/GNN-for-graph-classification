import os
import time
import argparse
import torch
import torch.nn as nn
from utils import set_seed, get_dataset, get_dataloaders
from models import GCNClassifier, SAGEClassifier, GINClassifier

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_model(config, num_features):
    """
    Instantiate model based on type.
    """
    model_type = config['model_type'].upper()
    if model_type == 'GCN':
        return GCNClassifier(
            num_features=num_features,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            pooling=config['pooling'],
            dropout=config['dropout'],
            use_bn=config['use_bn'],
            use_residual=config['use_residual']
        )
    elif model_type == 'SAGE':
        return SAGEClassifier(
            num_features=num_features,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            pooling=config['pooling'],
            dropout=config['dropout'],
            use_bn=config['use_bn'],
            use_residual=config['use_residual']
        )
    elif model_type == 'GIN':
        return GINClassifier(
            num_features=num_features,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            pooling=config['pooling'],
            dropout=config['dropout'],
            use_bn=config['use_bn'],
            use_residual=config['use_residual']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def validate_pipeline(model, train_loader, device):
    """
    Perform shape and gradient connectivity checks on the model.
    """
    model.eval()
    # 1. Shape validation
    dummy_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        out = model(dummy_batch)
    expected_shape = (dummy_batch.y.size(0), 2)
    assert out.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {out.shape}"
    print(f"Shape check passed! Output shape: {out.shape}")
    
    # 2. Gradient flow validation
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    out = model(dummy_batch)
    loss = criterion(out, dummy_batch.y)
    loss.backward()
    
    non_zero_grads = sum(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.parameters()
    )
    print(f"Gradient check: {non_zero_grads} parameter blocks have non-zero gradients.")
    assert non_zero_grads > 0, "Backward pass failed: No gradients detected!"
    print("Gradient flow check passed!")
    optimizer.zero_grad()

def run_sanity_overfit(dataset, config, device):
    """
    Sanity check to overfit the GNN on a tiny subset of 5 graphs.
    """
    print("\n--- Running Tiny-Subset Sanity Overfit Test ---")
    tiny_indices = list(range(5))
    # Collect small subset list of Data objects
    tiny_set = [dataset[i] for i in tiny_indices]
    
    from torch_geometric.loader import DataLoader as PyGDataLoader
    tiny_loader = PyGDataLoader(tiny_set, batch_size=5, shuffle=False)
    
    model = get_model(config, dataset.num_node_features)
    model.to(device)
    model.train()
    
    # Use larger lr for quick overfit convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    success = False
    for epoch in range(1, 101):
        for batch in tiny_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        if loss.item() < 0.01:
            print(f"Sanity overfit check passed at epoch {epoch}! Final Loss: {loss.item():.6f}")
            success = True
            break
            
    if not success:
        print(f"Warning: Sanity overfit check did not reach loss < 0.01. Final Loss: {loss.item():.6f}")
    print("------------------------------------------------\n")
    return success

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
        
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.num_graphs
            
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description="GNN Graph Classification on PROTEINS dataset")
    parser.add_argument('--model_type', type=str, default='GIN', choices=['GCN', 'SAGE', 'GIN'])
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'sum', 'max'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--use_features', type=str2bool, default=True)
    parser.add_argument('--use_bn', type=str2bool, default=True)
    parser.add_argument('--use_residual', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--save_path', type=str, default='best_model.pt')
    parser.add_argument('--skip_sanity', action='store_true', help="Skip overfit sanity check")
    
    args = parser.parse_args()
    config = vars(args)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Load dataset and splits
    dataset = get_dataset(use_features=config['use_features'])
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=config['batch_size'], seed=config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize GNN model
    model = get_model(config, dataset.num_node_features)
    model.to(device)
    
    # Shape & gradient flow validation
    validate_pipeline(model, train_loader, device)
    
    # Sanity overfit test
    if not config['skip_sanity']:
        run_sanity_overfit(dataset, config, device)
        # Re-seed to ensure full training has the same start states/splits
        set_seed(config['seed'])
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    print("=== Starting Training ===")
    
    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        # Log epoch summary
        print(f"[Epoch {epoch:03d}] Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f}, acc={val_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.5f} | time={elapsed:.2f}s")
        
        # Checkpoint based on best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, config['save_path'])
            
    print(f"Training finished! Best validation accuracy: {best_val_acc:.4f}. Model saved to {config['save_path']}")

if __name__ == '__main__':
    main()
