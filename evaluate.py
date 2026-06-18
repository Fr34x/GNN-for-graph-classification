import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import set_seed, get_dataset, get_dataloaders
from train import get_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN Graph Classification Model")
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pt', help="Path to saved model checkpoint")
    parser.add_argument('--plot_path', type=str, default='confusion_matrix.png', help="Path to save confusion matrix plot")
    
    args = parser.parse_args()
    
    # Load model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print(f"=== Evaluating Model ===")
    print(f"Model Type    : {config['model_type']}")
    print(f"Pooling       : {config['pooling']}")
    print(f"Num Layers    : {config['num_layers']}")
    print(f"Hidden Dim    : {config['hidden_dim']}")
    print(f"Use Features  : {config['use_features']}")
    print(f"Use BN        : {config['use_bn']}")
    print(f"Use Residual  : {config['use_residual']}")
    print(f"Seed          : {config['seed']}")
    print(f"Best Val Epoch: {checkpoint['epoch']}")
    print(f"=========================\n")
    
    # Ensure exact same seed and splits
    set_seed(config['seed'])
    
    # Load dataset & loaders
    dataset = get_dataset(use_features=config['use_features'])
    _, _, test_loader = get_dataloaders(dataset, batch_size=config['batch_size'], seed=config['seed'])
    
    # Initialize and load model
    model = get_model(config, dataset.num_node_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            correct += int((preds == batch.y).sum())
            total += batch.num_graphs
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch.y.cpu().tolist())
            
    test_loss = total_loss / total
    test_acc = correct / total
    
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({correct}/{total} graphs correctly classified)")
    print("\nClassification Report:")
    report = classification_report(all_targets, all_preds, target_names=['Non-Enzyme', 'Enzyme'])
    print(report)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Enzyme', 'Enzyme'])
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title(f"Confusion Matrix: {config['model_type']} (Test Acc: {test_acc:.3f})")
    plt.tight_layout()
    plt.savefig(args.plot_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to {args.plot_path}")

if __name__ == '__main__':
    main()
