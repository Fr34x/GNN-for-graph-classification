import os
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_seed, get_dataset, get_dataloaders
from train import get_model, train, evaluate

# Create plots folder if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Helper to run a single experiment run
def run_experiment(config):
    print(f"\n>>> Running Config: Model={config['model_type']}, Pooling={config['pooling']}, "
          f"Layers={config['num_layers']}, Dim={config['hidden_dim']}, "
          f"Features={config['use_features']}, BN={config['use_bn']}, Residual={config['use_residual']} ...")
    
    set_seed(config['seed'])
    
    dataset = get_dataset(use_features=config['use_features'])
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, batch_size=config['batch_size'], seed=config['seed']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(config, dataset.num_node_features)
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best model for test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Results -> Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    return {
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'history': history
    }

def main():
    default_config = {
        'model_type': 'GIN',
        'pooling': 'mean',
        'num_layers': 3,
        'hidden_dim': 64,
        'dropout': 0.5,
        'use_features': True,
        'use_bn': True,
        'use_residual': True,
        'epochs': 80,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'seed': 2025
    }
    
    summary_results = []
    
    # ----------------------------------------------------
    # 1. Model Architecture Comparison
    # ----------------------------------------------------
    print("\n====================================================")
    print("EXPERIMENT 1: Model Architecture Comparison")
    print("====================================================")
    model_results = {}
    for m_type in ['GCN', 'SAGE', 'GIN']:
        cfg = default_config.copy()
        cfg['model_type'] = m_type
        res = run_experiment(cfg)
        summary_results.append({
            'Experiment': 'Model Type',
            'Config': m_type,
            'Val Acc': res['best_val_acc'],
            'Test Acc': res['test_acc']
        })
        model_results[m_type] = res
        
    # Plot training curves for models
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for m_type, res in model_results.items():
        plt.plot(res['history']['train_loss'], label=f"{m_type} Train")
        plt.plot(res['history']['val_loss'], linestyle='--', label=f"{m_type} Val")
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for m_type, res in model_results.items():
        plt.plot(res['history']['train_acc'], label=f"{m_type} Train")
        plt.plot(res['history']['val_acc'], linestyle='--', label=f"{m_type} Val")
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()

    # ----------------------------------------------------
    # 2. Pooling Ablation
    # ----------------------------------------------------
    print("\n====================================================")
    print("EXPERIMENT 2: Pooling Ablation")
    print("====================================================")
    pooling_results = {}
    for pool in ['mean', 'sum', 'max']:
        cfg = default_config.copy()
        cfg['pooling'] = pool
        res = run_experiment(cfg)
        summary_results.append({
            'Experiment': 'Pooling',
            'Config': pool,
            'Val Acc': res['best_val_acc'],
            'Test Acc': res['test_acc']
        })
        pooling_results[pool] = res
        
    # Plot pooling comparison
    plt.figure(figsize=(6, 4))
    pools = list(pooling_results.keys())
    test_accs = [pooling_results[p]['test_acc'] for p in pools]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.bar(pools, test_accs, color=colors, edgecolor='black', width=0.5)
    plt.title('Pooling Method Impact on Test Accuracy')
    plt.ylabel('Test Accuracy')
    plt.ylim(0.5, 0.85)
    for i, acc in enumerate(test_accs):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('plots/pooling_comparison.png')
    plt.close()

    # ----------------------------------------------------
    # 3. Feature Ablation
    # ----------------------------------------------------
    print("\n====================================================")
    print("EXPERIMENT 3: Feature Ablation")
    print("====================================================")
    feature_results = {}
    for use_feats in [True, False]:
        cfg = default_config.copy()
        cfg['use_features'] = use_feats
        res = run_experiment(cfg)
        label = "With Node Features" if use_feats else "Constant Features"
        summary_results.append({
            'Experiment': 'Features',
            'Config': label,
            'Val Acc': res['best_val_acc'],
            'Test Acc': res['test_acc']
        })
        feature_results[label] = res
        
    # Plot feature comparison
    plt.figure(figsize=(6, 4))
    feats = list(feature_results.keys())
    test_accs = [feature_results[f]['test_acc'] for f in feats]
    plt.bar(feats, test_accs, color=['#3f51b5', '#ff9800'], edgecolor='black', width=0.4)
    plt.title('Node Features vs. No Node Features')
    plt.ylabel('Test Accuracy')
    plt.ylim(0.5, 0.85)
    for i, acc in enumerate(test_accs):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('plots/features_comparison.png')
    plt.close()

    # ----------------------------------------------------
    # 4. Regularization Ablation
    # ----------------------------------------------------
    print("\n====================================================")
    print("EXPERIMENT 4: Regularization Ablation")
    print("====================================================")
    reg_configs = {
        'None': {'dropout': 0.0, 'use_bn': False, 'use_residual': False},
        'Dropout Only': {'dropout': 0.5, 'use_bn': False, 'use_residual': False},
        'BatchNorm Only': {'dropout': 0.0, 'use_bn': True, 'use_residual': False},
        'Residual Only': {'dropout': 0.0, 'use_bn': False, 'use_residual': True},
        'Full Regularization': {'dropout': 0.5, 'use_bn': True, 'use_residual': True}
    }
    reg_results = {}
    for name, params in reg_configs.items():
        cfg = default_config.copy()
        cfg['dropout'] = params['dropout']
        cfg['use_bn'] = params['use_bn']
        cfg['use_residual'] = params['use_residual']
        res = run_experiment(cfg)
        summary_results.append({
            'Experiment': 'Regularization',
            'Config': name,
            'Val Acc': res['best_val_acc'],
            'Test Acc': res['test_acc']
        })
        reg_results[name] = res
        
    # Plot regularization comparison
    plt.figure(figsize=(10, 5))
    names = list(reg_results.keys())
    test_accs = [reg_results[n]['test_acc'] for n in names]
    plt.barh(names, test_accs, color='#9c27b0', edgecolor='black', height=0.5)
    plt.title('Impact of Regularization on Test Accuracy')
    plt.xlabel('Test Accuracy')
    plt.xlim(0.5, 0.85)
    for i, acc in enumerate(test_accs):
        plt.text(acc + 0.005, i, f"{acc:.4f}", va='center')
    plt.tight_layout()
    plt.savefig('plots/regularization_comparison.png')
    plt.close()

    # ----------------------------------------------------
    # 5. Depth and Capacity Ablation
    # ----------------------------------------------------
    print("\n====================================================")
    print("EXPERIMENT 5: Depth and Capacity Ablation")
    print("====================================================")
    depth_capacity_results = []
    for layers in [2, 3, 5]:
        for h_dim in [32, 64, 128]:
            cfg = default_config.copy()
            cfg['num_layers'] = layers
            cfg['hidden_dim'] = h_dim
            res = run_experiment(cfg)
            summary_results.append({
                'Experiment': 'Depth/Capacity',
                'Config': f"Layers={layers}, Dim={h_dim}",
                'Val Acc': res['best_val_acc'],
                'Test Acc': res['test_acc']
            })
            depth_capacity_results.append({
                'Layers': layers,
                'Hidden Dim': h_dim,
                'Test Acc': res['test_acc']
            })
            
    # Plot Depth/Capacity Heatmap or Table visualization
    df_dc = pd.DataFrame(depth_capacity_results)
    pivot_df = df_dc.pivot(index='Layers', columns='Hidden Dim', values='Test Acc')
    
    plt.figure(figsize=(7, 5))
    # We display values as text inside a styled grid
    plt.imshow(pivot_df.values, cmap='Purples', interpolation='nearest')
    plt.title('Test Accuracy vs. Depth & Capacity')
    plt.colorbar(label='Test Accuracy')
    plt.xticks(range(len(pivot_df.columns)), pivot_df.columns)
    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.xlabel('Hidden Channels (Capacity)')
    plt.ylabel('GNN Layers (Depth)')
    # Add values text
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.values[i, j]
            plt.text(j, i, f"{val:.4f}", ha='center', va='center', color='black' if val < 0.76 else 'white')
    plt.tight_layout()
    plt.savefig('plots/depth_capacity_comparison.png')
    plt.close()

    # ----------------------------------------------------
    # Save all summary statistics to CSV
    # ----------------------------------------------------
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv('plots/experiments_summary.csv', index=False)
    print("\n====================================================")
    print("ALL EXPERIMENTS COMPLETED!")
    print("Summary saved to plots/experiments_summary.csv")
    print("====================================================\n")
    print(df_summary)

if __name__ == '__main__':
    main()
