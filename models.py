import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class GNNBase(nn.Module):
    """
    Base class for GNN models. Handles global pooling and final classification.
    """
    def __init__(self, hidden_dim, num_classes=2, pooling='mean', dropout=0.5):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        
        # MLP Classifier applied after pooling
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def pool(self, x, batch):
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            return global_add_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

class GCNClassifier(GNNBase):
    """
    GCN model for Graph Classification.
    """
    def __init__(self, num_features, hidden_dim, num_layers=3, num_classes=2,
                 pooling='mean', dropout=0.5, use_bn=True, use_residual=True):
        super().__init__(hidden_dim, num_classes, pooling, dropout)
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i in range(self.num_layers):
            prev_x = x
            x = self.convs[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply residual connection (if dimensions match)
            if self.use_residual and prev_x.size() == x.size():
                x = x + prev_x
                
        # Global Pooling
        x = self.pool(x, batch)
        
        # Final classification
        return self.classifier(x)

class SAGEClassifier(GNNBase):
    """
    GraphSAGE model for Graph Classification.
    """
    def __init__(self, num_features, hidden_dim, num_layers=3, num_classes=2,
                 pooling='mean', dropout=0.5, use_bn=True, use_residual=True):
        super().__init__(hidden_dim, num_classes, pooling, dropout)
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        
        self.convs.append(SAGEConv(num_features, hidden_dim))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i in range(self.num_layers):
            prev_x = x
            x = self.convs[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if self.use_residual and prev_x.size() == x.size():
                x = x + prev_x
                
        x = self.pool(x, batch)
        return self.classifier(x)

class GINClassifier(GNNBase):
    """
    GIN (Graph Isomorphism Network) model for Graph Classification.
    """
    def __init__(self, num_features, hidden_dim, num_layers=3, num_classes=2,
                 pooling='mean', dropout=0.5, use_bn=True, use_residual=True):
        super().__init__(hidden_dim, num_classes, pooling, dropout)
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_bn else None
        
        # In GIN, each layer requires its own MLP
        # First layer MLP
        mlp1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp1, train_eps=True))
        if use_bn:
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        # Hidden layer MLPs
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i in range(self.num_layers):
            prev_x = x
            x = self.convs[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if self.use_residual and prev_x.size() == x.size():
                x = x + prev_x
                
        x = self.pool(x, batch)
        return self.classifier(x)
