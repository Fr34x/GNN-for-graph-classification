# Project Report: Graph Neural Networks for Graph Classification

## 1. Abstract & Motivation
Real-world structural data such as molecular compounds, social networks, and protein structures are naturally represented as graphs. Unlike grid-like structures (e.g. images), graphs possess irregular, non-Euclidean topologies, making standard Deep Learning architectures like CNNs unsuitable. Graph Neural Networks (GNNs) extend deep learning to graph-structured data by employing a message passing framework where node features are iteratively aggregated from local neighborhoods.

This project focuses on building an end-to-end graph classification pipeline to predict whether a protein is an enzyme or a non-enzyme, using the benchmark `PROTEINS` dataset. We systematically analyze key GNN architectural choices, including model type, graph pooling methods, node feature usage, network capacity/depth, and regularization techniques, to evaluate their impact on generalization performance.

---

## 2. Dataset Description: PROTEINS
The `PROTEINS` dataset from TUDataset consists of:
*   **Total Graphs**: 1,113 protein graphs.
*   **Target Classes**: Binary classification (Class 0: Non-Enzyme, Class 1: Enzyme).
*   **Average Nodes per Graph**: ~39.06 nodes.
*   **Average Edges per Graph**: ~72.82 edges.
*   **Node Features**: A 3-dimensional one-hot vector indicating secondary structure elements (SSE) type: helix, sheet, or turn.

---

## 3. Methodology & GNN Models
We implemented three GNN architectures for graph-level prediction:
1.  **Graph Convolutional Network (GCN)**: Applies a localized first-order approximation of spectral graph convolutions:
    $$h_v^{(l+1)} = \text{ReLU}\left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_u \tilde{d}_v}} W^{(l)} h_u^{(l)} \right)$$
2.  **GraphSAGE**: Learns aggregator functions that generalize message passing to unseen nodes by concatenating the node's own representation with its neighborhood aggregation:
    $$h_v^{(l+1)} = \text{ReLU}\left( W_1 h_v^{(l)} + W_2 \cdot \text{Aggregate}\left(\{h_u^{(l)}, \forall u \in \mathcal{N}(v)\}\right) \right)$$
3.  **Graph Isomorphism Network (GIN)**: A maximally powerful GNN that is as discriminative as the 1-Weisfeiler-Lehman (1-WL) isomorphism test. It models neighborhood aggregation using a Multi-Layer Perceptron (MLP):
    $$h_v^{(l+1)} = \text{MLP}^{(l)} \left( (1 + \epsilon^{(l)}) h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} h_u^{(l)} \right)$$

### Graph-Level Classification Pipeline
*   **Message Passing**: Node representations are updated for $L$ layers.
*   **Global Pooling (Readout)**: Aggregates updated node features into a fixed-length graph representation:
    $$h_G = \text{Readout}\left( \{h_v^{(L)}, \forall v \in G\} \right)$$
    We compare Mean, Sum, and Max pooling.
*   **MLP Classifier**: Maps $h_G$ to class logits:
    $$\hat{y}_G = \text{Linear}\left(\text{ReLU}\left(\text{Dropout}\left(\text{Linear}(h_G)\right)\right)\right)$$

---

## 4. Reproducibility & Pipeline Validation
Adhering to the course best practices, we integrated several diagnostic checks into the pipeline:
1.  **Reproducibility**: Global seeding (`SEED = 2025`) of `random`, `numpy`, and `torch` libraries to ensure identical splits and training trajectories across runs.
2.  **Hyperparameter Centralization**: Configuration parameters are managed centrally in a python dictionary.
3.  **Forward Shape Check**: Programmatic validation of the output shape before training to ensure batch-level dimensions map correctly: `assert out.shape == (batch_size, num_classes)`.
4.  **Backward Gradient Flow Check**: Asserting that a loss backward pass populates gradients for all parameter blocks, preventing silent bugs like disconnected layers or dead activations.
5.  **Sanity Overfit Test**: Overfitting the model on a tiny subset of 5 graphs. The model successfully converged to a loss $< 0.01$ within 5 epochs, verifying pipeline correctness.

---

## 5. Ablation Studies & Results
All experiments were run with an 80/10/10 train/val/test split. Below is the summary of results:

### 5.1 Model Architecture Comparison
We compared GCN, GraphSAGE, and GIN under default hyperparameters (3 layers, 64 hidden channels, mean pooling, dropout=0.5, batch normalization, residuals).
*   **GCN**: Val Acc = 82.88%, Test Acc = 67.86%
*   **GraphSAGE**: Val Acc = 78.38%, Test Acc = 52.68%
*   **GIN**: Val Acc = 81.08%, Test Acc = 68.75%

*Analysis*: GIN and GCN perform strongly on PROTEINS, while GraphSAGE exhibits suboptimal generalization under default conditions. GIN's expressive capacity aligns well with the structural task.

![Model Comparison](plots/model_comparison.png)

### 5.2 Pooling Ablation
We compared Global Mean, Sum, and Max pooling using the GIN baseline.
*   **Global Mean**: Test Acc = 68.75%
*   **Global Sum**: Test Acc = 65.18%
*   **Global Max**: Test Acc = 69.64%

*Analysis*: Max pooling slightly outperforms Mean pooling on this dataset, indicating that highlighting the most prominent structural feature of the protein is slightly more discriminative than taking the average structure. Sum pooling suffers, likely because it is highly sensitive to graph sizes, which vary widely in the dataset.

![Pooling Comparison](plots/pooling_comparison.png)

### 5.3 Feature Ablation
We compared GIN trained with the 3D node attributes against GIN trained with constant features of size 1 (equivalent to learning structure alone).
*   **With Node Features**: Test Acc = 68.75%
*   **Constant Features**: Test Acc = 50.89%

*Analysis*: Omitting node attributes results in a massive drop in performance to near-random guessing (50.89%). This demonstrates that secondary structure element types (helix, sheet, turn) are critical attributes for protein category prediction, and topology alone is insufficient.

![Feature Comparison](plots/features_comparison.png)

### 5.4 Regularization Ablation
We ablated GIN regularization components: Dropout (0.5), Batch Normalization (BN), and Residual Connections.
*   **None (No Regularization)**: Test Acc = 65.18%
*   **Dropout Only**: Test Acc = 71.43%
*   **BatchNorm Only**: Test Acc = 66.07%
*   **Residual Only**: Test Acc = 72.32%
*   **Full Regularization**: Test Acc = 68.75%

*Analysis*: Residual connections and dropout on their own act as strong regularizers (72.32% and 71.43% test accuracy respectively). Full regularization (BN + Residuals + Dropout) slightly over-regularizes the model on this small dataset, but outperforms having no regularization (65.18%).

![Regularization Comparison](plots/regularization_comparison.png)

### 5.5 Depth and Capacity Ablation
We grid-searched the number of GNN layers (2, 3, 5) and hidden dimensions (32, 64, 128) using GIN.

| GNN Layers | Hidden Dimension | Test Accuracy |
| :---: | :---: | :---: |
| **2** | 32 | 67.86% |
| **2** | 64 | **71.43%** |
| **2** | 128 | 68.75% |
| **3** | 32 | 68.75% |
| **3** | 64 | 68.75% |
| **3** | 128 | 70.54% |
| **5** | 32 | 67.86% |
| **5** | 64 | 66.07% |
| **5** | 128 | 65.18% |

*Analysis*: The best combination is 2 GNN layers with a hidden dimension of 64 (71.43%). Going deeper (5 layers) causes performance to degrade (e.g. 65.18% at 5 layers, 128 dims) due to over-smoothing (where node representations become increasingly similar) and overfitting.

![Depth & Capacity Comparison](plots/depth_capacity_comparison.png)

---

## 6. Final Model Evaluation
We trained our final baseline GIN model using full features, 3 layers, and 64 hidden dimensions. It achieved a test accuracy of **69.64%** (78 out of 112 graphs classified correctly).

### Classification Report
*   **Non-Enzyme (Class 0)**: Precision = 0.69, Recall = 0.80, F1-Score = 0.74 (Support = 61)
*   **Enzyme (Class 1)**: Precision = 0.71, Recall = 0.57, F1-Score = 0.63 (Support = 51)

*Analysis*: The model is better at identifying Non-Enzymes (higher recall of 0.80) than Enzymes (recall of 0.57).

![Confusion Matrix](plots/confusion_matrix.png)

---

## 7. Conclusion
In this project, we successfully implemented a robust GNN graph classification pipeline. The systematic ablations show that:
1.  **Node Features are critical**: GNNs require node semantic features to successfully classify protein graphs.
2.  **Regularization (Residuals & Dropout)**: Essential to counter overfitting on small datasets.
3.  **Model depth matters**: Shorter GNN models (2-3 layers) generalize better, whereas deeper architectures (5 layers) suffer from over-smoothing and capacity overfitting.
4.  **GIN and GCN**: Well-suited architectures for graph-level properties, providing stable training and strong generalization.
