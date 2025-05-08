import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv(r"D:\MAHESH\time\12841247\r4.2\r4.2\http.csv")  # Update path
df["combined_features"] = df["pc"] + " " + df["url"]

# Label assignment: mark certain URLs as malicious
df["label"] = df["url"].apply(lambda x: 1 if "xyz" in x or "malicious" in x else 0)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(max_features=1000, dtype=np.float32)
X_features = vectorizer.fit_transform(df["combined_features"])

# Encode PCs and URLs
pc_encoder = LabelEncoder()
url_encoder = LabelEncoder()
df["pc_id"] = pc_encoder.fit_transform(df["pc"])
df["url_id"] = url_encoder.fit_transform(df["url"])
pc_count = len(pc_encoder.classes_)
url_count = len(url_encoder.classes_)

# Graph construction: bipartite graph (PCs ↔ URLs)
G = nx.Graph()
G.add_nodes_from(df["pc_id"].unique(), bipartite=0)
G.add_nodes_from(df["url_id"].unique() + pc_count, bipartite=1)  # Shift URL IDs
edges = list(zip(df["pc_id"], df["url_id"] + pc_count))  # Ensure unique node IDs
G.add_edges_from(edges)

# Generate Node Features (aggregate TF-IDF vectors)
node_features = np.zeros((pc_count + url_count, X_features.shape[1]), dtype=np.float32)
for idx, row in df.iterrows():
    node_features[row["pc_id"]] += X_features[idx].toarray().squeeze()
    node_features[row["url_id"] + pc_count] += X_features[idx].toarray().squeeze()

# Convert to PyG graph
graph_data = from_networkx(G)
graph_data.x = torch.tensor(node_features, dtype=torch.float32)

# Generate node-level labels
labels = np.zeros(pc_count + url_count)
for _, row in df.iterrows():
    if row["label"] == 1:
        labels[row["pc_id"]] = 1
        labels[row["url_id"] + pc_count] = 1
graph_data.y = torch.tensor(labels, dtype=torch.long)

# Train/test masks (semi-supervised node classification)
train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
train_indices, test_indices = train_test_split(np.arange(graph_data.num_nodes), test_size=0.3, stratify=labels)
train_mask[train_indices] = True
test_mask[test_indices] = True
graph_data.train_mask = train_mask
graph_data.test_mask = test_mask

# Define GCN Model
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Model instantiation
model = GCNModel(input_dim=1000, hidden_dim=64, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Initialize lists to store metrics
epoch_losses = []
epoch_auc_scores = []

# Training loop
model.train()
for epoch in range(1, 31):
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = loss_fn(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Store the loss
    epoch_losses.append(loss.item())
    
    # Evaluate AUC-ROC on the validation set
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
        preds = logits.argmax(dim=1)
        y_true = graph_data.y[test_mask].numpy()
        y_score = logits[test_mask][:, 1].numpy()
        auc = roc_auc_score(y_true, y_score)
        
    # Store the AUC-ROC score
    epoch_auc_scores.append(auc)
    
    # Print loss for each epoch
    print(f"Epoch {epoch} - Loss: {loss.item():.4f} - AUC-ROC: {auc:.4f}")

# Evaluation after training
model.eval()
with torch.no_grad():
    logits = model(graph_data.x, graph_data.edge_index)
    preds = logits.argmax(dim=1)
    y_true = graph_data.y[test_mask].numpy()
    y_pred = preds[test_mask].numpy()
    y_score = logits[test_mask][:, 1].numpy()

# Classification Report
print("\n🔹 Classification Report:")
print(classification_report(y_true, y_pred))

# AUC-ROC Score
auc = roc_auc_score(y_true, y_score)
print(f"\n🔹 AUC-ROC Score: {auc:.4f}")

# Plot performance metrics
plt.figure(figsize=(12, 5))

# Plot Loss over Epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, 31), epoch_losses, label='Training Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Plot AUC-ROC over Epochs
plt.subplot(1, 2, 2)
plt.plot(range(1, 31), epoch_auc_scores, label='AUC-ROC', color='orange')
plt.title('AUC-ROC Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC-ROC')
plt.grid(True)

plt.tight_layout()
plt.show()
