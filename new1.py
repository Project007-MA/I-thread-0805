import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# ================================
# Data Preprocessing
# ================================
class LogDataset(Dataset):
    def __init__(self, log_data, labels, edge_index=None):
        self.log_data = log_data
        self.labels = labels
        self.edge_index = edge_index  # Optional graph structure
    
    def __len__(self):
        return len(self.log_data)
    
    def __getitem__(self, idx):
        if self.edge_index is not None:
            return torch.tensor(self.log_data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32), self.edge_index
        return torch.tensor(self.log_data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def load_data():
    # Load only device.csv data
    device_data = pd.read_csv("device.csv")
    
    # Preprocess the device data
    label_encoder = LabelEncoder()
    for col in device_data.select_dtypes(include=['object']).columns:
        device_data[col] = label_encoder.fit_transform(device_data[col])
    
    labels = device_data['label'].values  # Assuming 'label' column exists
    features = device_data.drop(columns=['label']).values  # Exclude the label column for features
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)  # Normalize the features
    
    return train_test_split(features, labels, test_size=0.2, random_state=42)

# ================================
# Time-Series Transformer with Time-Aware Attention
# ================================
class TimeAwareTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TimeAwareTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, model_dim)
    
    def forward(self, x):
        x = x.unsqueeze(0)  # Add seq_len dimension
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Take the last output of the sequence
        return x

# ================================
# Graph Neural Network (GNN)
# ================================
class GNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ================================
# Residual Hybrid Network with Attention
# ================================
class ResidualAttention(nn.Module):
    def __init__(self, input_dim):
        super(ResidualAttention, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = attn_weights * x + self.fc(x)
        return x

# ================================
# CNN with Spatial Attention
# ================================
class SpatialAttentionCNN(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttentionCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.attn = nn.Linear(64, 1)
        self.fc = nn.Linear(64, input_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Reshape for CNN (batch_size, channels=1, seq_len)
        x = self.conv(x)
        attn_weights = torch.softmax(self.attn(x), dim=1)
        x = attn_weights * x + self.fc(x)
        return x

# ================================
# Final Classification/Regression Model
# ================================
class InsiderThreatDetectionModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(InsiderThreatDetectionModel, self).__init__()
        self.time_transformer = TimeAwareTransformer(input_dim, model_dim, num_heads, num_layers)
        self.gnn = GNN(model_dim, model_dim)
        self.residual_attention = ResidualAttention(model_dim)
        self.spatial_attention_cnn = SpatialAttentionCNN(model_dim)
        self.fc = nn.Linear(model_dim, 1)
    
    def forward(self, x, edge_index):
        x_time = self.time_transformer(x)
        x_graph = self.gnn(x, edge_index)
        x_combined = self.residual_attention(x_time + x_graph)
        x_spatial = self.spatial_attention_cnn(x_combined)
        return torch.sigmoid(self.fc(x_spatial))

# ================================
# Training and Evaluation with Loss Tracking
# ================================
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    train_losses = []  # List to store loss values for each epoch
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels, edge_index in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, edge_index)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Plot the loss curve
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

# ================================
# Test Model with Accuracy Logging
# ================================
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, edge_index in test_loader:
            outputs = model(inputs, edge_index)
            predicted = (outputs > 0.5).float()
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Load Data
train_data, test_data, train_labels, test_labels = load_data()

# Edge index example: Assuming you have a method to generate an edge index (e.g., based on some graph structure)
# Replace with actual edge index logic
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # Dummy example

train_dataset = LogDataset(train_data, train_labels, edge_index=edge_index)
test_dataset = LogDataset(test_data, test_labels, edge_index=edge_index)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model Setup
input_dim = train_data.shape[1]
model_dim = 64
num_heads = 4
num_layers = 2

model = InsiderThreatDetectionModel(input_dim, model_dim, num_heads, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Train Model
train_model(model, train_loader, criterion, optimizer, epochs=5)

# Test Model
test_model(model, test_loader)
