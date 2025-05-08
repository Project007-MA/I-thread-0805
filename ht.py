import pandas as pd
import re
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from urllib.parse import urlparse
from transformers import TimeSeriesTransformerModel

# Load dataset
def load_data(file_path):
    return pd.read_csv(r"D:\MAHESH\time\12841247\r4.2\r4.2\http.csv")

# Check for randomly generated domain names
def is_suspicious_domain(url):
    domain = urlparse(url).netloc
    return bool(re.search(r'\d|[^a-zA-Z0-9.-]', domain)) or len(domain.split('.')) > 3

# Check for malicious keywords
def contains_malicious_keywords(content, malicious_keywords):
    return any(keyword in content.lower() for keyword in malicious_keywords)

# Detect malicious URLs
def detect_malicious_activity(df):
    malicious_keywords = {"attack", "malware", "phishing", "exploit", "trojan", "spyware", "ransomware"}
    df['suspicious_domain'] = df['url'].apply(is_suspicious_domain)
    df['malicious_content'] = df['content'].apply(lambda x: contains_malicious_keywords(str(x), malicious_keywords))
    df['malicious'] = df['suspicious_domain'] | df['malicious_content']
    return df[df['malicious']]

# Construct Activity Graph
def build_activity_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['user'], row['pc'], timestamp=row['date'])
    return G

# Train Time-Series Transformer
def train_transformer_model(input_data):
    model = TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, torch.randint(0, 2, (output.shape[0],)))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    
    return model

