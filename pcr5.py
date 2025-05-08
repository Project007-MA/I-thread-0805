import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Load the dataset
file_path = "device.csv"
df = pd.read_csv(file_path, delimiter="\t")

# Simulate CPU and memory usage (random values for demonstration)
np.random.seed(42)
df["cpu_usage"] = np.random.rand(len(df)) * 100  # Simulating CPU usage in percentage
df["memory_usage"] = np.random.rand(len(df)) * 100  # Simulating Memory usage in percentage

# Normalize CPU and Memory usage
scaler = MinMaxScaler()
df[["cpu_usage", "memory_usage"]] = scaler.fit_transform(df[["cpu_usage", "memory_usage"]])

# Simulating anomaly detection (Assuming values above 0.8 are anomalies)
df["anomalous_cpu"] = df["cpu_usage"].apply(lambda x: x if x > 0.8 else np.nan)
df["anomalous_memory"] = df["memory_usage"].apply(lambda x: x if x > 0.8 else np.nan)

# Apply rolling average for smoother visualization
df["cpu_usage_smooth"] = df["cpu_usage"].rolling(window=3, min_periods=1).mean()
df["memory_usage_smooth"] = df["memory_usage"].rolling(window=3, min_periods=1).mean()

# Plot CPU Usage
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["cpu_usage_smooth"], label="System Average CPU", color="blue", linewidth=2)
plt.scatter(df.index, df["anomalous_cpu"], color="red", label="Anomalous CPU Usage", marker='o', s=50)
plt.xlabel("Time (events index)")
plt.ylabel("CPU Usage (Normalized)")
plt.title("Legitimate and Anomalous CPU Usage")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Plot Memory Usage
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["memory_usage_smooth"], label="System Average Memory", color="blue", linewidth=2)
plt.scatter(df.index, df["anomalous_memory"], color="red", label="Anomalous Memory Usage", marker='o', s=50)
plt.xlabel("Time (events index)")
plt.ylabel("Memory Usage (Normalized)")
plt.title("Legitimate and Anomalous Memory Usage")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
