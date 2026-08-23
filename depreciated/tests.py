import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# Set the seed for reproducibility
np.random.seed(42)

# Generate a low-dimensional dataset with strictly positive values
n_samples = 100
n_features = 10
true_rank = 5  # Low rank indicating the dimensionality of the data

# Create a random matrix with low rank
U = np.random.randn(n_samples, true_rank)
V = np.random.randn(true_rank, n_features)
low_dim_data = np.dot(U, V)

# Make the dataset strictly positive
low_dim_data += np.abs(np.min(low_dim_data)) + 1

# Convert to DataFrame for better visualization
df_low_dim_data = pd.DataFrame(low_dim_data, columns=[f'Feature_{i+1}' for i in range(n_features)])

# Perform NNMF
nmf = NMF(n_components=true_rank, random_state=42)
W = nmf.fit_transform(df_low_dim_data)
H = nmf.components_

# Reconstruct the dataset to verify
reconstructed_data = np.dot(W, H)

df_low_dim_data, pd.DataFrame(reconstructed_data, columns=[f'Feature_{i+1}' for i in range(n_features)])
