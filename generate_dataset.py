import numpy as np
from sklearn.datasets import make_blobs


# Save generated data to NPY format
def save_data_to_npy(X, y, x_filename="X.npy", y_filename="y.npy"):
    np.save(x_filename, X)
    np.save(y_filename, y)


# Generate sample data for clustering
n_samples = 1000000
n_features = 2
n_clusters = 5
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

# Save the dataset
save_data_to_npy(X, _, x_filename="X.npy", y_filename="y.npy")