import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool


# 1. Non-parallel version of k-means
def kmeans(X, n_clusters, max_iter=300, tol=1e-4):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return centroids, labels


# 2. Parallel version of k-means
def parallel_compute_labels(args):
    X_chunk, centroids = args
    distances = np.linalg.norm(X_chunk[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def kmeans_parallel(X, n_clusters, max_iter=300, tol=1e-4, n_processes=4):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    pool = Pool(n_processes)
    chunk_size = X.shape[0] // n_processes

    for _ in range(max_iter):
        chunks = [(X[i:i + chunk_size], centroids) for i in range(0, X.shape[0], chunk_size)]
        labels = np.concatenate(pool.map(parallel_compute_labels, chunks))

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    pool.close()
    pool.join()
    return centroids, labels


# 3. Data load
def load_data_from_npy(x_filename="X.npy", y_filename="y.npy"):
    X = np.load(x_filename)
    y = np.load(y_filename)
    return X, y


# 4. Benchmarking and plotting
def benchmark_kmeans(X, n_clusters):
    workers = [1, 2, 4, 8, 16]
    non_parallel_times = []
    parallel_times = []

    # Measure time for the non-parallel version
    start_time = time()
    kmeans(X, n_clusters)
    non_parallel_times.append(time() - start_time)

    # Measure time for the parallel version with varying workers
    for n_processes in workers:
        start_time = time()
        kmeans_parallel(X, n_clusters, n_processes=n_processes)
        parallel_times.append(time() - start_time)

    # Plot results
    plt.figure(figsize=(10, 6))
    # plt.plot([1] + workers, non_parallel_times + parallel_times, label="Non-Parallel", marker='o')
    plt.plot(workers, parallel_times, label="Parallel", marker='o')
    plt.axhline(y=non_parallel_times[0], color='r', linestyle='--', label='Non-Parallel')
    plt.title("Execution Time Comparison: Non-Parallel vs Parallel K-Means")
    plt.xlabel("Number of Workers (Processes)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid()

    # Save plot as an image
    plt.savefig("kmeans_speedup_comparison.png")

    plt.show()


if __name__ == '__main__':
    # Load dataset
    X, _ = load_data_from_npy()
    n_clusters = len(np.unique(_))

    # Benchmark
    benchmark_kmeans(X, n_clusters)
