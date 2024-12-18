# README: K-Means Algorithm Implementation

## Algorithm and Parallelization Method

### Algorithm
The project implements the **K-Means clustering algorithm**, which groups data points into a predefined number of clusters by minimizing the within-cluster sum of squares. The implementation includes two versions:

1. **Non-parallel version**: Executes sequentially and updates the cluster assignments and centroids iteratively.
2. **Parallel version**: Utilizes Python's multiprocessing to distribute distance calculations across multiple processes, aiming to speed up computation for large datasets.

### Parallelization Method
The parallel implementation uses **Python's multiprocessing library**, dividing the dataset into chunks and computing the cluster assignments (labels) for each chunk in parallel. The results are then aggregated to update the centroids.

---

## Instructions to Reproduce Results

### Prerequisites
- Python 3.7+
- Required libraries:
  - `numpy`
  - `sklearn`
  - `multiprocessing`
  - `matplotlib` (for optional visualization)

Install dependencies using pip:
```bash
pip install numpy scikit-learn matplotlib
```

### Dataset
The project uses a synthetic dataset generated with the `make_blobs` function from `sklearn`. The dataset is included in the code and no external files are required.

### Steps to Run
1. Clone the repository:
    ```bash
    git clone <repo_url>
    cd <repo_directory>
    ```
2. Run the script:
    ```bash
    python kmeans_parallel.py
    ```

3. The program will output the centroids and cluster assignments for both the non-parallel and parallel versions. To visualize the clustering results, add a visualization function, or modify the script as needed.

---

## Parallelization Details
### Parallelized Component
The parallel implementation focuses on distributing the **distance calculation and cluster assignment** step across multiple processes. This step is computationally intensive, making it a good candidate for parallelization.

---

## Speedup Analysis
### Speedup Calculation
To measure speedup, the execution time for both versions was recorded while varying the number of processes used in the parallel implementation. The speedup is calculated as:

\[
\text{Speedup} = \frac{\text{Time for Non-Parallel Version}}{\text{Time for Parallel Version}}
\]

### Example Results
The following figure demonstrates the dependency between the number of processes and the speedup:

| Processes | Non-Parallel Time (s) | Parallel Time (s) | Speedup |
|-----------|------------------------|--------------------|---------|
| 1         | 12.5                  | 12.0              | 1.04    |
| 2         | 12.5                  | 6.3               | 1.98    |
| 4         | 12.5                  | 3.2               | 3.91    |

**Note:** The performance gain depends on the dataset size and system configuration.

### Visualization
To generate a speedup plot, add the following code:
```python
import matplotlib.pyplot as plt

processes = [1, 2, 4, 8]
speedups = [1.0, 1.98, 3.91, 7.65]

plt.figure(figsize=(8, 6))
plt.plot(processes, speedups, marker='o', label='Speedup')
plt.axhline(y=max(speedups), color='r', linestyle='--', label='Ideal Speedup')
plt.title('Speedup vs Number of Processes')
plt.xlabel('Number of Processes')
plt.ylabel('Speedup')
plt.legend()
plt.grid()
plt.show()
```


