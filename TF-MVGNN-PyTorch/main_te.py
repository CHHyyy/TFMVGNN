from transferEntropy import transfer_entropy  # 明确导入所需函数
import pandas as pd
import numpy as np
from tqdm import tqdm


def transfer_entropy_batchwise(source, target, k, batch_size=1440, local=False):
    assert k > 0, "History length k must be positive"
    assert batch_size > k, "Batch size must be greater than history length k"

    num_batches = int(np.ceil(len(source) / batch_size))
    te_values = []

    for batch in range(num_batches):
        start = batch * batch_size
        end = min(start + batch_size, len(source))
        if end - start > k:
            try:
                te = transfer_entropy(source[start:end], target[start:end], k, local=local)
                te_values.append(te)
            except Exception as e:
                print(f"Error calculating TE from batch {batch}: {e}")

    return np.mean(te_values) if te_values else np.nan


def compute_te_for_series_batchwise(data, num_series, k=1, batch_size=1440, local=False):
    te_matrix = np.zeros((num_series, num_series))

    for i in range(num_series):
        for j in tqdm(range(num_series), desc=f"Node {i}"):
            if i != j:
                te_matrix[i, j] = transfer_entropy_batchwise(data[:, i], data[:, j], k, batch_size, local)

    return te_matrix


data = pd.read_csv("data/los_flow.csv", header=None, skiprows=1).values
n_nodes = data.shape[1]
te_matrix = compute_te_for_series_batchwise(data, n_nodes, k=1, batch_size=1440)

output_file_path = 'transfer_entropy_matrix.csv'
pd.DataFrame(te_matrix).to_csv(output_file_path, index=False)
print(f"Transfer entropy matrix saved to {output_file_path}")
