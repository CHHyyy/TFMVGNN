import numpy as np
import pandas as pd
import torch


def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=0)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(data, seq_len, pre_len, train_ratio=0.6, val_ratio=0.2, normalize=True):
    if normalize:
        max_val = np.max(data)
        data = data / max_val

    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_X, train_Y = [], []
    val_X, val_Y = [], []
    test_X, test_Y = [], []

    # 生成训练集
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(train_data[i:i+seq_len])
        train_Y.append(train_data[i+seq_len:i+seq_len+pre_len])

    # 生成验证集
    for i in range(len(val_data) - seq_len - pre_len):
        val_X.append(val_data[i:i+seq_len])
        val_Y.append(val_data[i+seq_len:i+seq_len+pre_len])

    # 生成测试集
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(test_data[i:i+seq_len])
        test_Y.append(test_data[i+seq_len:i+seq_len+pre_len])

    return np.array(train_X), np.array(train_Y), np.array(val_X), np.array(val_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(data, seq_len, pre_len, train_ratio=0.6, val_ratio=0.2, normalize=True):
    train_X, train_Y, val_X, val_Y, test_X, test_Y = generate_dataset(
        data, seq_len, pre_len, train_ratio, val_ratio, normalize
    )

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(val_X), torch.FloatTensor(val_Y))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y))

    return train_dataset, val_dataset, test_dataset

