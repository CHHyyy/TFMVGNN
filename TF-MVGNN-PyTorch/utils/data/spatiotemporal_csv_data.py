import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions

class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        batch_size: int = 32,
        seq_len: int = 12,
        pre_len: int = 3,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        normalize: bool = True,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=12)
        parser.add_argument("--pre_len", type=int, default=3)
        parser.add_argument("--train_ratio", type=float, default=0.6)
        parser.add_argument("--val_ratio", type=float, default=0.2)
        parser.add_argument("--normalize", action='store_true', default=True)
        return parser

    def setup(self, stage: str = None):
        # 使用 train_ratio 和 val_ratio 调整 generate_torch_datasets_with_test_set 的调用
        self.train_dataset, self.val_dataset, self.test_dataset = utils.data.functions.generate_torch_datasets(
            self._feat,
            self.seq_len,
            self.pre_len,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    def test_dataloader(self):
        # 新增的方法，用于获取测试数据集的 DataLoader
        return DataLoader(self.test_dataset, batch_size=len(self.test_dataset))


    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj
