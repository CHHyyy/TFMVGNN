import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses


class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes = x.size()
        # (batch_size, num_nodes, hidden_dim)
        hidden = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        hidden = hidden.reshape((-1, hidden.size(2)))
        # (batch_size * num_nodes, pre_len)
        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = x.size(2)
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val  # 标准化预测值
        y = y * self.feat_max_val  # 标准化真实值
        # 计算损失
        loss = self.loss(predictions, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        # 可选：返回预测和真实值，便于进一步分析
        return {'predictions': predictions, 'y': y}

    def test_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions * self.feat_max_val  # 如果数据进行了缩放，进行反向缩放
        y = y * self.feat_max_val

        # 计算指标
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)

        # 记录指标
        self.log('test_rmse', rmse, on_epoch=True, prog_bar=True)
        self.log('test_mae', mae, on_epoch=True, prog_bar=True)
        self.log('test_r2', r2, on_epoch=True, prog_bar=True)
        self.log('test_var', explained_variance, on_epoch=True, prog_bar=True)

        # 返回结果用于后续分析或 `test_epoch_end` 汇总
        return {'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2, 'test_var': explained_variance}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=0)
        parser.add_argument("--loss", type=str, default="mse")
        return parser