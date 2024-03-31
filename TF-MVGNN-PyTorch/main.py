import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import models
import tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging


DATA_PATHS = {
    "pems03": {"feat": "data/los_flow.csv", "adj": "data/los_te_adj.csv"}
}


def get_model(args, dm):
    model = None
    if args.model_name == "DLGNN":
        model = models.DLGNN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task


def get_callbacks(args):
    # 确保checkpoint_callback正确配置以监控验证集损失并保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # 监控验证集损失
        mode="min",  # 最小化验证集损失
        save_top_k=1,  # 只保存最佳模型
        filename='{epoch}-{val_loss:.2f}',  # 文件名包含epoch和验证集损失
    )
    callbacks = [
        checkpoint_callback,
    ]
    return callbacks

def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        gpus=1,
        max_epochs=150
    )

    # 开始训练和验证
    trainer.fit(task, dm)
    best_model_path = callbacks[0].best_model_path  #
    best_model = task.load_from_checkpoint(best_model_path)

    # 在测试集上评估
    test_results = trainer.test(best_model, datamodule=dm)
    return test_results



def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset",  default="pems03"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        default="DLGNN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()

    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()
        if args.send_email:
            tb = traceback.format_exc()
            subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
            utils.email.send_email(tb, subject)
        exit(-1)

    if args.send_email:
        subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
        utils.email.send_experiment_results_email(args, results, subject=subject)