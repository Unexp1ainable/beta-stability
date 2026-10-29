import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from beta_stability.algorithm_preparation import prepare_common

    import beta_stability.config as config_module
    from argparse import Namespace
    import logging
    from beta_stability import init_logger
    from data_utils import load_raw
    import numpy as np
    import models
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
    from lightning.pytorch.loggers.tensorboard import TensorBoardLogger


    logger = logging.getLogger("beta_stability")
    init_logger()


    # embeddings = {}
    # identities = {}
    # uuids = {}
    # for dataset_name in datasets:
    #     es, us, ids = load_raw(dataset_name)
    #     embeddings[dataset_name] = es
    #     identities[dataset_name] = ids
    #     uuids[dataset_name] = us
    return ModelCheckpoint, TensorBoardLogger, models


@app.cell
def _():
    return


@app.cell
def _():
    from data_utils import TopKDataModule

    return (TopKDataModule,)


@app.cell
def _():
    import lightning
    import pandas as pd



    return lightning, pd


@app.cell
def _():
    BATCH_SIZE = 64
    NUM_WORKERS = 64
    k = 10

    datasets = [
        "beluga",
        "forestelephants",
        "giraffe",
        "GZCD",
        "lion",
        "plainszebra",
        "whaleshark",
    ]
    return BATCH_SIZE, NUM_WORKERS, datasets, k


@app.cell
def _(
    BATCH_SIZE,
    ModelCheckpoint,
    NUM_WORKERS,
    TensorBoardLogger,
    TopKDataModule,
    datasets,
    k,
    lightning,
    models,
):

    note = f"k{k}"
    model_name = "first"

    datamodule = TopKDataModule(datasets, k, NUM_WORKERS, BATCH_SIZE)
    test_batch = next(iter(datamodule.val_dataloader()))

    model = models.RerankingModel(
        input_size=3,
        hidden_size=32,
        output_size=1,
        num_layers=2,
        batch_size=BATCH_SIZE,
        lr=0.001,
        note=note,
        k=k,
        heads=2,
        datasets=datasets
    )

    model(test_batch.x, test_batch.edge_index, test_batch.edge_attr)

    val_loss_callback = ModelCheckpoint(
        save_top_k=2,
        filename="val_loss-{epoch}-{step}-{top1_acc:.3f}",
        monitor="val_loss",
        mode="min",
        save_last=False,
    )

    top1acc_callback = ModelCheckpoint(
        save_top_k=2,
        filename="top1_acc-{epoch}-{step}-{top1_acc:.3f}",
        monitor="top1_acc",
        mode="max",
        save_last=False,
    )

    save_last_callback = ModelCheckpoint(filename="latest", save_top_k=1, mode="max", monitor="epoch")

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[1],
        log_every_n_steps=1,
        max_epochs=80,
        accumulate_grad_batches=1,
        callbacks=[
            val_loss_callback,
            save_last_callback,
            top1acc_callback,
            # StochasticWeightAveraging(swa_lrs=1e-2)
            # EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.01),
        ],
        logger=TensorBoardLogger("logs/residual", name=model_name),
        # num_sanity_val_steps=0,
        # fast_dev_run=True
    )

    ckpt_path = None
    if ckpt_path is None:
        trainer.validate(model, datamodule)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    return datamodule, top1acc_callback, trainer


@app.cell
def _(datamodule, models, pd, top1acc_callback, trainer):
    best_model = models.RerankingModel.load_from_checkpoint(top1acc_callback.best_model_path)
    trainer.test(best_model, datamodule)
    print("\n--- Baseline ---")
    trainer.test(best_model, datamodule)
    tr_model = best_model.get_test_results()
    tr_rank1 = datamodule.get_rank_1_acc_test()
    tr_model["method"]="Model"
    tr_rank1["method"]="Baseline"

    print(f"Best model: {top1acc_callback.best_model_path}")
    print(f"Alpha: {best_model.alpha}")

    res = pd.concat([tr_model, tr_rank1], axis=0)
    print(res)
    res
    return


if __name__ == "__main__":
    app.run()
