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
    return (models,)


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
def _(BATCH_SIZE, NUM_WORKERS, TopKDataModule, datasets, k, lightning):

    note = f"k{k}"
    model_name = "first"

    datamodule = TopKDataModule(datasets, k, NUM_WORKERS, BATCH_SIZE)

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[1],
    )
    return datamodule, trainer


@app.cell
def _(datamodule, models, pd, trainer):
    model_path = "logs/reranking_transfer/checkpoints/top1_acc-epoch=41-step=8610-top1_acc=0.731.ckpt"
    best_model = models.RerankingModel.load_from_checkpoint(model_path)
    trainer.test(best_model, datamodule)
    print("\n--- Baseline ---")
    trainer.test(best_model, datamodule)
    tr_model = best_model.get_test_results()
    tr_rank1 = datamodule.get_rank_1_acc_test()
    diff = tr_model - tr_rank1
    tr_model["method"]="Model"
    tr_rank1["method"]="Baseline"

    print(f"Best model: {model_path}")
    print(f"Alpha: {best_model.alpha}")

    res = pd.concat([tr_model, tr_rank1], axis=0)
    print(res)
    res
    return (diff,)


@app.cell
def _(diff):
    diff
    return


if __name__ == "__main__":
    app.run()
