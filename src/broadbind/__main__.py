import pickle
import toml
import polars as pl
from torch.optim import SGD
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from broadbind.config import Config
from broadbind.dataset import make_dataframe, split_train_val_test
from broadbind.model import GNN
from broadbind.dataset import BroadBindDataModule
from broadbind.model import BroadBindNetwork


def main():
    config = Config(**toml.load("config.toml"))
    seed_everything(config.random_seed)
    if not all(file.is_file() for file in config.paths.dict().values()):
        with open("data/reactions.pkl", "rb") as f:
            reactions = pickle.load(f)
        df = make_dataframe(
            reactions=reactions,
            bound_sites=config.bound_sites,
            properties=config.properties,
        )
        print(df.estimated_size(unit="gb"))
        train, validation, test = split_train_val_test(
            df, random_seed=config.random_seed
        )
        train.write_parquet(config.paths.train)
        validation.write_parquet(config.paths.validation)
        test.write_parquet(config.paths.test)
    n_features = len(config.properties)
    model = GNN(
        input_dimension=n_features,
        output_dimension=1,
        spatial_dimension=3,
        **config.model.dict(),
    )
    optimizer = SGD(params=model.parameters(), lr=config.training.learning_rate)
    criterion = MSELoss()

    data_module = BroadBindDataModule(
        **config.paths.dict(),
        batch_size=config.training.batch_size,
    )
    scheduler = StepLR(
        optimizer,
        step_size=config.training.decay_period,
    )
    scheduler_config = {
        "scheduler": scheduler,
        # "interval": "step",
        # "frequency": config.scheduler.step_frequency,
    }
    network = BroadBindNetwork(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler_config=scheduler_config,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoints,
        filename="{epoch}_{step}_{train_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=config.logging.checkpoint_every_n_steps,
    )
    logger = TensorBoardLogger(
        config.paths.logs,
        name="embeddings",
    )
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)
    callbacks = [
        checkpoint_callback,
        lr_logger,
        RichProgressBar(),
    ]
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip,
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=config.logging.log_every_n_steps,
        fast_dev_run=config.fast_dev_run,
    )
    trainer.fit(
        ehr2vec,
        datamodule=data_module,
        # ckpt_path="",
    )


if __name__ == "__main__":
    main()
