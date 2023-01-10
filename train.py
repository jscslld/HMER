from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from dataloader import CustomDataLoader
from lit_model import LitModel


def main():
    datamodule = CustomDataLoader(
        batch_size=8,
        num_workers=0,
        pin_memory=False
    )
    datamodule.setup()

    lit_model = LitModel(
        lr=1e-3,
        weight_decay=1e-4,
        milestones=[10],
        gamma=0.5,
        d_model=128,
        dim_feedforward=256,
        nhead=4,
        dropout=0.3,
        num_decoder_layers=3,
        max_output_len=200
    )

    callbacks= []
    callbacks.append(
        ModelCheckpoint(
            save_top_k=1,
            save_weights_only=True,
            mode="min",
            monitor="val/loss",
            filename="{epoch}-{val/loss:.2f}-{val/cer:.2f}"
        )
    )
    callbacks.append(
        EarlyStopping(
            patience=3,
            mode="min",
            monitor="val/loss",
            min_delta=1e-3
        )
    )
    logger = WandbLogger(project="人工智能大作业 - 手写公式识别 - test")

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=30,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger
    )

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
