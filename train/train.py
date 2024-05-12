import torch
import lightning as L
from data.FeathersImageDataModule import FeathersImageDataModule
from models import (
    Resnet18Model,
    Resnet34Model,
    Resnet101Model,
    Resnet152Model,
    Resnet50Model,
    Dense121Model,
    Dense161Model,
    Dense169Model,
    Dense201Model
)
from lightning.pytorch.callbacks import ModelCheckpoint

EPOCHS = 5

postfix = "_weighted"

models = [
    {
        'name': 'densenet121',
        'cls': Dense121Model,
    },
    {
        'name': 'densenet161',
        'cls': Dense161Model,
    },
    {
        'name': 'densenet169',
        'cls': Dense169Model,
    },
    {
        'name': 'densenet201',
        'cls': Dense201Model,
    },
    {
        'name': 'resnet50',
        'cls': Resnet50Model,
    },
    {
        'name': 'resnet18',
        'cls': Resnet18Model,
    },
    {
        'name': 'resnet34',
        'cls': Resnet34Model,
    },
    {
        'name': 'resnet101',
        'cls': Resnet101Model,
    },
    {
        'name': 'resnet152',
        'cls': Resnet152Model,
    },
]

for model_data in models:
    wandb_logger = L.pytorch.loggers.WandbLogger(
        project="Feathers-resognition",
        name=model_data['name'] + postfix,
        log_model="all",
        config={
            "learning_rate": 0.001,
            "architecture": model_data['name'],
            "dataset": "FeathersV1",
            "epochs": EPOCHS,
        }
    )

    dm = FeathersImageDataModule(
        "../dataset/images",
        "../dataset/data/feathers_data.csv",
        use_sampler=True
    )
    model = model_data['cls'](
        num_classes=dm.num_classes(),
        model_config={
            "save_to": f"../models/weights/{model_data['name'] + postfix}_weights.pt",
        }
    )

    checkpoint_callback = ModelCheckpoint(dirpath=f"{model_data['name'] + postfix}-chkp/")
    # early_stopping = EarlyStopping('val_loss')

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log_graph=False)
    # model.load_state_dict(torch.load(from_load))
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    trainer.fit(model, dm)
    torch.save(model.state_dict(), f"../models/weights/{model_data['name'] + postfix}_trained.pt")