import torch
import lightning as L
from data.FeathersImageDataModule import FeathersImageDataModule
from models import Dense121Model
from lightning.pytorch.callbacks import ModelCheckpoint


EPOCHS = 5

wandb_logger = L.pytorch.loggers.WandbLogger(
    project="Feathers-resognition",
    name="densenet121",
    log_model="all",
    config={
        "learning_rate": 0.001,
        "architecture": "densenet121",
        "dataset": "FeathersV1",
        "epochs": EPOCHS,
    }
)

dm = FeathersImageDataModule(
    "../dataset/images",
    "../dataset/data/feathers_data.csv"
)
model = Dense121Model(
    num_classes=dm.num_classes(),
    model_config={
        "save_to": "../models/weights/densenet121_weights.pt",
    }
)

checkpoint_callback = ModelCheckpoint(dirpath='model-chkp/')
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
torch.save(model.state_dict(), "../models/weights/densenet121_trained.pt")