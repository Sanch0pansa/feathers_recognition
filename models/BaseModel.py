import torch
import lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class BaseModel(pl.LightningModule):
    def __init__(self, get_model, num_classes, model_config: dict, *args, **kwargs):
        super().__init__()
        self.model = get_model(num_classes=num_classes, **model_config)
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        predictions = torch.argmax(y_hat, dim=1)
        acc = accuracy(predictions, y, task="multiclass", num_classes=self.num_classes)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        predictions = torch.argmax(y_hat, dim=1)
        acc = accuracy(predictions, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)