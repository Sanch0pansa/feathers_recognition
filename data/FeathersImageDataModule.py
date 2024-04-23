import torch
import lightning as L
from torch.utils.data import random_split, DataLoader
from FeathersImageDataset import FeathersImageDataset
from torchvision.transforms import transforms


class FeathersImageDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_data_dir: str = "./",
                 train_data_file: str = "./feathers_data_normalized.csv",
                 test_data_dir: str | None = None,
                 test_data_file: str | None = None,
                 split_ratio: float = 0.2,
                 img_width: int = 48,
                 img_height: int = 240,
                 batch_size: int = 32):
        super().__init__()

        self.batch_size = batch_size

        self.train_data_dir = train_data_dir
        self.train_data_file = train_data_file

        self.test_data_dir = test_data_dir if test_data_dir else self.train_data_dir
        self.test_data_file = test_data_file if test_data_file else self.train_data_file

        self.split_ratio = split_ratio

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_width, img_height)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_width, img_height)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.feathers_train = None
        self.feathers_test = None
        self.feathers_val = None
        self.feathers_predict = None

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            feathers_full = FeathersImageDataset(self.train_data_file,
                                                 self.train_data_dir,
                                                 transform=self.train_transform)
            data_count = len(feathers_full)
            validation_count = int(data_count * self.split_ratio)

            self.feathers_train, self.feathers_val = random_split(
                feathers_full, 
                [data_count - validation_count, validation_count], 
                generator=torch.Generator()
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.feathers_test = FeathersImageDataset(self.test_data_file,
                                                      self.test_data_dir,
                                                      transform=self.test_transform)

        if stage == "predict":
            self.feathers_predict = FeathersImageDataset(self.train_data_file,
                                                         self.train_data_dir,
                                                         transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.feathers_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.feathers_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.feathers_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.feathers_predict, batch_size=self.batch_size)
    

if __name__ == "__main__":
    dm = FeathersImageDataModule("../dataset/images", "../dataset/data/feathers_data.csv")
    dm.setup("fit")

    dloader = dm.train_dataloader()

    clss = dict()

    for batch_index, (batch, label) in enumerate(dloader):
        print(batch_index, batch.size(), label.size())

    print(clss)