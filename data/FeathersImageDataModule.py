import torch
import lightning as L
from torch.utils.data import random_split, DataLoader
from data.FeathersImageDataset import FeathersImageDataset
from torchvision.transforms import transforms


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for idx, item in images.iterrows():
        count[item['species']] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, item in images.iterrows():
        weight[idx] = weight_per_class[item['species']]
    return weight


class FeathersImageDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_data_dir: str = "./",
                 train_data_file: str = "./feathers_data_normalized.csv",
                 test_data_dir: str | None = None,
                 test_data_file: str | None = None,
                 split_ratio: float = 0.2,
                 img_width: int = 224,
                 img_height: int = 224,
                 batch_size: int = 32,
                 use_sampler: bool = False,):
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

        self.train_sampler = None
        self.val_sampler = None

        self.use_sampler = use_sampler

    def num_classes(self) -> int:
        feathers_full = FeathersImageDataset(self.train_data_file,
                                             self.train_data_dir,
                                             transform=self.train_transform)

        return feathers_full.num_classes

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.feathers_train = FeathersImageDataset(self.train_data_file,
                                                       self.train_data_dir,
                                                       transform=self.train_transform)

            self.feathers_val = FeathersImageDataset(self.test_data_file,
                                                     self.test_data_dir,
                                                     transform=self.test_transform)

            if self.use_sampler:
                weights = make_weights_for_balanced_classes(self.feathers_train.imgs,
                                                            len(self.feathers_train.classes))
                weights = torch.DoubleTensor(weights)
                self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

                weights = make_weights_for_balanced_classes(self.feathers_val.imgs,
                                                            len(self.feathers_val.classes))
                weights = torch.DoubleTensor(weights)
                self.val_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

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
        if self.train_sampler:
            return DataLoader(self.feathers_train, batch_size=self.batch_size, sampler=self.train_sampler)
        return DataLoader(self.feathers_train, batch_size=self.batch_size)

    def val_dataloader(self):
        if self.val_sampler:
            return DataLoader(self.feathers_val, batch_size=self.batch_size, sampler=self.val_sampler)
        return DataLoader(self.feathers_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.feathers_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.feathers_predict, batch_size=self.batch_size)
    

if __name__ == "__main__":
    dm = FeathersImageDataModule(
        "../dataset/images",
        "../dataset/data/train_top_100_species.csv",
        "../dataset/images",
        "../dataset/data/test_top_100_species.csv",

        use_sampler=True)
    dm.setup("fit")

    dloader = dm.train_dataloader()

    clss = dict()

    for batch_index, (batch, label) in enumerate(dloader):
        print(batch_index, batch.size(), label.size())

    print(clss)