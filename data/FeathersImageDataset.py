import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from json import dumps
import cv2


class FeathersImageDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 img_dir, 
                 transform=None, 
                 target_transform=None,
                 sort_indices=True,
                 ):
        self.data = pd.read_csv(annotations_file)
        self.classes_codes, self.classes_uniques = pd.factorize(self.data['species'], sort=sort_indices)
        self.data['species'] = pd.factorize(self.data['species'], sort=sort_indices)[0]

        self.imgs = pd.DataFrame(columns=('filename', 'species'))
        for idx, row in self.data.iterrows():
            self.imgs.loc[idx] = [row['filename'], row['species']]

        self.img_dir = img_dir

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return len(set(self.data['species']))

    def __getitem__(self, idx):
        filename = self.data['filename'][idx]
        filename_parts = filename.split("_")

        img_path = os.path.join(self.img_dir, filename_parts[0], "_".join(filename_parts[1:-1]), filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        w, h, _ = image.shape
        if w > h:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        label = self.data['species'][idx]

        if self.transform:
            image_tensor = self.transform(image)

        else:
            image_tensor = torch.from_numpy(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image_tensor.float(), label

    @property
    def classes(self):
        return list(self.classes_uniques)

    @property
    def class_to_idx(self):
        data = dict()
        for (i, name) in enumerate(self.classes_uniques):
            data[name] = i
        return data

    @property
    def idx_to_class(self):
        data = [""] * len(self.classes_uniques)
        for (i, name) in enumerate(self.classes_uniques):
            data[int(i)] = name
        return data


if __name__ == "__main__":
    
    transform_a = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(48, 240)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset = FeathersImageDataset("../dataset/data/feathers_data.csv",
                                   "../dataset/images",
                                   transform=transform_a, sort_indices=False)
    with open("idx-to-class_all.json", "w") as f:
        f.write(dumps(dataset.idx_to_class, indent="\t"))

    dataset = FeathersImageDataset("../dataset/data/train_top_100_species.csv",
                                   "../dataset/images",
                                   transform=transform_a, sort_indices=True)
    with open("idx-to-class_top100.json", "w") as f:
        f.write(dumps(dataset.idx_to_class, indent="\t"))
