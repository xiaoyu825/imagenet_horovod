import os
import pandas as pd
from torch.utils.data.dataset import Dataset
import cv2

DATA_DIR = '../imagenet/'
IMAGE_DIR = os.path.join(DATA_DIR, 'all/')


class MetDataset(Dataset):
    def __init__(self, transforms=None, csv_file=None, state='train'):
        self.df = pd.read_csv(os.path.join(DATA_DIR, csv_file),sep=" ",header=None, names=['ImageName', 'label'])
        self.transforms = transforms
        self.state = state
        # self.cutout = Cutout(10, 30)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_path = IMAGE_DIR + self.df.loc[idx, 'ImageName']
        # image = Image.open(img_path).convert("RGB")
        # if self.transforms:
        #     image = self.transforms(image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        # if self.state == 'train':
        #     image = self.cutout(image)
        if self.state in ['train', 'val']:
            label = self.df.loc[idx, 'label']
            return image, label-1
        else:
            return image
