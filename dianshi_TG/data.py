import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

class TGDataset(Dataset):
    """
    Arguments:
        A pandas dataframe
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv, img_path, img_ext, transform, _mlb):
    
        tmp_df = csv
        assert tmp_df[0].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
"Some images referenced in the CSV file were not found"
        self.mlb = _mlb
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df[0]
        self.y_train = self.mlb.transform(tmp_df[1].str.split()).astype(np.float32)

    def _im_read(self, path):
        img = Image.open(path)
        return img.convert('RGB')
    
    def __getitem__(self, index):
        img = self._im_read(self.img_path + self.X_train[index] + self.img_ext)
        img = self.transform(img)
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)

class data_reader(object):
    def __init__(self, args, CSV_PATH, training=True):
        self.args = args
        self.path = args.path
        self.csv_path = CSV_PATH
        self.training = training
        self.train_df, self.val_df, self.mlb = self._read_csv()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.trm = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
    def _read_csv(self):
        data_df = pd.read_csv(self.csv_path, header=None)
        train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        mlb = MultiLabelBinarizer()
        mlb.fit(data_df[1].str.split())
        return train_df, val_df, mlb
        
    def get_train_loader(self):
        assert self.training, "In training mode!"
        self.train_dataset = TGDataset(self.train_df, self.path, '', self.trm, self.mlb)
        self.val_dataset = TGDataset(self.val_df, self.path, '', self.trm, self.mlb)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.bs, shuffle=True, num_workers=1)
        return self.train_loader, self.val_loader
