from pathlib import Path
from typing import Callable, Dict, Optional
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import ToTensor, Resize, CenterCrop
from data_handling.base import BaseDataModuleClass
from datetime import datetime
import torch.nn.functional as F
from data_handling.caching import SharedCache
    

# Load CSV root of MIMIC
MIMIC_ROOT = Path("/vol/biodata/data/chest_xray/mimic-cxr-jpg-1024/")

class MimicDataModule(BaseDataModuleClass):
    def create_datasets(self):
        self.target_size = self.config.data.augmentations.resize
        print("Train df: ")
        self.dataset_train = MimicDataset(
            csv_file=MIMIC_ROOT / Path("meta/mimic.sample.train.csv"),
            target_size=self.target_size,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
        )
        print("Validation df: ")
        self.dataset_val = MimicDataset(
            csv_file=MIMIC_ROOT / Path("meta/mimic.sample.val.csv"),
            target_size=self.target_size,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
        )
        print("Test df: ")
        self.dataset_test = MimicDataset(
            csv_file=MIMIC_ROOT / Path("meta/mimic.sample.test.csv"),
            target_size=self.target_size,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
        )

    @property
    def dataset_name(self):
        return "mimic"

class MimicDataset(Dataset):
    def __init__(
        self,
        csv_file,
        target_size, # size of imgs
        transform: Callable,
        parents=None,
        cache: bool = False,
    ):
        super().__init__()
        self.samples = {}
        df = pd.read_csv(csv_file) 
        df=df[df['disease'] != 'Other'] # We only focus on Pleuray Effusion
        print(f"{df.sex.value_counts(normalize=True)}")
        print(f"{df.disease.value_counts(normalize=True)}")
        print(f"{df.race.value_counts(normalize=True)}")
        df['disease'] = df['disease'].replace({'No Finding': 0, 'Pleural Effusion': 1})
        print(f"Len dataset {len(df)}")
        df.fillna(0, inplace=True)
        self.img_paths = df.path_preproc.values
        self.sex = df.sex_label.values
        self.age = df.age.values
        self.race = df.race_label.values
        self.finding = df.disease.values
        self.transform = transform
        self.parents = parents
        self.target_size = target_size

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[1, self.target_size[0], self.target_size[1]],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def get_samples(self):
        samples = {}
        samples['sex']= torch.tensor(self.sex).unsqueeze(-1)
        samples['age'] = torch.tensor(self.age).unsqueeze(-1) / 100.
        samples['race'] = F.one_hot(torch.tensor(self.race), num_classes=3).squeeze().float()
        samples["finding"] = torch.tensor(self.finding).unsqueeze(-1)
        return samples
        
    def read_image(self, idx):
        img = io.imread(
            Path("/vol/biodata/data/chest_xray/mimic-cxr-jpg-1024/data") / self.img_paths[idx], as_gray=True
        )
        img = img / (img.max() + 1e-12)
        img = CenterCrop(self.target_size)(Resize(self.target_size, antialias=True)(ToTensor()(img)))
        return img

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            img = self.cache.get_slot(idx)
            if img is None:
                img = self.read_image(idx)
                self.cache.set_slot(idx, img, allow_overwrite=False)
        else:
            img = self.read_image(idx)

        sample = {}
        sample["shortpath"] = str(self.img_paths[idx])
        sample["age"] = torch.tensor(self.age[idx]).unsqueeze(-1) / 100.
        sample["race"] = F.one_hot(torch.tensor(self.race[idx]), num_classes=3).squeeze().float()
        sample["sex"] = torch.tensor(self.sex[idx]).unsqueeze(-1)
        sample["finding"] = torch.tensor(self.finding[idx]).unsqueeze(-1)
        sample["x"] = self.transform(img).float() if self.transform is not None else img.float()
        # Only used for causal models
        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()

        return sample
