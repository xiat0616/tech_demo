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
from data_handling.exclude_idx import exclude_idx

# Root for mimic images
MIMIC_ROOT = Path("/vol/biodata/data/chest_xray/mimic-cxr-jpg-1024/")

# Root for mimic segmentations
SEG_ROOT = Path("/vol/biomedic3/tx1215/DATA/chest_xray/mimic-cxr-segmentation")

# Targets for segmentations, e.g. Left-Lung
seg_targets = ["Left-Clavicle",
               "Right-Clavicle",
               "Left-Scapula",
               "Right-Scapula",
               "Left-Lung",
               "Right-Lung",
               "Left-Hilus-Pulmonis",
               "Right-Hilus-Pulmonis",
               "Heart",
               "Aorta",
               "Facies-Diaphragmatica",
               "Mediastinum",
               "Weasand",
               "Spine"]

def get_min_max_valumes(volume_name=None, target_size=(256,64)):
    _min, _max  = None, None
    if target_size==(256,64):
        if "Left-Lung" in volume_name:
            _min, _max = 303, 5931
        elif "Right-Lung" in volume_name:
            _min, _max  = 302, 5948
        elif "Heart" in volume_name:
            _min, _max  = 300, 3756
        else: 
            print(f"wrong volume name: {volume_name}")
    elif target_size==(224,224):
        if "Left-Lung" in volume_name:
            _min, _max = 934, 18153
        elif "Right-Lung" in volume_name:
            _min, _max  = 912, 18218
        elif "Heart" in volume_name:
            _min, _max  = 897, 11522
        else: 
            print(f"wrong volume name: {volume_name}")
    return  _min, _max 

def return_seg_volumes(segs, threshold=0.5):
    """
    return the volume values of each segmentation.
    """
    segs_volume = {k: 0 for k in segs.keys()}
    for target_key in segs.keys():
        _seg = segs[target_key]
        _seg = torch.where(_seg<threshold, 0,1) # Maybe discard this, as we want to be able to backpropagate gradients for segmantations.
        _seg_volume = _seg.flatten(1).sum(dim=1)
        segs_volume[target_key] = _seg_volume
    return segs_volume

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
            seg_target_list=self.config.data.seg_target_list,
            drop_id_list=exclude_idx["train"],
        )
        print("Validation df: ")
        self.dataset_val = MimicDataset(
            csv_file=MIMIC_ROOT / Path("meta/mimic.sample.val.csv"),
            target_size=self.target_size,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
            seg_target_list=self.config.data.seg_target_list,
            drop_id_list=exclude_idx["valid"],
        )
        print("Test df: ")
        self.dataset_test = MimicDataset(
            csv_file=MIMIC_ROOT / Path("meta/mimic.sample.test.csv"),
            target_size=self.target_size,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
            seg_target_list=self.config.data.seg_target_list,
            drop_id_list=exclude_idx["test"],
        )

    @property
    def dataset_name(self):
        return "mimic_seg"

class MimicDataset(Dataset):
    def __init__(
        self,
        csv_file,
        target_size, # size of imgs
        transform: Callable,
        parents=None,
        cache: bool = False,
        seg_target_list=None, # List of segmentation target to load
        drop_id_list=None,
    ):
        super().__init__()
        
        df = pd.read_csv(csv_file) 
        df=df[df['disease'] != 'Other'] # We only focus on Pleuray Effusion
        print(f"{df.sex.value_counts(normalize=True)}")
        print(f"{df.disease.value_counts(normalize=True)}")
        print(f"{df.race.value_counts(normalize=True)}")
        df['disease'] = df['disease'].replace({'No Finding': 0, 'Pleural Effusion': 1})
        print(f"Len dataset {len(df)}")
        df.fillna(0, inplace=True)
        df.reset_index(inplace=True)
        if drop_id_list is not None:
            df = df.drop(drop_id_list)
        
        self.df = df
        self.img_paths = df.path_preproc.values
        self.sex = df.sex_label.values
        self.age = df.age.values
        self.race = df.race_label.values
        self.finding = df.disease.values
        self.transform = transform
        self.parents = parents
        self.target_size = target_size
        self.seg_target_list = seg_target_list

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

    def read_image(self, idx):
        img = io.imread(
            Path("/vol/biodata/data/chest_xray/mimic-cxr-jpg-1024/data") / self.img_paths[idx], as_gray=True
        )
        img = img / (img.max() + 1e-12)
        img = CenterCrop(self.target_size)(Resize(self.target_size, antialias=True)(ToTensor()(img)))
        return img

    def read_segs(self, idx):
        segs = {target: None for target in self.seg_target_list}
        for target in self.seg_target_list:
            seg = io.imread(
                SEG_ROOT/ target/ self.img_paths[idx].split('/')[1], as_gray=True
        )
            seg = seg / (seg.max()+1e-12)
            seg = CenterCrop(self.target_size)(Resize(self.target_size, antialias=True)(ToTensor()(seg)))
            segs[target] = seg
        return segs
    
    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            img = self.cache.get_slot(idx)
            if img is None:
                img = self.read_image(idx)
                self.cache.set_slot(idx, img, allow_overwrite=False)
        else:
            img = self.read_image(idx)
        # Get seg volumes
        seg_volumes = None
        
        sample = {}

        if self.seg_target_list is not None:
            _segs = self.read_segs(idx)
            seg_volumes = return_seg_volumes(_segs)
        sample["shortpath"] = str(self.img_paths[idx])
        sample["age"] = torch.tensor(self.age[idx]).unsqueeze(-1) / 100.
        sample["race"] = F.one_hot(torch.tensor(self.race[idx]), num_classes=3).squeeze().float()
        sample["sex"] = torch.tensor(self.sex[idx]).unsqueeze(-1)
        sample["finding"] = torch.tensor(self.finding[idx]).unsqueeze(-1)
        sample["x"] = self.transform(img).float() if self.transform is not None else img.float()
        if seg_volumes is not None:
            for k in seg_volumes.keys():
                min, max = get_min_max_valumes(volume_name=k, target_size=self.target_size)
                _segs[k] = torch.where(_segs[k]<0.5, 0,1)
                sample[k] = _segs[k]
                sample[f"{k}_volume"] = (seg_volumes[k]-min)/max
                # sample[f"{k}_volume"] = seg_volumes[k]
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
