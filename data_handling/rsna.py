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

def norm(batch):
    for k, v in batch.items():
        if k in ['age']:
            batch[k] = batch[k].unsqueeze(0).float()
            batch[k] = batch[k] / 100.
        elif k in ['sex', 'finding']:
            batch[k] = batch[k].unsqueeze(0).float()
    return batch

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

# Please update this with your own paths.
DATA_DIR_RSNA = Path("/vol/biomedic3/mb121/rsna-pneumonia-detection-challenge")
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"
PATH_TO_PNEUMONIA_WITH_METADATA_CSV = (
    Path(__file__).parent / "pneumonia_dataset_with_metadata.csv"
)

# Root for RSNA segmentations
SEG_ROOT = Path("/vol/biomedic3/tx1215/DATA/chest_xray/rsna-segmentation") 

class RNSAPneumoniaDetectionDataset(Dataset):
    def __init__(
        self,
        config,
        target_size, # size of imgs
        df: pd.DataFrame,
        transform: Callable,
        parents = None,
        seg_target_list = None, 
        drop_id_list=None,
    ) -> None:
        """
        Torchvision dataset for loading RSNA dataset.
        Args:
            root: the data directory where the images can be found
            dataframe: the csv file mapping patient id, metadata, file names and label.
            transform: the transformation (i.e. preprocessing and / or augmentation)
            to apply to the image after loading them.

        This dataset returns a dictionary with the image data, label and metadata.
        """
        super().__init__()
        self.config = config
        self.target_size = target_size
        self.transform = transform
        self.parents = parents
        self.df = df
        self.targets = self.df.label_rsna_pneumonia.values.astype(np.int64)
        self.subject_ids = self.df.patientId.unique()
        self.filenames = [
            DATA_DIR_RSNA_PROCESSED_IMAGES / f"{subject_id}.png"
            for subject_id in self.subject_ids
        ]
        self.genders = self.df["Patient Gender"].values
        self.genders = [0 if g == 0 else 1 for g in self.genders]
        self.ages = self.df["Patient Age"].values.astype(int)
        
        self.seg_target_list = seg_target_list

    def read_image(self, idx):
        img = io.imread(self.filenames[idx], as_gray=True)
        # img = img / (img.max() + 1e-12)
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

    def __getitem__(self, index: int):
        img = self.read_image(index)

        seg_volumes = None

        if self.seg_target_list is not None:
            _segs = self.read_segs(index)
            seg_volumes = return_seg_volumes(_segs)

        sample = {
            "finding": self.targets[index],
            "sex": 1. if self.genders[index] == "M" else 0.,
            "age": self.ages[index],
        }
        # Get seg volumes
        if seg_volumes is not None:
            for k in seg_volumes.keys():
                # min, max = get_min_max_valumes(volume_name=k, target_size=self.target_size)
                _segs[k] = torch.where(_segs[k]<0.5, 0,1)
                sample[k] = _segs[k]
                # sample[f"{k}_volume"] = (seg_volumes[k]-min)/max
                sample[f"{k}_volume"] = seg_volumes[k]

        for k, v in sample.items():
            sample[k] = torch.tensor(v)

        sample = norm(sample)

        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()

        sample["x"] = img
        # print(f"self.filenames[index]: {str(self.filenames[index]).split('/')[-1]}")
        sample["shortpath"] = str(self.filenames[index]).split('/')[-1]
        return sample

    def __len__(self) -> int:
        return len(self.filenames)



class RSNAPneumoniaDataModule(BaseDataModuleClass):
    def create_datasets(self):
        """
        Pytorch Lightning DataModule defining train / val / test splits for the RSNA dataset.
        """
        self.target_size = self.config.data.augmentations.resize
        if not DATA_DIR_RSNA_PROCESSED_IMAGES.exists():
            print(
                f"Data dir: {DATA_DIR_RSNA_PROCESSED_IMAGES} does not exist."
                + " Have you updated default_paths.py?"
            )

        if not PATH_TO_PNEUMONIA_WITH_METADATA_CSV.exists():
            print(
                """
                The dataset can be found at
                https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
                This dataset is originally a (relabelled) subset of the NIH dataset
                https://www.kaggle.com/datasets/nih-chest-xrays/data from
                which i took the metadata.
                To get the full csv with all the metadata please run
                data_handling/csv_generation_code/rsna_generate_full_csv.py
                """
            )
        df_with_all_labels = pd.read_csv(PATH_TO_PNEUMONIA_WITH_METADATA_CSV)
        df_with_all_labels = df_with_all_labels.loc[
            df_with_all_labels["View Position"] == "PA"
        ]

        random_seed_for_splits = 33

        indices_train_val, indices_test = train_test_split(
            np.arange(len(df_with_all_labels)),
            test_size=0.3,
            random_state=random_seed_for_splits,
        )
        train_val_df = df_with_all_labels.iloc[indices_train_val]
        test_df = df_with_all_labels.iloc[indices_test]

        # Further split train and val
        indices_train, indices_val = train_test_split(
            np.arange(len(train_val_df)),
            test_size=0.15,
            random_state=random_seed_for_splits,
        )

        train_df = train_val_df.iloc[indices_train]
        val_df = train_val_df.iloc[indices_val]
        print(
            f"N patients train {indices_train.shape[0]}, val {indices_val.shape[0]}, test {indices_test.shape[0]}"  # noqa
        )

        # print(f"self.parents: {self.parents}")
        self.dataset_train = RNSAPneumoniaDetectionDataset(
            config=self.config,
            target_size = self.target_size,
            df=train_df,
            transform=self.train_tsfm,
            parents=self.parents,
        )
        self.dataset_val = RNSAPneumoniaDetectionDataset(
            config=self.config,
            target_size= self.target_size,
            df=val_df,
            transform=self.val_tsfm,
            parents=self.parents,
        )

        self.dataset_test = RNSAPneumoniaDetectionDataset(
            config=self.config,
            target_size=self.target_size,
            df=test_df,
            transform=self.val_tsfm,
            parents=self.parents,
        )

        print("#train: ", len(self.dataset_train))
        print("#val:   ", len(self.dataset_val))
        print("#test:  ", len(self.dataset_test))

    @property
    def num_classes(self):
        return 2