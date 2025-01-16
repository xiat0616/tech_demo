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
from data_handling.caching import SharedCache
# from data_handling.exclude_id_padchest import exclude_idx

PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
PADCHEST_IMAGES = PADCHEST_ROOT / "images"

# Root for PADCHEST segmentations
SEG_ROOT = Path("/vol/biomedic3/tx1215/DATA/chest_xray/padchest-segmentation")

# Targets for segmentations, e.g. Left-Lung. Maybe not necessary here
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
    if target_size==(224,224):
        if "Left-Lung" in volume_name:
            _min, _max = 2079, 15859
        elif "Right-Lung" in volume_name:
            _min, _max  = 2017, 16566
        elif "Heart" in volume_name:
            _min, _max  = 175, 10261
        else: 
            print(f"wrong volume name: {volume_name}")
    else:
        raise NotImplementedError(f"Wrong target size :{target_size}")
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

def prepare_padchest_csv():
    # Load the main PadChest CSV
    df = pd.read_csv(
        PADCHEST_ROOT / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    )
    df = df.loc[df.Projection == "PA"]
    df = df.loc[df.Pediatric == "No"]

    # Load the manually labeled CSV for filtering
    manual_labels = pd.read_csv(PADCHEST_ROOT / "manual_labelled_padchest.csv")

    # Merge the dataframes on "StudyDate_DICOM", "StudyID", "PatientID", and "ImageID"
    df = pd.merge(
        df,
        manual_labels,
        on=["StudyDate_DICOM", "StudyID", "PatientID", "ImageID"],
        how="inner",
    )

    # Filter out images marked as bad
    df = df.loc[df.bad == 0]

    def process(x, target):
        if isinstance(x, str):
            list_labels = x[1:-1].split(",")
            list_labels = [label.replace("'", "").strip() for label in list_labels]
            return target in list_labels
        else:
            return False

    for label in [
        "pneumonia",
        "exclude",
        "suboptimal study",
    ]:
        df[label] = df.Labels.astype(str).apply(lambda x: process(x, label))
    
    df = df.loc[~df.exclude]
    df = df.loc[~df["suboptimal study"]]
    
    df["Manufacturer"] = df.Manufacturer_DICOM.apply(
        lambda x: "Phillips" if x == "PhilipsMedicalSystems" else "Imaging"
    )
    df = df.loc[df["PatientSex_DICOM"].isin(["M", "F"])]
    df["PatientAge"] = (
        df.StudyDate_DICOM.apply(lambda x: datetime.strptime(str(x), "%Y%M%d").year)
        - df.PatientBirth
    )
    invalid_filenames = [
        "216840111366964013829543166512013353113303615_02-092-190.png",
        "216840111366964013962490064942014134093945580_01-178-104.png",
        "216840111366964012989926673512011151082430686_00-157-045.png",
        "216840111366964012558082906712009327122220177_00-102-064.png",
        "216840111366964012959786098432011033083840143_00-176-115.png",
        "216840111366964012373310883942009152114636712_00-102-045.png",
        "216840111366964012487858717522009280135853083_00-075-001.png",
        "216840111366964012819207061112010307142602253_04-014-084.png",
        "216840111366964012989926673512011074122523403_00-163-058.png",
        "216840111366964013590140476722013058110301622_02-056-111.png",
        "216840111366964012339356563862009072111404053_00-043-192.png",
        "216840111366964013590140476722013043111952381_02-065-198.png",
        "216840111366964012819207061112010281134410801_00-129-131.png",
        "216840111366964013686042548532013208193054515_02-026-007.png",
        "216840111366964012989926673512011083134050913_00-168-009.png"
    ]
    df = df.loc[~df.ImageID.isin(invalid_filenames)]
    return df

class PadChestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        target_size, 
        parents: Optional = None,
        seg_target_list=None, # List of segmentation target to load
        cache: bool = False,
    ): 
        super().__init__()
        df.fillna(0, inplace=True)
        df.reset_index(inplace=True)
        print(f"Len {len(df)}")
        print(df.pneumonia.value_counts(normalize=True))
        print(df.PatientSex_DICOM.value_counts(normalize=True))
        self.parents = parents
        self.finding = df.pneumonia.astype(int).values
        self.img_paths = df.ImageID.values
        self.sex = df.PatientSex_DICOM.values
        self.ages = df.PatientAge.values
        self.scanner = df.Manufacturer.values
        self.transform = transform
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
        try:
            img = io.imread(PADCHEST_IMAGES / self.img_paths[idx], as_gray=True)
        except:  # noqa
            from PIL import ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            print(self.img_paths[idx])
            img = io.imread(PADCHEST_IMAGES / self.img_paths[idx], as_gray=True)
            print("success")
            ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = img / (img.max() + 1e-12)
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img.float()
    
    # Read segmentations
    def read_segs(self, idx):
        segs = {target: None for target in self.seg_target_list}
        for target in self.seg_target_list:
            seg = io.imread(
                SEG_ROOT/ target/ self.img_paths[idx], as_gray=True
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
                self.cache.set_slot(idx, img, allow_overwrite=True)

        else:
            img = self.read_image(idx)

        # Get seg volumes
        seg_volumes = None
        sample = {}
        if self.seg_target_list is not None:
            _segs = self.read_segs(idx)
            seg_volumes = return_seg_volumes(_segs)

        sample["finding"] = self.finding[idx]
        sample["age"] = self.ages[idx] / 100
        sample["sex"] = 0 if self.sex[idx] == "M" else 1
        sample["scanner"] = 0 if self.scanner[idx] == "Phillips" else 1
        sample["shortpath"] = self.img_paths[idx]

        if seg_volumes is not None:
            for k in seg_volumes.keys():
                # min, max = get_min_max_valumes(volume_name=k, target_size=self.target_size)
                _segs[k] = torch.where(_segs[k]<0.5, 0,1)
                sample[k] = _segs[k]
                # sample[f"{k}_volume"] = (seg_volumes[k]-min)/max
                sample[f"{k}_volume"] = seg_volumes[k]

        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()

            img = self.transform(img)
            sample["x"] = img.float()

        return sample

class PadChestDataModule(BaseDataModuleClass):
    def create_datasets(self):
        self.target_size = self.config.data.augmentations.resize

        # Load the DataFrames from pre-saved CSV files in PADCHEST_ROOT
        self.train_df = pd.read_csv(PADCHEST_ROOT / "train_dataset.csv")
        self.val_df = pd.read_csv(PADCHEST_ROOT / "val_dataset.csv")
        self.test_df = pd.read_csv(PADCHEST_ROOT / "test_dataset.csv")

        self.dataset_train = PadChestDataset(
            df=self.train_df,
            target_size=self.target_size,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
            seg_target_list=self.config.data.seg_target_list,
        )

        self.dataset_val = PadChestDataset(
            df=self.val_df,
            target_size=self.target_size,
            transform=self.val_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
            seg_target_list=self.config.data.seg_target_list,
        )

        self.dataset_test = PadChestDataset(
            df=self.test_df,
            target_size=self.target_size,
            transform=self.val_tsfm,
            parents=self.parents,
            cache=True,
            seg_target_list=self.config.data.seg_target_list,
        )

# class PadChestDataModule(BaseDataModuleClass):
#     def create_datasets(self):
#         self.target_size = self.config.data.augmentations.resize
#         df = prepare_padchest_csv()
#         train_val_id, test_id = train_test_split(
#             df.PatientID.unique(),
#             test_size=0.20,
#             random_state=33,
#         )

#         train_id, val_id = train_test_split(
#             train_val_id,
#             test_size=0.10,
#             random_state=33,
#         )

#         if self.config.data.prop_train < 1.0:
#             rng = np.random.default_rng(self.config.seed)
#             train_id = rng.choice(
#                 train_id,
#                 size=int(self.config.data.prop_train * train_id.shape[0]),
#                 replace=False,
#             )

#         # Create DataFrames for train, val, and test splits
#         train_df = df.loc[df.PatientID.isin(train_id)]
#         val_df = df.loc[df.PatientID.isin(val_id)]
#         test_df = df.loc[df.PatientID.isin(test_id)]

#         # Save the DataFrames to CSV files in the current folder
#         train_df.to_csv("train_dataset.csv", index=False)
#         val_df.to_csv("val_dataset.csv", index=False)
#         test_df.to_csv("test_dataset.csv", index=False)

#         self.dataset_train = PadChestDataset(
#             df=train_df,
#             target_size=self.target_size,
#             transform=self.train_tsfm,
#             parents=self.parents,
#             cache=self.config.data.cache,
#             seg_target_list=self.config.data.seg_target_list,
#         )

#         self.dataset_val = PadChestDataset(
#             df=val_df,
#             target_size=self.target_size,
#             transform=self.val_tsfm,
#             parents=self.parents,
#             cache=self.config.data.cache,
#             seg_target_list=self.config.data.seg_target_list,
#         )

#         self.dataset_test = PadChestDataset(
#             df=test_df,
#             target_size=self.target_size,
#             transform=self.val_tsfm,
#             parents=self.parents,
#             cache=True,
#             seg_target_list=self.config.data.seg_target_list,
#         )
