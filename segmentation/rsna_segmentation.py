# %%
import sys
sys.path.append('..')
import numpy as np
import skimage
import os
os.path.abspath('')

import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv
import torchvision.transforms as tf
from pathlib import Path  
from tqdm import tqdm
from causal_models.train_setup import setup_dataloaders



# %%
# Load CSV root of MIMIC
DATA_DIR_RSNA = Path("/vol/biomedic3/mb121/rsna-pneumonia-detection-challenge")
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"
# PATH_TO_PNEUMONIA_WITH_METADATA_CSV = (
#     Path(__file__).parent / "pneumonia_dataset_with_metadata.csv"
# )
PATH_TO_PNEUMONIA_WITH_METADATA_CSV = "/vol/biomedic3/tx1215/heartflow_mimic_crop/data_handling/pneumonia_dataset_with_metadata.csv"


SEG_ROOT = Path("/vol/biomedic3/tx1215/DATA/chest_xray/rsna-segmentation") # Where to save segmentations
os.makedirs(SEG_ROOT, exist_ok=True)

# %% [markdown]
# ### Load Model

# %%
model = xrv.baseline_models.chestx_det.PSPNet()
model.eval()

for i in range(len(model.targets)):
    print(model.targets[i].replace(' ', '-'))


# %% [markdown]
# ### Data loader

# %%
class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

args = Hparams()
args.hps = "rsna224_224"
args.parents_x = ["finding", "sex", "age"]
args.batch_size = 1

dataloaders = setup_dataloaders(args=args, cache=False, shuffle_train=False)

# %% [markdown]
# ### Get segmentations and save them

# %%
mininterval = 0.1
_mode="test"
loader = tqdm(
    enumerate(dataloaders[_mode]), total=len(dataloaders[_mode]), mininterval=mininterval
)
# transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])

for i in range(len(model.targets)):
    os.makedirs(SEG_ROOT/ model.targets[i].replace(' ', '-'), exist_ok=True)

for i, batch in loader:
    img =  skimage.io.imread(Path(DATA_DIR_RSNA_PROCESSED_IMAGES)/batch['shortpath'][0], as_gray=True)
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img[None, ...] # Make single color channel
    # img = transform(img)
    img = torch.from_numpy(img)
    with torch.no_grad():
        pred = model(img)
    pred = torch.sigmoid(pred)
    pred = pred *255
    # print(len(model.targets))
    # print(batch['shortpath'][0])
    for j in range(len(model.targets)):
        # print(Path(SEG_ROOT)/Path(model.targets[j].replace(' ', '-')) / batch['shortpath'][0])
        skimage.io.imsave(Path(SEG_ROOT)/Path(model.targets[j].replace(' ', '-')) / batch['shortpath'][0], pred[0,j].numpy().astype(np.uint8))

# %%
# img = skimage.io.imread("/vol/biodata/data/chest_xray/mimic-cxr-jpg-1024/data/preproc_1024x1024/s54299881_c4ca4bc3-56adf429-80528854-dd35290f-b36bf7a6.jpg")
# img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range

# print(np.shape(img))
# img = img[None, ...] # Make single color channel

# transform = torchvision.transforms.Compose([xrv.datasets.XRayResizer(512)])

# img = transform(img)
# img = torch.from_numpy(img)

# print(img.size(), img.min(), img.max())


# %%
# seg = skimage.io.imread("/vol/biomedic3/tx1215/DATA/chest_xray/mimic-cxr-segmentation/Heart/s54299881_c4ca4bc3-56adf429-80528854-dd35290f-b36bf7a6.jpg")
# seg = seg/255.
# seg[seg<0.5] = 0.
# seg[seg>=0.5] =1.
# print(seg.min(), seg.max(), np.shape(seg))

# %%
# plt.figure(figsize = (20,10))
# plt.imshow(seg, cmap='gray')
# plt.axis('off')
# plt.tight_layout()

# %%
# # with torch.no_grad():
# pred = model(img)
# print(pred.size(), pred.min(), pred.max())

# pred = torch.sigmoid(pred)

# print(pred.size(), pred.min(), pred.max())


# %%
# plt.figure(figsize = (42,10))
# plt.subplot(1, len(model.targets) + 1, 1)
# plt.imshow(img[0], cmap='gray')
# for i in range(len(model.targets)):
#     plt.subplot(1, len(model.targets) + 1, i+2)
#     plt.imshow(pred[0, i].detach().numpy())
#     plt.title(model.targets[i])
#     plt.axis('off')
# plt.tight_layout()

# %%
# for i in range(2):
#     print(i)

# %%


# %%


# %%


# %%


# %%



