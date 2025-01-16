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
DEVICE="cuda:0"
PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
PADCHEST_IMAGES = PADCHEST_ROOT / "images"

SEG_ROOT = Path("/vol/biomedic3/tx1215/DATA/chest_xray/padchest-segmentation") # Where to save segmentations
os.makedirs(SEG_ROOT, exist_ok=True)

# %% [markdown]
# ### Load Model

# %%
model = xrv.baseline_models.chestx_det.PSPNet()
model.to(DEVICE)
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
args.hps = "padchest224_224"
args.parents_x = ["sex", "age"]
args.batch_size = 1

dataloaders = setup_dataloaders(args=args, cache=False, shuffle_train=False)

# %% [markdown]
# ### Get segmentations and save them
# %%
mininterval = 0.1
_mode="valid"
loader = tqdm(
    enumerate(dataloaders[_mode]), total=len(dataloaders[_mode]), mininterval=mininterval
)
# transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])

for i in range(len(model.targets)):
    os.makedirs(SEG_ROOT/ model.targets[i].replace(' ', '-'), exist_ok=True)

for i, batch in loader:
    img =  skimage.io.imread(Path(PADCHEST_IMAGES)/batch['shortpath'][0], as_gray=True)
    img = img / (img.max() + 1e-12) * 255
    # print(f"img max: {img.max()}, min: {img.min()}")

    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img[None, ...] # Make single color channel
    # img = transform(img)
    img = torch.from_numpy(img).to(DEVICE)
    with torch.no_grad():
        pred = model(img)
    pred = torch.sigmoid(pred)
    pred = pred *255
    # print(len(model.targets))
    # print(batch['shortpath'][0])
    for j in range(len(model.targets)):
        # print(Path(SEG_ROOT)/Path(model.targets[j].replace(' ', '-')) / batch['shortpath'][0])
        skimage.io.imsave(Path(SEG_ROOT)/Path(model.targets[j].replace(' ', '-')) / batch['shortpath'][0], pred[0,j].cpu().numpy().astype(np.uint8))

# %%



