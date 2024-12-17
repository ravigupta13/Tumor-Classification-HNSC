from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader  
import os
import openslide as op 
import numpy as np

def label_tag(img_label):
    ref = {'tumor': 0, 'non_tumor': 1}
    return ref[img_label]

# Placeholder function for deconvolution-based normalization
def deconvolution_based_normalization(img, W_target):
    # Implement normalization logic here
    # For demonstration, returning the image unchanged
    return img

class PatchDataset(Dataset):
    def __init__(self, transforms=transforms.ToTensor(), path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/tmh_hnsc_data.csv', cn_patch_save_dir='/workspace/CLAM_latest/w_matrix_tcga_hnsc/patches_tmh_cn_scaled'):
        # Store the path, transformations, and save directory
        self.path = path
        self.transforms = transforms
        self.df = pd.read_csv(self.path)
        self.scaling_factor = 1.26  # Scaling factor to match the pixel resolution
        self.patch_size = 256  # Base patch size
        self.cn_patch_save_dir = cn_patch_save_dir
        os.makedirs(self.cn_patch_save_dir, exist_ok=True)  # Ensure the save directory exists
        self.saved_patches = 0  # Initialize the counter for saved patches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image directory and name setup
        img_location = '/wsi_dataset/tmh/tmh_hnsc'
        img_name = self.path.split('/')[-1].split('.')[0] + '.svs'
        img_path = os.path.join(img_location, img_name)
        
        # Coordinates and patch setup
        x = int(self.df.loc[idx, 'dim1'])
        y = int(self.df.loc[idx, 'dim2'])
        new_patch_size = int(round(self.patch_size * self.scaling_factor))
        
        # Open the WSI and extract the patch
        wsi = op.OpenSlide(img_path)
        patch = wsi.read_region((x, y), 0, (self.patch_size, self.patch_size))
        
        # Convert to RGB and resize to the target scale
        patch_rgb = patch.convert('RGB')
        patch_rescaled = patch_rgb.resize((new_patch_size, new_patch_size), Image.LANCZOS)
        
        # Normalize color using the target stain matrix
        patch_np = np.array(patch_rescaled)
        W_target = np.array([[0.11952107, 0.58167602, -0.08860755], 
                             [0.84116847, 0.71330381, 0.48945947], 
                             [0.49709718, 0.36627007, -0.83117183]])
        img_cn = deconvolution_based_normalization(patch_np, W_target=W_target)
        
        # Check if img_cn is a numpy array
        if not isinstance(img_cn, np.ndarray):
            raise TypeError("deconvolution_based_normalization did not return a numpy array.")

        # Convert to uint8 if necessary
        img_cn = img_cn.astype(np.uint8) if img_cn.dtype != np.uint8 else img_cn
        
        # Convert to PIL Image
        img_cn = Image.fromarray(img_cn)

        # Resize back to 256x256
        img_cn = img_cn.resize((self.patch_size, self.patch_size), Image.LANCZOS)
        
        # Save the patch only if fewer than 100 patches have been saved
        if self.saved_patches < 100:
            save_path = os.path.join(self.cn_patch_save_dir, f"patch_{self.saved_patches}.png")
            img_cn.save(save_path)
            self.saved_patches += 1  # Increment the counter
        
        # Apply transformations if any
        if self.transforms:
            img_cn = self.transforms(img_cn)
        
        # Return the processed image and coordinates
        return img_cn, x, y
