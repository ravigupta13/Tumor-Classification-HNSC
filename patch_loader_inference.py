from PIL import Image
import torch
import pandas as pd
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader  
import os
import openslide as op 
import numpy as np

def label_tag(img_label):
    # ref={0:"LYM",1:"MUC",2:"NORM",3:"TUM",4:"ADI",5:"BACK",6:"DEB",7:"MUS",8:"STR"}
    ref={'tumor':0, 'non_tumor':1}
    return ref[img_label]

class Patchdataset:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path= '/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/tmh_hnsc_data.csv'):
        # store the inputs and outputs
        self.path=path
        # print(f'path used in dataloader is {self.path}')
        self.transforms=transforms
        self.df=pd.read_csv(self.path)

    def __len__(self):
        return len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        img_location='/wsi_dataset/tmh/tmh_hnsc'
        img_name=self.path.split('/')[-1]
        img_name=img_name.split('.')[0]+'.svs'
        img_path=os.path.join(img_location, img_name)
        # print(img_path)
        x=self.df.loc[idx,'dim1']
        y=self.df.loc[idx,'dim2']
        scaling_factor= 1.26 #target tmh have 0.2000 and source tcga have 0.2550 micrometer
        new_patch_size = int(round(patch_size * scaling_factor))
        coord=(x, y)
        # print(f'coords are {coord}')
        # print('------------------')
        wsi= op.OpenSlide(img_path)
        # print('------------------')
        level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
        level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
        level_zero_img_rgb = level_zero_img.resize((new_patch_size, new_patch_size), Image.LANCZOS)
        
        level_zero_img_rgb.save("rescaled_patch.png")
        img= np.array(level_zero_img_rgb)
        W_target = np.array([[ 0.11952107  0.58167602 -0.08860755], [ 0.84116847  0.71330381  0.48945947], [ 0.49709718  0.36627007 -0.83117183]])
        img_cn = deconvolution_based_normalization(img, W_target=W_target)
        # print(img)
        if self.transforms is not None:
            img=self.transforms(img_cn)
        return img, x, y






     
