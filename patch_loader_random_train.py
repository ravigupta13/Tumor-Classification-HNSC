from PIL import Image
import torch
import pandas as pd
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
import openslide as op 
import numpy as np
import os
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization

def label_tag(img_label):
    
    ref={'tumor':0, 'non_tumor':1}
    return ref[img_label]

class patchdataset:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum1/data_train.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)['path'].tolist()

    def __len__(self):
        return 131072#len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
        random_csv_path = np.random.choice(self.df)
        data = pd.read_csv(random_csv_path)
        random_idx = np.random.randint(0, len(data))
        row = data.iloc[random_idx]
        coord = (row['dim1'], row['dim2'])
        label = random_csv_path.split('/')[-2]
        img_name_csv=random_csv_path.split('/')[-1]
        img_name = img_name_csv.replace('csv','svs')
        img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotations'
        # print(img_name)
        img_path=os.path.join(img_location,img_name)

        label=torch.tensor(label_tag(label))
        #reading coordinates as patch image
        wsi= op.OpenSlide(img_path)
        level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
        level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
        img= np.array(level_zero_img_rgb)
        # img_cn = deconvolution_based_normalization(img, W_target=W_target)
        if self.transforms is not None:
            img=self.transforms(img)
        return img,label

class patchdataset_test:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/home1/ravi/Desktop/hpv_project/tum_ntum/data_train.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)['path'].tolist()

    def __len__(self):
        return 8192#len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
        random_csv_path = np.random.choice(self.df)
        data = pd.read_csv(random_csv_path)
        random_idx = np.random.randint(0, len(data))
        row = data.iloc[random_idx]
        coord = (row['dim1'], row['dim2'])
        label = random_csv_path.split('/')[-2]
        img_name_csv=random_csv_path.split('/')[-1]
        img_name = img_name_csv.replace('csv','svs')
        img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotations'
        # print(img_name)
        img_path=os.path.join(img_location,img_name)

        label=torch.tensor(label_tag(label))
        #reading coordinates as patch image
        wsi= op.OpenSlide(img_path)
        level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
        level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
        img= np.array(level_zero_img_rgb)
        # img_cn = deconvolution_based_normalization(img, W_target=W_target)
        if self.transforms is not None:
            img=self.transforms(img)
        return img,label





     
