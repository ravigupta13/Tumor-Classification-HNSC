from PIL import Image
import torch
import pandas as pd
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
import openslide as op 
import numpy as np
import os
import random
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization


def label_tag(img_label):
    
    ref={'no':0, 'yes':1}
    return ref[img_label]

class patchdataset:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/hpv_data_train.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)
        # print(self.df)
    def __len__(self):
        return 49152#len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        bin_label = random.randint(0, 1)
        if bin_label == 0:
            random_csv_path = self.df[self.df['label']=='no'].sample(1)['path'].tolist()[0]
            data = pd.read_csv(random_csv_path)
            # print(f'jo print karna hai uska lenth: {len(data)} and random path is {random_csv_path}')
            random_idx = np.random.randint(0, len(data))
            # print(f'random index {random_idx} and length of data {len(data)}')
            row = data.iloc[random_idx]
            coord = (row['dim1'], row['dim2'])
            img_name_csv=random_csv_path.split('/')[-1] ######################################################################
            img_name = img_name_csv.replace('_res34_new_8900_22_10.csv','.svs')
            # img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotation_g'
            img_location='/workspace/hpv_project/hpv_svs'
            # print(img_name)
            img_path=os.path.join(img_location,img_name)
            # print(f'image path in training if no {img_path}')
            wsi= op.OpenSlide(img_path)
            # print(f'this is sucessful acces of path')
            level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
            level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
            img= np.array(level_zero_img_rgb)
            W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
            # if np.var(img) > 1e-5:  # Check if the image is valid (non-empty, 3 channels)
            #     img_cn = deconvolution_based_normalization(img, W_target=W_target)
            #     if self.transforms is not None:
            #         img=self.transforms(img_cn)
            #     return img, 0
            # else:
            #     raise ValueError("Empty or invalid image encountered.")
            #     if self.transforms is not None:
            #         img=self.transforms(img_cn)
            #     return img, 0
            # # img_cn = deconvolution_based_normalization(img, W_target=W_target)
            if self.transforms is not None:
                img=self.transforms(img)
            return img, 0
        if bin_label == 1:
            
            random_csv_path = self.df[self.df['label']=='yes'].sample(1)['path'].tolist()[0]
            data = pd.read_csv(random_csv_path)
            # print(f'jo print karna hai uska lenth: {len(data)} and random path is {random_csv_path}')
            random_idx = np.random.randint(0, len(data))
            # print(f'random index {random_idx} and length of data {len(data)}')
            row = data.iloc[random_idx]
            coord = (row['dim1'], row['dim2'])
            img_name_csv=random_csv_path.split('/')[-1] 
            img_name = img_name_csv.replace('_res34_new_8900_22_10.csv','.svs')
            # print(f'img_name is {img_name}')
            # img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotation_g'
            img_location='/workspace/hpv_project/hpv_svs'
            # print(img_name)
            img_path=os.path.join(img_location,img_name)
            # print(f'image path in training data if yes {img_path}')
            wsi= op.OpenSlide(img_path)
            # print(f'this is sucessful acces of path')
            level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
            level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
            img= np.array(level_zero_img_rgb)
            W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
            # if np.var(img) > 1e-5:  # Check if the image is valid (non-empty, 3 channels)
            #     img_cn = deconvolution_based_normalization(img, W_target=W_target)
            #     if self.transforms is not None:
            #         img=self.transforms(img_cn)
            #     return img, 0
            # else:
            #     raise ValueError("Empty or invalid image encountered.")
            #     if self.transforms is not None:
            #         img=self.transforms(img_cn)
            #     return img, 0

            # img_cn = deconvolution_based_normalization(img, W_target=W_target)
            if self.transforms is not None:
                img=self.transforms(img)
            return img, 1
            

class patchdataset_test:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/hpv_data_test.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)    #['path']#.tolist()

    def __len__(self):
        return 5000#len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        # W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
        random_row = self.df.sample(1)
        random_csv_path = random_row['path'].values[0]   # Get the randomly selected path
        label = random_row['label'].values[0]
        # print(f'label is {label}')
        # random_csv_path = np.random.choice(self.df)

        data = pd.read_csv(random_csv_path)
        random_idx = np.random.randint(0, len(data))
        row = data.iloc[random_idx]
        coord = (row['dim1'], row['dim2'])
        # label = row['label']
        # print(f'label is {label}')
        # print(f'random csv in testing time is {random_csv_path}')
        # label = random_csv_path.split('/')[-2]
        img_name_csv=random_csv_path.split('/')[-1]
        img_name = img_name_csv.replace('_res34_new_8900_22_10.csv','.svs')
        img_location='/workspace/hpv_project/hpv_svs'
        # print(img_name)
        img_path=os.path.join(img_location,img_name)
        # print(f'image path in test {img_path}' )
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





     
