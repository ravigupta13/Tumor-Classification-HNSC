import pandas as pd
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


def get_neg_patch(df, curr_neg_classes, transforms):
    is_found=True
    while is_found:
        random_csv_path = df[df['label']=='non_tumor'].sample(1)['path'].tolist()[0]
        # print(f'this is going here -------------------------------------------------')
        data = pd.read_csv(random_csv_path)
        # print(curr_neg_classes)
        data = data[data['class']==curr_neg_classes]
        # if data.empty:break
        # print(f'data is available {len(data)}')
        if len(data)>0: is_found= False
    

    # print(f'lenth od data {len(data)}')
    random_idx = np.random.randint(0, len(data))
    row = data.iloc[random_idx]
    coord = (row['dim1'], row['dim2'])
    img_name_csv=random_csv_path.split('/')[-1]
    img_name = img_name_csv.replace('csv','svs')
    img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotation_g'
    img_path=os.path.join(img_location,img_name)
    # print(f'image path in non_tumor {img_path}')
    wsi= op.OpenSlide(img_path)
    level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
    level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
    img= np.array(level_zero_img_rgb)
    W_target = np.array([[ 0.11952107  0.58167602 -0.08860755], [ 0.84116847  0.71330381  0.48945947], [ 0.49709718  0.36627007 -0.83117183]])
    img_cn = deconvolution_based_normalization(img, W_target=W_target)
    if transforms is not None:
        img=transforms(img_cn)
    return img, 1


def label_tag(img_label):
    
    ref={'tumor':0, 'non_tumor':1}
    return ref[img_label]

class patchdataset:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/data_256_train_fltr.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)
        # print(self.df)
    def __len__(self):
        return 49152#len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        W_target = np.array([[ 0.11952107  0.58167602 -0.08860755], [ 0.84116847  0.71330381  0.48945947], [ 0.49709718  0.36627007 -0.83117183]])
        bin_label = random.randint(0, 1)
        if bin_label == 0:
            # print(f'we are here')
            random_csv_path = self.df[self.df['label']=='tumor'].sample(1)['path'].tolist()[0]
            data = pd.read_csv(random_csv_path)
            # print(f'jo print karna hai uska lenth: {len(data)} and random path is {random_csv_path}')
            random_idx = np.random.randint(0, len(data))
            # print(f'random index {random_idx} and length of data {len(data)}')
            row = data.iloc[random_idx]
            coord = (row['dim1'], row['dim2'])
            img_name_csv=random_csv_path.split('/')[-1]
            img_name = img_name_csv.replace('csv','svs')
            # print(f'img_name is {img_name}')
            img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotation_g'
            # print(img_name)
            img_path=os.path.join(img_location,img_name)
            # print(f'img path in train for tumor {img_path}')
            #reading coordinates as patch image
            wsi= op.OpenSlide(img_path)
            # print(f'this is sucessful acces of path')
            level_zero_img= wsi.read_region(coord, 0, (256,256)) #rgba image
            level_zero_img_rgb=level_zero_img.convert('RGB')  #convert to rgb
            img= np.array(level_zero_img_rgb)
            img_cn = deconvolution_based_normalization(img, W_target=W_target)
            if self.transforms is not None:
                img=self.transforms(img_cn)
            return img, 0
        if bin_label == 1:
            # print(f'else we are here')
            neg_classes = ['hfu', 'nep', 'ts', 'adipose', 'ct', 'nerve', 'dermis', 'sweat', 'sebaceous', 'sg', 'muscle', 'necrosis',  'lymph', 'inf', 'subepi', 'blood', 'granluation']
            #'skin-epidermis', 'Dysplastic squamous epithelium', 'bv'
            curr_neg_classes = random.choice(neg_classes)
            # print(f'choosed current class is : {curr_neg_classes}')
            neg_patch=get_neg_patch(self.df, curr_neg_classes, self.transforms)
            # print(f' neg_patch is {neg_patch}')
            return get_neg_patch(self.df, curr_neg_classes, self.transforms)
            

class patchdataset_test:
    #Load the Dataset
    def __init__(self,transforms=transforms.ToTensor(),path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/data_256_test_fltr.csv'):
        # store the inputs and outputs
        self.path=path
        self.transforms=transforms
        self.df=pd.read_csv(self.path)['path'].tolist()

    def __len__(self):
        return 5000#len(self.df)

    # get a row at an index
    def __getitem__(self, idx):
        # W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
        random_csv_path = np.random.choice(self.df)
        data = pd.read_csv(random_csv_path)
        random_idx = np.random.randint(0, len(data))
        row = data.iloc[random_idx]
        coord = (row['dim1'], row['dim2'])
        label = random_csv_path.split('/')[-2]
        img_name_csv=random_csv_path.split('/')[-1]
        img_name = img_name_csv.replace('csv','svs')
        img_location='/workspace/hnsc_for_tumor/tum_ntum/manual_annotation_g'
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





     
