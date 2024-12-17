
import time
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
import torch, math
import torch.fft
import glob
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# !pip install vit-pytorch
import pandas as pd
import time
import torch.nn.functional as F
import pywt
from torch.autograd import Function
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
import torch.optim as optim
# !pip install torchsummary
from torchsummary import summary
# !pip install einops
from math import ceil
import os
import copy
import torchvision.models as models
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import warnings
warnings.filterwarnings("ignore")
# helpers
from einops import reduce

from patch_loader_inference import Patchdataset
from tqdm import tqdm
# from FNET_model import FNet2D
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.cuda.empty_cache()

transform = transforms.Compose([transforms.ToTensor()]) #transforms.Resize([256, 256]), 


batch_size = 1024

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
model = convnext_tiny(weights= ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier[2] = nn.Linear(768, 2)
model_path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum1/model_wt/Convnext_tiny/18_wc_0.3cj_june/Resnet34_best_val_0.8473_iteration_3_35.pth'
epoch_itr=model_path.split('/')[-1].split('.')[1].split('_')[-4]+'_'+model_path.split('/')[-1].split('.')[1].split('_')[-2]+'_'+model_path.split('/')[-1].split('.')[1].split('_')[-1]
model.load_state_dict(torch.load(model_path),strict=False)

path_svs='/wsi_dataset/tmh/tmh_hnsc'
# path='/workspace/hnsc_for_tumor/tum_ntum/wflt_tissue_coord_csv_396_slide/TCGA-CQ-5329-01Z-00-DX1.csv'
csv_paths=glob.glob('/workspace/hnsc_for_tumor/tum_ntum/tmh_hnsc_coord_filt_256/*.csv')
# csv_paths=glob.glob('/workspace/hnsc_for_tumor/tum_ntum/wflt_tissue_coord_csv_396_slide/*.csv')
# print(csv_paths)
# exit()
for i in range (len(csv_paths)):
    csv_path=csv_paths[i]
    slide_id=csv_path.split('/')[-1].split('.')[0]
    check_csv_path_all='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum1/inference/csv/tmh_inference_res34_new'
    check_csv=csv_path.split('/')[-1]
    check_csv1=check_csv.split('.')[0]+'_convnet_8473_3_35.csv'
    print(f'check csv is {check_csv1}')
    # break;
    check_csv_path=check_csv_path_all+'/'+check_csv1
    
    if os.path.isfile(check_csv_path):
        print(f"{check_csv_path} already exists. Skipping the rest of the code.")
    else:
        print(f"{check_csv_path} does not exist. Executing the rest of the code.")
        
        path_svs1=path_svs+'/'+check_csv.split('.')[0]+'.svs'
        print(f'here is ------------------{path_svs1}')
        if os.path.isfile(path_svs1):
            print(f'file is in hpv_svs')

            val_dataset = Patchdataset(path=csv_path ,transforms=transform)
            testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

            device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

            model=model

            ##############################################################################
            # print(model)
            
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler()


            optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
            results = []  # Prepare to collect batch results

            for epoch in range(1):  # Loop over the dataset multiple times
                t0 = time.time()
                running_corrects = 0
                running_loss = 0.0

                for data in tqdm(testloader):
                    images, x, y = data[0], data[1].cpu().numpy(), data[2].cpu().numpy()
                    with torch.no_grad():
                        # t1=time.time()
                        outputs = model(images.to(device))
                        # t2=time.time()
                        # timer=t2-t1
                        # print(f'model ko time laga {timer}')
                        s_out = F.softmax(outputs, dim=1)
                        preds = torch.argmax(s_out, dim=1).cpu().numpy()

                        # Collect batch results
                        for i in range(images.size(0)):
                            probs = s_out[i].cpu().tolist()
                            prob=probs[0]
                            label_=preds[i]
                            if label_==0:

                                results.append({
                                    'dim1': x[i],
                                    'dim2': y[i],
                                    'Softmax': s_out[i].cpu().tolist(),
                                    'label': label_,
                                    'probability':prob
                                })

            # Convert collected results to DataFrame
            df = pd.DataFrame(results)

            print('Finished Testing')
            # df.to_csv('inference_result.csv', index=False)
            # df.to_csv(f'./inference/inference_csv/resnet18_wts_24ap/{slide_id}_inference_resnet18_{epoch_itr}.csv', index=False)
            df.to_csv(f'./inference/inference/csv/tmh_inference_res34_new/{slide_id}_convnet_{epoch_itr}.csv', index=False)
        else:
            pass

