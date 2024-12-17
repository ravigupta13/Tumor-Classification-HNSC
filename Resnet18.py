
import time
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
import torch, math
import torch.fft
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# !pip install vit-pytorch
# import albumentations as A
# from albumentations import Compose, RandomBrightnessContrast, HueSaturationValue
# from albumentations.pytorch.transforms import ToTensorV2
# from albumentations.core.transforms_interface import to_tuple, ImageOnlyTransform
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
from patch_loader_random_train_abhijeet import patchdataset
from patch_loader_random_train_abhijeet import patchdataset_test
from tqdm import tqdm
# from FNET_model import FNet2D
import tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.cuda.empty_cache()
out='./model_wt/Resnet18_new/sep12_cn_0.3cj'
if not os.path.exists(out):
    os.makedirs(out)

PATH = os.path.join(out,'Resnet18.pth')

color_jitter = transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1))

# Randomly apply the ColorJitter transformation with a probability of 0.3
random_color_jitter = transforms.RandomApply([color_jitter], p=0.30)
# Define torchvision transforms (excluding ToTensor as we will handle conversion later)
torch_transform2 = transforms.Compose([transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    transforms.Resize((256, 256)), 
    transforms.RandomRotation(degrees=135),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
torch_transform = transforms.Compose([transforms.ToTensor(),
    random_color_jitter,  # This includes ColorJitter with a probability of 0.3
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # Other transformations applied to every image   
])


torch_transform1 = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



batch_size = 1024

train_dataset = patchdataset(path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/data_256_train_fltr.csv',transforms=torch_transform)
val_dataset = patchdataset_test(path='/workspace/hnsc_for_tumor/tum_ntum/tum_ntum_new/data_256_test_fltr.csv' ,transforms=torch_transform1)


trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
# print(f' len of train loader is {len(trainloader)} and test loader is {len(testloader)}')
# for batch in testloader:
#     print(batch)

# exit()
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(f' device is {device}')
###################################### modified model ########################
# import torch.nn as nn
# import torchvision.models as models

# class ModifiedResNet34(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ModifiedResNet34, self).__init__()
#         self.model = models.resnet34(pretrained=True)
#         self.model.fc = nn.Sequential(
#             nn.Dropout(p=0.6),
#             nn.Linear(self.model.fc.in_features, num_classes)
#         )

#     def forward(self, x):
#         return self.model(x)
##############################################################################

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

# 2. Define MobileNet Model
# model = models.mobilenet_v2(pretrained=True)  # Load a pre-trained MobileNet
# model.classifier[1] = torch.nn.Linear(model.last_channel, 2)  # Adjust for number of classes in CIFAR-10
# model = models.resnet34(pretrained=True)
# model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

# model = ModifiedResNet34(num_classes=2).to(device)
# print(model)



model.to(device)

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

best_val_acc = 0.0  # Initialize the best validation accuracy

for epoch in range(500):  # loop over the dataset multiple times
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    
    for i, data in enumerate(tqdm(trainloader), start=1):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # Perform validation and potentially save the model after every 5 training iterations
        if i % 5 == 0:
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            
            with torch.no_grad():
                for val_data in testloader:
                    val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_inputs)
                    _, val_preds = torch.max(val_outputs, 1)
                    val_loss = criterion(val_outputs, val_labels)
                    
                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    val_running_corrects += torch.sum(val_preds == val_labels.data)
                
                val_loss = val_running_loss / len(testloader.dataset)
                val_acc = val_running_corrects.double() / len(testloader.dataset)
                
                # Print training and validation metrics
                train_loss = running_loss / (batch_size * i)
                train_acc = running_corrects.double() / (batch_size * i)
                print(f'Epoch {epoch+1}, Iteration {i}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
                print(f'Epoch {epoch+1}, Iteration {i}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
                
                # Check if this is the best validation accuracy so far; if so, save the model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = os.path.join(out, f'Res18_best_val_{best_val_acc:.4f}_iteration_{epoch+1}_{i}.pth')
                    torch.save(best_model_wts, best_model_path)
                    print(f'New best model saved with validation accuracy: {best_val_acc:.4f} at iteration {i} of epoch {epoch+1}')

            # Reset the model to training mode
            model.train()

# Continue with your training loop, including end-of-epoch evaluations if necessary

print('Finished Training')
