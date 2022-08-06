import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

from collections import namedtuple
from utils.torch_utils import select_device

class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h


class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x

class Bottleneck(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
            
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x

##############################################################################
class Look_Classifier():
    def __init__(self, model_path, config, device):
        self.device = select_device(device)
        ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
        resnet18_config = ResNetConfig(block = BasicBlock,n_blocks = [2,2,2,2],channels = [64, 128, 256, 512])
        resnet34_config = ResNetConfig(block = BasicBlock,n_blocks = [3,4,6,3],channels = [64, 128, 256, 512])
        resnet50_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 4, 6, 3],channels = [64, 128, 256, 512])
        resnet101_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 4, 23, 3],channels = [64, 128, 256, 512])
        resnet152_config = ResNetConfig(block = Bottleneck,n_blocks = [3, 8, 36, 3],channels = [64, 128, 256, 512])
        OUTPUT_DIM = 2
        if config == "resnet18":
            config = resnet18_config
        elif config == "resnet34":
            config = resnet34_config
        elif config == "resnet50":
            config = resnet50_config
        elif config == "resnet101":
            config = resnet101_config
        elif config == "resnet152":
            config = resnet152_config
        else:
            config = resnet50_config
        
        self.model = ResNet(config, OUTPUT_DIM)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
    
    def predict_DriverGaze(self,img,obj,ImgSize=224):
        classes = ['forward', 'other']
        [(xmin,ymin),(xmax,ymax)] = obj['bbox']
        trans = transforms.ToPILImage()
        cropped_im = trans(img[:,:,::-1])
        cropped_im = transforms.functional.crop(cropped_im,ymin,xmin,ymax-ymin,xmax-xmin)
        means = torch.from_numpy(np.array([0.5743, 0.5249, 0.5142]))
        stds = torch.from_numpy(np.array([0.2522, 0.2462, 0.2368]))
        eval_transforms = transforms.Compose([
            transforms.Resize(ImgSize),
            transforms.ToTensor(),
            transforms.Normalize(mean = means,std = stds)
            ])
        img_normalized = eval_transforms(cropped_im).float()
        img_normalized = img_normalized.unsqueeze_(0)
        img_normalized = img_normalized.to(self.device)
        with torch.no_grad():
            y_pred, _ = self.model(img_normalized)
            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)
            prob = top_pred.cpu()
        return classes[prob]
##############################################################################