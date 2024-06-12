from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
# from model import ACM
from skimage.feature.tests.test_orb import img
from model.ACM.model_ACM import ASKCResUNet as ACM
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from model.SA_nets.SANet import SANet
from model.UCF.UCF import UCFNet
from model.AGPCNet import get_segmentation_model
class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        #pdb.set_trace()
        self.cal_loss = FocalIoULoss()#SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name=='UCF':
            self.model=UCFNet(theta_r=0, theta_0=0.7, theta_1=0, theta_2=0.7, n_blocks=7)
        elif model_name =='SANet':
            self.model=SANet(0.33,0.5)
        elif model_name=='AGPCNet':
            self.model=get_segmentation_model('agpcnet_1')
            
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        
    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
