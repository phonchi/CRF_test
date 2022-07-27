import torchvision
import torch.nn as nn
import torch
from collections import OrderedDict

from convcrf import GaussCRF

class crf(nn.Module):
    
    def __init__(self, backbone, config, num_classes=21, shape=(480, 480), use_gpu=False, fullscaleFeat=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.shape = shape
        self.backbone = backbone
        self.use_gpu = use_gpu

        self.config = config
        
        self.use_feat = False
        if fullscaleFeat is not None:
            self.use_feat = True
        
        self.gausscrf = GaussCRF(conf=self.config, shape=self.shape, nclasses=self.num_classes,
                    use_gpu= self.use_gpu, fullscaleFeat = self.use_feat)
        
    def forward(self, x):
        
        crf_output = OrderedDict()
         
        unary = self.backbone(x)['out']
        
        fullscaleFeat = None
        if self.use_feat:
            fullscaleFeat = unary
            
        crf_output['backbone'] = unary
        crf_output['out'] = self.gausscrf(unary, x, fullscaleFeat=fullscaleFeat)
        
        return crf_output

def create_model(pretrain_path=None, num_classes=21, drop_rate=0.5):

    model = torchvision.models.segmentation.fcn_resnet101()
    model.classifier[3].p = drop_rate

    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

    return model

def create_crf_model(pretrain_path, config, num_classes=21, drop_rate=0.5, freeze_backbone=True, fullscaleFeat=None):
    backbone = create_model(pretrain_path, num_classes, drop_rate)
    
    if freeze_backbone:
        for params in backbone.parameters():
            params.requires_grad = False
            
    model = crf(backbone, config, fullscaleFeat=fullscaleFeat)

    return model
