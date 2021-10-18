''' Modified from https://github.com/alinlab/LfF/blob/master/module/util.py '''

import torch.nn as nn
from module.resnet import resnet20
from module.mlp import *
from torchvision.models import resnet18, resnet50

def get_model(model_tag, num_classes):
    if model_tag == "ResNet20":
        return resnet20(num_classes)
    elif model_tag == "ResNet20_OURS":
        model = resnet20(num_classes)
        model.fc = nn.Linear(128, num_classes)
        return model
    elif model_tag == "ResNet18":
        print('bringing no pretrained resnet18 ...')
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "mlp_DISENTANGLE":
        return MLP_DISENTANGLE(num_classes=num_classes)
    elif model_tag == 'resnet_DISENTANGLE':
        print('bringing no pretrained resnet18 disentangle...')
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(1024, num_classes)
        return model
    else:
        raise NotImplementedError
