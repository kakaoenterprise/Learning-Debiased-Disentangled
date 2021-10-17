''' Modified from https://github.com/alinlab/LfF/blob/master/module/mlp.py'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_DISENTANGLE(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP_DISENTANGLE, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(32, num_classes)

    def extract(self, x):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)
        return feat

    def predict(self, x):
        prediction = self.classifier(x)
        return prediction

    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        final_x = self.classifier(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, num_classes)


    def forward(self, x, mode=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        final_x = self.classifier(x)
        if mode == 'tsne' or mode == 'mixup':
            return x, final_x
        else:
            if return_feat:
                return final_x, feat
            else:
                return final_x

class Noise_MLP(nn.Module):
    def __init__(self, n_dim=16, n_layer=3):
        super(Noise_MLP, self).__init__()

        layers = []
        for i in range(n_layer):
            layers.append(nn.Linear(n_dim, n_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, z):
        x = self.style(z)
        return x
