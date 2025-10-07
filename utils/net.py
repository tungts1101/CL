import torch
from torch import nn
from .inc_net import get_backbone, SimpleContinualLinear
from .toolkit import count_parameters


class SimpleNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.norm = nn.LayerNorm(self.backbone.out_dim)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim
    
    def update_fc(self, num_classes, freeze_old=True):
        if self.fc == None:
            self.fc = SimpleContinualLinear(
                self.feature_dim, num_classes, feat_expand=False, with_norm=False, with_bias=False
            )
        else:
            self.fc.update(num_classes, freeze_old=freeze_old)

    def get_backbone_trainable_params(self):
        params = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                params[name] = param
        return params

    def get_features(self, x):
        z = self.backbone(x)
        if self.norm != None:
            z = self.norm(z)
        return z

    def extract_vector(self, x):
        return self.get_features(x)

    def forward(self, x):
        z = self.get_features(x)
        y = self.fc(z)
        return y

    def __repr__(self):
        trainable_params = count_parameters(self, trainable=True)
        total_params = count_parameters(self)
        backbone_trainable_params = count_parameters(self.backbone, trainable=True)
        backbone_total_params = count_parameters(self.backbone)
        return f"Model(trainable_params={trainable_params:,}, total_params={total_params:,}, percentage={trainable_params * 100 / total_params:.2f})\n" \
               f"Backbone: trainable_params={backbone_trainable_params:,}, total_params={backbone_total_params:,}, percentage={backbone_trainable_params * 100 / backbone_total_params:.2f}"