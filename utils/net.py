import torch
from torch import nn
from .inc_net import get_backbone, SimpleContinualLinear, CosineLinear
from .toolkit import count_parameters
import copy


class SimpleNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.norm = nn.LayerNorm(self.backbone.out_dim)
        self.backbone.out_dim = 768
        self.fc = None
        self._fc_type = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim
    
    def update_fc(self, num_classes, fc_type="mlp", freeze_old=True):
        if self.fc == None:
            if fc_type == "mlp":
                self.fc = SimpleContinualLinear(
                    self.feature_dim, num_classes, feat_expand=False, with_norm=False, with_bias=False
                )
            elif fc_type == "cosine":
                self.fc = CosineLinear(self.feature_dim, num_classes)
            self._fc_type = fc_type
        else:
            fc_type = fc_type if self._fc_type == None else self._fc_type
            if fc_type == "mlp":
                self.fc.update(num_classes, freeze_old=freeze_old)
            else:
                fc = CosineLinear(self.feature_dim, num_classes)
                if self.fc is not None:
                    nb_output = self.fc.out_features
                    weight = copy.deepcopy(self.fc.weight.data)
                    fc.weight.data[:nb_output] = weight
                    if self.fc.bias is not None:
                        bias = copy.deepcopy(self.fc.bias.data)
                        fc.bias.data[:nb_output] = bias
                
                del self.fc
                self.fc = fc

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
        return f"Model({trainable_params:,} / {total_params:,} = {trainable_params * 100 / total_params:.2f}), " \
               f"Backbone({backbone_trainable_params:,} / {backbone_total_params:,} = {backbone_trainable_params * 100 / backbone_total_params:.2f})"