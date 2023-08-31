import torch.nn as nn
from experiments.models.lenet import LeNet, LeNet2
from experiments.models.mobilenetv2 import MobileNetV2, MobileNetV2Pytorch
from experiments.models.efficientnetV2 import effnetv2_s
from experiments.utils import model_load
from pathlib import Path
import torchvision
import torch
import math


class ClientModel(nn.Module):
    def __init__(self, id, embed_dim, model_path, network, **args):
        super(ClientModel, self).__init__()
        self.id = id
        if network == "lenet":
            self.model = LeNet(embed_dim)
        elif network == "lenet2":
            self.model = LeNet2(embed_dim)
        elif network == "mobilenetV2":
            self.model = MobileNetV2(embed_dim)
        elif network == "MobileNetV2Pytorch":
            self.model = MobileNetV2Pytorch(embed_dim, args['pretrained'])
        elif network == "efficientnetV2":
            self.model = effnetv2_s(embed_dim)
        if model_path:
            model_dir = Path(model_path)
            self.model = model_load(self.model, model_dir / "best_model_" + str(id) + ".pt")

    def forward(self, x):
        return self.model(x)


class ClientModelNoise(ClientModel):
    def __init__(self, id, embed_dim, model_path, network, corruption_type, corruption_severity):
        super(ClientModelNoise, self).__init__(id, embed_dim, model_path, network)
        self.corruption_type = corruption_type
        self.corruption_severity = [0.0, 0.1, 0.2, 0.4, 2 / 3, 1.0, 1.5][corruption_severity]

    def forward(self, x):
        corr_x = x + torch.randn_like(x, dtype=x.dtype).to(x.device) * self.corruption_severity
        return self.model(corr_x)


class ClientModelPatches(ClientModel):
    def __init__(self, id, embed_dim, model_path, network, grid_side_size, row_id, col_id, **args):
        super(ClientModelPatches, self).__init__(id, embed_dim, model_path, network, **args)
        self.grid_side_size = grid_side_size
        self.row_id = row_id
        self.col_id = col_id

    def forward(self, x):
        #batch_size = x.shape[0]
        side_size = x.shape[-1]
        patch_size = math.ceil(x.shape[-1] / self.grid_side_size)
        pad_dim = patch_size * self.grid_side_size - side_size
        if pad_dim > 0:
            x = torchvision.transforms.Pad((0, 0, pad_dim, pad_dim))(x)
        patch = x[:, :,
                  self.row_id * patch_size: (self.row_id + 1) * patch_size,
                  self.col_id * patch_size: (self.col_id + 1) * patch_size]
        resized_imgs = torchvision.transforms.Resize(size=side_size)(patch)
        return self.model(resized_imgs)