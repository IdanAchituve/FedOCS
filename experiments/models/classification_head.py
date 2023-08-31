'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=84, num_classes=10):
        # TODO: what is the embedding size in case of max and min?
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
