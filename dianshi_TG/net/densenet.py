import torch
import torch.nn as nn
import torchvision

class Densenet(nn.Module):
	def __init__(self, out_features, pretrained = False):
		super(Densenet, self).__init__()
		
		densenet = torchvision.models.densenet201(pretrained)
		self.get_features = densenet.features
		self.fc = nn.Linear(densenet.classifier.in_features, out_features, bias = True)
		
	def forward(self, x):
		return nn.Sequential(self.get_features,
							 self.relu,
							 nn.AvgPool2d(kernel_size = 7,
										  stride = 1),
							 self.fc)(x)