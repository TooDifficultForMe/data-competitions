import torch
import torch.nn as nn
import torchvision

class Resnet(nn.Module):
	def __init__(self, out_features, pretrained = False):
		super(Resnet, self).__init__()
		
		resnet = torchvision.models.resnet101(pretrained)
		
		self.conv1 = nn.Sequential(resnet.conv1,
									resnet.bn1,
									resnet.relu,
								    resnet.maxpool)
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
		
		self.avgpool = resnet.avgpool
		self.fc = nn.Linear(2048, out_features, bias = True)
		
	def forward(self, x):
		x = self.conv1(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x