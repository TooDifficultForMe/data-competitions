import torch
import torch.nn as nn
import torchvision

class Inception(nn.Module):
	def __init__(self, out_features, pretrained = False):
		super(Inception, self).__init__()
		
		inception = torchvision.models.inception_v3(pretrained)
		self.conv1 = nn.Sequential(inception.Conv2d_1a_3x3,
								   inception.Conv2d_2a_3x3,
								   inception.Conv2d_2b_3x3,
								   nn.MaxPool2d(kernel_size = 3,
											    stride = 2))
		self.conv2 = nn.Sequential(inception.Conv2d_3b_1x1,
								   inception.Conv2d_4a_3x3,
								   nn.MaxPool2d(kernel_size = 3,
											    stride = 2))
		
		self.mixed1 = nn.Sequential(inception.Mixed_5b,
								    inception.Mixed_5c,
								    inception.Mixed_5d)
		self.mixed2 = nn.Sequential(inception.Mixed_6a,
								    inception.Mixed_6c,
								    inception.Mixed_6d,
								    inception.Mixed_6e)
		self.mixed3 = nn.Sequential(inception.Mixed_7a,
								    inception.Mixed_7b,
								    inception.Mixed_7c)
		
		self.fc = nn.Linear(inception.fc.in_features, out_features)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.mixed1(x)
		x = self.mixed2(x)
		x = self.mixed3(x)
		x = nn.AvgPool2d(kernel_size = 8)(x)
		x = nn.Dropout2d()(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x