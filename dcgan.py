import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

generator_channels = [512, 256, 128, 64, 32, 3]
discriminator_channels = generator_channels[::-1]
latent_feature_size = 100

# channels
c1, c2, c3, c4, c5, c6 = generator_channels
# height and width of initial layer
d1 = 4


class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        # 5 layers
        
        # channels
        c1, c2, c3, c4, c5, c6 = generator_channels
        # height and width of initial layer
        d1 = 4
    
        # first dense layer
        # 100 -> 512 * 4 * 4
        self.linear1 = nn.Linear(latent_feature_size, c1 * d1 * d1)
        self.bn1 = nn.BatchNorm2d(c1)
        
        # 512 * 4 * 4 -> 256 * 8 * 8
        self.conv2 = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        
        # 256 * 8 * 8 -> 128 * 16 * 16
        self.conv3 = nn.ConvTranspose2d(c2, c3, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        
        # 128 * 16 * 16 -> 64 * 32 * 32
        self.conv4 = nn.ConvTranspose2d(c3, c4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(c4)
        
        # 64 * 32 * 32 -> 32 * 64 * 64
        self.conv5 = nn.ConvTranspose2d(c4, c5, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(c5)
        
        # 32 * 64 * 64 -> 3 * 128 * 128
        self.conv6 = nn.ConvTranspose2d(c5, c6, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(c6)
        
    def forward(self, x):
        # input layer
        x = self.linear1(x)
        self.conv1 = x.view(-1, c1, d1, d1)
        x = self.bn1(self.conv1)
        x = F.relu(x)
        
        # layer 1
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # layer 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # layer 3
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # layer 4
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # layer 5
        x = self.conv6(x)
        x = self.bn6(x)
        
        # output image
        x = F.tanh(x)
        
        return x
    
    
    
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 5 layers

        # channels
        c1, c2, c3, c4, c5, c6 = discriminator_channels
        # height and width of initial layer
        d1 = 128

        # layer 1
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)

        # layer 2
        self.conv2 = nn.Conv2d(c2, c3, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c3)

        # layer 3
        self.conv3 = nn.Conv2d(c3, c4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c4)

        # layer 4
        self.conv4 = nn.Conv2d(c4, c5, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(c5)

        # layer 5
        self.conv5 = nn.Conv2d(c5, c6, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(c6)

        self.fc = nn.Linear(c6 * 4 * 4, 1)
      
    def forward(self, x):
        self.batch_size = x.shape[0]
        
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
    
        # layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2)
        
        # output layer
        x = x.view(self.batch_size, -1)
        x = self.fc(x)

        x = F.sigmoid(x)
        
        return x
        