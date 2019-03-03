import torch
import torchvision
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self):
        
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 5, stride=2),  
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(32, 16, 3, stride=2),  
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2)  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 4, stride=2),  
            nn.Sigmoid(),
            nn.ConvTranspose2d(32, 8, 4, stride=4),  
            nn.Sigmoid(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1), 
            nn.Sigmoid()
        )


    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    def get_embedding(self, x):
        return self.encoder(x)
    
    def get_image_from_embedding(self, x):
        return self.decoder(x)
    
    
class Emojis(Dataset):
    def __init__(self, path="pablo/images/emojis/", transforms=None):
        self.path = path
        self.transform = transform
        
    def __getitem__(self, index):
        image = Image.open(f'{self.path}{index}.png')
        image.load()
        image = np.array(image)
        mask=image[:,:,3]==0
        image[mask] = np.array((0,0,0,0))
        
        
        if self.transform:
            image = self.transform(image)
        
        return image

    def __len__(self):
        return 2363

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 500
batch_size = 128
learning_rate = 1e-3
PATH = "pablo/images/emojis/"

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = Emojis(transforms=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

#for printing
total_steps = len(dataloader)
for epoch in range(num_epochs): 
    for i, data in enumerate(dataloader):
        
        output = model(data)
        loss = criterion(output, data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (i+1) % 9 == 0:
            print(f'epoch {epoch}/{num_epochs}, step {i+1}/{total_steps}, with loss = {loss.item()}')


torch.save(model, "first_cuda.pt")