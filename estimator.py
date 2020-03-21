#!/usr/bin/env python

import sys
import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
#print('Using device: ', device)

def image_show_with_pose(image):
    """
    Plot Image with Pose info
    """
    plt.imshow(image)
    # Logic for showing the Pose in the given image given parameters for arms
    plt.pause(0.01)
    return

class HandPoseDataset(Dataset):
    """
    Hand Pose Dataset
    """
    def __init__(self, csv_file,root_dir, transform=None):
        """
        Args
        csv_file: path to csv file
        root_dir:path to where images are stored
        tranform(optional): includes various types of image transforms
        """
        self.pose_df = pd.read_csv(csv_file, header=None)
        self.pose_df = self.pose_df.T
        #print(self.pose_df)
        self.root_dir = root_dir
        #print(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pose_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pose_df.iloc[idx, 0])
        #image = io.imread(img_name)
        image = Image.open(img_name)
        pose_info = self.pose_df.iloc[idx, 1:]
        pose_info = np.asarray([pose_info])
        pose_info = pose_info.astype('float').reshape(1, 7)
        pose_info = np.squeeze(pose_info)
        #print(type(image))
        #print(type(pose_info))
        if self.transform:
            image = self.transform(image)
            pose_info = torch.from_numpy(pose_info).float()
        sample = image, pose_info
        #print(type(image), image.size())
        #print('pose type: ', type(pose_info), 'pose size: ', pose_info.size())
        return sample


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 11)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 11)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(32 * 7 * 11, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 20)
        self.fc6 = nn.Linear(20, 7)

    def forward(self, x):
        # -> n, 3, 480, 640
        #print('Initial: ', x.size())
        x = self.conv1(x)           # conv1-> 6, 470, 630
        #print('conv1: ', x.size())
        x = F.elu(x)
        #print('elu: ', x.size())
        x = self.pool(x)            # pool - > 6,235,315
        #print('pool: ', x.size())
        x = self.conv2(x)           # conv -> 16, 225, 305
        #print('conv2: ', x.size())
        x = F.elu(x)
        #print('elu: ', x.size())
        x = self.pool2(x)           # pool -> 16, 45, 61
        #print('pool2: ', x.size())
        x = self.conv3(x)           # 32, 41, 57
        #print('conv3: ', x.size())
        x = self.conv4(x)           # 32, 39, 55
        #print('conv4: ', x.size())
        x = self.pool2(x)            # 32, 7, 11
        #print('pool2: ', x.size())
        x = x.view(-1, 32 * 7 * 11)  # 2464
        #print('view(-1, 32 * 4 * 11): ', x.size())
        x = self.fc1(x)
        #print('fc1: ', x.size())
        x = torch.sigmoid(x)  # -> n, 500
        #print('sigmoid: ', x.size())
        x = self.fc2(x)
        #print('fc2: ', x.size())
        x = self.fc3(x)
        #print('fc3: ', x.size())
        x = self.fc4(x)
        #print('fc4: ', x.size())
        x = self.fc5(x)
        #print('fc5: ', x.size())
        x = self.fc6(x)
        #print('fc6: ', x.size())
        return x


model = ConvNet().to(device)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = HandPoseDataset(csv_file='test/All_poses.csv', root_dir='test/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=5)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
num_epochs = 2
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, y) in enumerate(train_loader):
        images = images.to(device)
        y = y.to(device)
        y_predicted = model(images)
        #print('y_pred size: ', y_predicted.size(), 'y size: ', y.size())
        loss = criterion(y_predicted, y)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (i + 1) % 10 == 0:
            print('Epoch ', (epoch + 1), '/', num_epochs, ' Step ', (i + 1), '/', n_total_steps, ' Loss: ', loss.item())

print('-------------------Model Performance Sample ----------------')
inputs, ys = next(iter(train_loader))
inputs = inputs.to(device)
print('True:', ys)
print('From model: ', model(inputs))
print('------------------- ---------------------------------------')

print('Evaluating on same training data ------------------')
with torch.no_grad():
    loss_array = []
    for i, (images, y) in enumerate(train_loader):
        images = images.to(device)
        y = y.to(device)
        y_predicted = model(images)
        loss = criterion(y_predicted, y)
        loss_array.append(float(loss))
    avg_loss = sum(loss_array)/len(loss_array)
print('MSE: ', avg_loss)