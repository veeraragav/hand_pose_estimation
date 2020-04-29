#!/usr/bin/env python

import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from PIL import Image
from scipy.spatial.transform import Rotation as R
import csv

csvfile = open('test/result_simple.csv', 'w')
csvwriter = csv.writer(csvfile)

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
        #self.pose_df = self.pose_df.T
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
        r = R.from_quat(pose_info[3:])
        r = r.as_euler('zyx', degrees=False)
        pose_info = pose_info[:6]
        pose_info[3:] = r
        #pose_i = pose_info[:3]
        #pose_i.append(r)
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
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 11)
        self.pool2 = nn.MaxPool2d(5, 5)

        self.fc1 = nn.Linear(12 * 45 * 61, 1500)
        self.fc2 = nn.Linear(1500, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, 20)
        self.fc6 = nn.Linear(20, 6)

    def forward(self, x):
        # -> n, 3, 480, 640
        #print('Initial: ', x.size())
        x = self.conv1(x)           # conv1-> 6, 470, 630
        #print('conv1: ', x.size())
        x = F.elu(x)
        #print('elu: ', x.size())
        x = self.pool1(x)            # pool - > 6,235,315
        #print('pool: ', x.size())
        x = self.conv2(x)           # conv -> 12, 225, 305
        #print('conv2: ', x.size())
        x = F.elu(x)
        #print('elu: ', x.size())
        x = self.pool2(x)           # pool -> 12, 45, 61
        #print('pool2: ', x.size())

        x = x.view(-1, 12 * 45 * 61)  # 32940
        #print('view(-1, 32 * 4 * 11): ', x.size())
        x = self.fc1(x)
        #print('fc1: ', x.size())
        x = F.elu(x)  # -> n, 1500
        #print('sigmoid: ', x.size())
        x = self.fc2(x)
        x = F.elu(x)
        #print('fc2: ', x.size())
        x = self.fc3(x)
        x = F.elu(x)
        #print('fc3: ', x.size())
        x = self.fc4(x)
        x = F.elu(x)
        #print('fc4: ', x.size())
        x = self.fc5(x)
        x = F.elu(x)
        #print('fc5: ', x.size())
        x = self.fc6(x)
        x = F.elu(x)
        #print('fc6: ', x.size())
        return x


model = ConvNet().to(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = HandPoseDataset(csv_file='test/demo1.csv', root_dir='test/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=30, shuffle=False, num_workers=5)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
num_epochs = 200
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#num_epochs = 100
n_total_steps = len(train_loader)


tb = SummaryWriter()

# get some random training images
dataiter = iter(train_loader)
images, y = dataiter.next()
images = images.to(device)
# create grid of images
img_grid = torchvision.utils.make_grid(images)

tb.add_image('images', img_grid)
tb.add_graph(model, images)

for epoch in range(num_epochs):
    loss_value = 0
    for i, (images, y) in enumerate(train_loader):
        images = images.to(device)
        y = y.to(device)
        y_predicted = model(images)
        #print('y_pred size: ', y_predicted.size(), 'y size: ', y.size())
        loss = criterion(y_predicted, y)
        loss_value = loss.item()
        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (i + 1) % 10 == 0:
            print('Epoch ', (epoch + 1), '/', num_epochs, ' Step ', (i + 1), '/', n_total_steps, ' Loss: ', loss.item())
    tb.add_scalar('Loss', loss_value, epoch)
    tb.add_histogram('conv1.bias', model.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', model.conv1.weight, epoch)
    tb.add_histogram('conv2.bias', model.conv2.bias, epoch)
    tb.add_histogram('conv2.weight', model.conv2.weight, epoch)
'''
print('-------------------Model Performance Sample ----------------')
inputs, ys = next(iter(train_loader))
inputs = inputs.to(device)
print('True:', ys)
print('From model: ', model(inputs))
print('------------------- ---------------------------------------')
'''
print('Evaluating on same training data ------------------')
with torch.no_grad():
    loss_array = []
    for i, (images, y) in enumerate(train_loader):
        images = images.to(device)
        y = y.to(device)
        y_predicted = model(images)
        #row = [str(s) for s in y_predicted]
        #row.insert(0, imgid)
        #csvwriter.writerow(row)
        loss = criterion(y_predicted, y)
        loss_array.append(float(loss.item()))
    avg_loss = sum(loss_array)/len(loss_array)
print('MSE: ', avg_loss)

loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=5)
print 'Writinf to csv...'
with torch.no_grad():
    loss_array = []
    for i, (images, y) in enumerate(loader):
        images = images.to(device)
        y = y.to(device)
        y_predicted = model(images)
        loss = criterion(y_predicted, y)
        loss_array.append(float(loss.item()))
        y_predicted_cpu = y_predicted.cpu()
        y_predicted_np = y_predicted_cpu.numpy()

        y_predicted_np1 = y_predicted_np[0]
        #print y_predicted_np, type(y_predicted_np)
        #print 'eulr ', y_predicted_np1[3:]
        r = R.from_euler('zyx', y_predicted_np1[3:], degrees=False)
        q = r.as_quat()
        #print 'quat ', q
        pose = np.concatenate((y_predicted_np1[:3], q))
        #print y_predicted_np1, pose, type(pose)
        imgid1 = 'frame_' + str(i*2) + '.png'
        row1 = [str(s) for s in pose]
        row1.insert(0, imgid1)
        csvwriter.writerow(row1)

        try:
            imgid2 = 'frame_' + str((i * 2) + 1) + '.png'
            y_predicted_np2 = y_predicted_np[1]
            r = R.from_euler('zyx', y_predicted_np2[3:], degrees=False)
            q = r.as_quat()
            pose = np.concatenate((y_predicted_np2[:3], q))
            row2 = [str(s) for s in pose]
            row2.insert(0, imgid2)
            csvwriter.writerow(row2)
        except:
            pass
        #print row1, row2

    avg_loss = sum(loss_array)/len(loss_array)
print('done..')
#print('MSE: ', avg_loss)
