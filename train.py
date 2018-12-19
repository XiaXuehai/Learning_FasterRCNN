# coding: utf-8
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.data_load import VOCData
from fr_net import fastnet

import numpy as np
import visdom
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def train():
    use_gpu = torch.cuda.is_available()

    learning_rate = 0.001
    num_epochs = 16
    batch_size = 1

    train_dataset = VOCData('data/train_2007.txt',
                            transform=[transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
                          )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    net = fastnet()
    if use_gpu:
        net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12], gamma=0.1)

    # init visdom
    iter_vis = 0
    vis = visdom.Visdom()
    win = vis.line(Y=np.array([0]), X=np.array([0]))

    for epoch in range(num_epochs):
        scheduler.step()
        losses = 0
        for i, (image, label, boxes, scale) in tqdm(enumerate(train_loader)):
            if use_gpu:
                image = image.cuda()

            loss = net(image, boxes, label, scale)
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100==0:
                vis.line(Y=np.array([losses/100]), X=np.array([iter_vis]), win=win, update='append')
                iter_vis += 1
                print('Epoch:{}, Image:{}, Loss:{:.3f}'.format(epoch, i+1, losses/100))
                losses = 0
        # save the model each epoch
        torch.save(net.state_dict(), 'weight/fastrcnn_{}.weight'.format(epoch))

if __name__ == '__main__':
    train()