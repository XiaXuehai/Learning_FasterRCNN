# coding: utf-8
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from data.data_load import VOCData
from fr_net import fastnet

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def train():
    use_gpu = torch.cuda.is_available()

    learning_rate = 0.001
    num_epochs = 1
    batch_size = 1

    train_dataset = VOCData('data/train.txt',
                            transform=[transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
                          )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = fastnet()
    if use_gpu:
        net = net.cuda()
    opimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    for epoch in range(num_epochs):
        losses = 0
        for i, (image, label, boxes) in tqdm(enumerate(train_loader)):
            if use_gpu:
                image = image.cuda()
                boxes = boxes.cuda()
                label = label.cuda()

            loss = net(image, boxes, label)
            losses += loss.item()

            opimizer.zero_grad()
            loss.backward()
            opimizer.step()

            if (i+1)%100==0:
                print('Epoch:{}, Image:{}, Loss:{:.3f}'.format(epoch, i+1, losses/100))
                losses = 0

        torch.save(net.state_dict(), 'weight/fastrcnn_1.weight')

if __name__ == '__main__':
    train()