# coding: utf-8

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VOCData(Dataset):
    def __init__(self, file_path, transform=None):
        super(VOCData, self).__init__()

        self.img_path = []
        self.label_paths = []
        with open(file_path, 'r') as ft:
            img_lines = ft.readlines()
        for line in img_lines:
            line = line[:-1]
            self.img_path.append(line)
            label = line.replace('JPEGImages', 'labels')
            label = label.replace('jpg', 'txt')
            self.label_paths.append(label)

        self.boxes = []
        self.labels = []
        for label_path in self.label_paths:
            with open(label_path, 'r') as ft:
                lines = ft.readlines()
            label = []
            box = []
            for line in lines:
                line = line[:-1]
                splited = line.split()
                label.append(int(splited[0]))
                x, y, w, h =  float(splited[1]), float(splited[2]), float(splited[3]), float(splited[4])
                box.append([x, y, w, h])
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.tensor(label))

        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        boxes = self.boxes[idx]
        img = cv2.imread(self.img_path[idx])
        img, boxes = self.random_flip(img, boxes)
        img, boxes = self.resize(img, boxes)

        # self.showimg(img, boxes)

        for t in self.transform:
            img = t(img)

        return img, self.labels[idx], boxes

    def resize(self, image, boxes, min_size=600, max_size=1000):
        h, w, c = image.shape
        scale1 = min_size / min(h, w)
        scale2 = max_size / max(h, w)
        scale = min(scale1, scale2)

        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h)) # (w, h)?!!

        x1 = (boxes[:, 0] - 0.5 * boxes[:, 2]) * new_w
        y1 = (boxes[:, 1] - 0.5 * boxes[:, 3]) * new_h
        x2 = (boxes[:, 0] + 0.5 * boxes[:, 2]) * new_w
        y2 = (boxes[:, 1] + 0.5 * boxes[:, 3]) * new_h
        boxes = np.vstack((x1, y1, x2, y2)).transpose()
        return image, boxes

    def random_flip(self, image, boxes):
        if np.random.rand(1) < 0.5:
            im_lr = np.fliplr(image).copy()
            boxes[:,0] = 1. - boxes[:,0]
            return im_lr, boxes
        return image, boxes

    def resize_pad(self, image, boxes, ratio=(600,1000)):
        h, w, _ = image.shape
        if(h/w > ratio[0]/ratio[1]):
            # h is bigger, pad w
            pad = int(ratio[1]/ratio[0] * h - w)
            padding = ((0, 0), (pad//2, pad//2), (0, 0))
            boxes[:, 0] = (boxes[:, 0]*w + pad//2)/(ratio[1]/ratio[0] * h)
            boxes[:, 2] = boxes[:, 2] * (w/(ratio[1]/ratio[0] * h))
        else:
            pad = int(ratio[0]/ratio[1] * w- h)
            padding = ((pad//2, pad//2), (0, 0), (0, 0))
            boxes[:, 1] = (boxes[:, 1] * h + pad // 2)/(ratio[0]/ratio[1] * w)
            boxes[:, 3] = boxes[:, 3] * (h/(ratio[0]/ratio[1] * w))
        image = np.pad(image, padding, 'constant', constant_values=0)
        image = cv2.resize(image, (ratio[1], ratio[0]))

        return image, boxes

    def showimg(self, image, boxes):
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('xx', image)
        cv2.waitKey()


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

if __name__ == '__main__':
    transform = [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    a = VOCData('train.txt', transform=transform)

    image, _, _ = a[1]
    print(image.shape)
