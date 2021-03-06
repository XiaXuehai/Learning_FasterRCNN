# coding: utf-8

import cv2
import numpy as np
import torch

from fr_net import fastnet
from data.data_load import VOC_label_name


def detect(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (800, 600))
    img_ = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img_ = torch.from_numpy(img_/255.)
    img_ = img_.unsqueeze(0)
    bboxs, labels, scores = net.predict(img_)

    for (box,label) in zip(bboxs, labels):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        class_name = VOC_label_name[label]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_name, (x1, int(y1+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('xx', img)
    cv2.imwrite('picture/2.jpg', img)
    cv2.waitKey()

if __name__ == '__main__':
    net = fastnet()
    net.load_state_dict(torch.load('weight/fastrcnn.weight'))
    net.eval()

    img_path = 'picture/2012_003435.jpg'
    detect(img_path)


    