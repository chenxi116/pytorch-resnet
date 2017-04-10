import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import sys
import resnet

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]

if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    imagenet_val_dir = '/media/Work_HD/cxliu/datasets/imagenet/ILSVRC2012_img_val/'
    imagenet_val_gt = '/media/Work_HD/cxliu/tools/caffe/data/ilsvrc12/val.txt'
    lines = np.loadtxt(imagenet_val_gt, str, delimiter='\n')
    c1, c5 = 0, 0
    # model = models.resnet101(pretrained=True)
    model = getattr(resnet, 'resnet101')()
    model.eval()
    model.load_state_dict(torch.load('resnet101.pth'))

    if use_gpu:
        model = model.cuda()
    for i, line in enumerate(lines):
        [imname, label] = line.split(' ')
        label = int(label)
        im = datasets.folder.default_loader(imagenet_val_dir + imname)
        inputs = data_transforms['val'](im)
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model(inputs.unsqueeze(0))
        top5 = outputs.data.cpu().numpy().squeeze().argsort()[::-1][0:5]
        top1 = top5[0]
        if label == top1:
            c1 += 1
        if label in top5:
            c5 += 1
        print('images: %d\ttop 1: %0.4f\ttop 5: %0.4f' % (i + 1, c1/(i + 1.), c5/(i + 1.)))