import os
import sys
import math
import logging
import argparse

import torch
import torch.nn as nn
import torchvision.utils
from torchvision import datasets, transforms

from muxnet import factory

#import os
#os.sys.path.append("C:\\Users\\ASUS\\AppData\\Roaming\\Python\\Python36\\site-packages\\timm\\models")
#from helpers import load_checkpoint
#from timm.models.helpers import load_checkpoint

""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='PyTorch inference')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'cifar10', 'cifar100'])
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch-size', type=int, default=200,
                    help='batch size')
parser.add_argument('--num-workers', type=int, default=2,
                    help='number of workers for data loading')
# model related
parser.add_argument('--model', type=str, default=None,
                    help='location of a json file of specific model declaration')
parser.add_argument('--pretrained', type=str, default=None,
                    help='location of initial weight to load')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.dataset == 'imagenet':
    num_classes = 1000
elif args.dataset == 'cifar100':
    num_classes = 100
else:
    num_classes = 10

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():

    logging.info("args = %s", args)

    # Data
    valid_transform = _data_transforms(args)
    valid_data = _dataset(args.data, args.dataset, valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_workers,drop_last=False)

    # Model
    print(args.model)
    net = factory(args.model, pretrained=True, num_classes=num_classes)

    #load_checkpoint(net, args.pretrained, use_ema=True)

    net = net.to(device)

    # inference
    criterion = nn.CrossEntropyLoss().to(device)
    
    #在测试集上测试模型的性能 95.92%
    #test_acc=eval_on_dataloader("test",valid_queue,10000,net,drop_last=False)
    #print("test_acc:",test_acc)#0.9592
    #下面研究模型的架构

    from torchsummary import summary
    import os
    from distill import build_adaptive_teacher
    from distill import train_activation_map_adapt_teacher

    train_data = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_workers)
    print(train_data.__len__())
    print(valid_data.__len__())
    test_loader=valid_queue
    #adaptive_teacher=build_adaptive_teacher(net)
    adaptive_teacher=torch.load("adaptive_teacher_init.pkl")
    train_activation_map_adapt_teacher(adaptive_teacher,train_loader,test_loader)
    #test_fun()
   
    #print(net)
    #summary(net,(3,32,32))
    


    os.system("pause")
    os.system("pause")
    os.system("pause")


#def infer(valid_queue, net, criterion):
#    net.eval()
#    test_loss = 0
#    correct = 0
#    total = 0

#    with torch.no_grad():
#        for step, (inputs, targets) in enumerate(valid_queue):
#            inputs, targets = inputs.to(device), targets.to(device)
#            outputs = net(inputs)
#            loss = criterion(outputs, targets)

#            test_loss += loss.item()
#            _, predicted = outputs.max(1)
#            total += targets.size(0)
#            correct += predicted.eq(targets).sum().item()

#            if step % 5 == 0:
#                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

#    acc = 100.*correct/total
#    logging.info('valid acc %f', 100. * correct / total)

#    return test_loss/total, acc


def _dataset(data, dataset, valid_transform):

    if dataset == 'cifar100':
        valid_data = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)

    elif dataset == 'cifar10':
        valid_data = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)

    elif dataset == 'imagenet':
        valid_dir = os.path.join(data, 'val')
        valid_data = datasets.ImageFolder(valid_dir, valid_transform)

    else:
        raise KeyError

    return valid_data


def _data_transforms(args):

    interpolation = 3
    if 'cifar' in args.dataset:
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]

    elif 'imagenet' in args.dataset:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    else:
        raise KeyError

    if 'imagenet' in args.dataset:
        valid_transform = transforms.Compose([
            transforms.Resize(int(math.ceil(224 / 0.875))),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        valid_transform = transforms.Compose([
            transforms.Resize(224, interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    return valid_transform


if __name__ == '__main__':
    main()