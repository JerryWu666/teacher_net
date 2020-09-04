

EPOCHES=50

import torch.nn as nn
import torch
#ResNet的BasicBlock
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,\
        downsample=None, groups=1,
                 base_width=64, dilation=1, \
                     norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(\
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(\
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#用于查看网络架构
from torchsummary import summary


#过渡模块，用于特征图尺寸和通道数的匹配
class TransitionModule(nn.Module):
    def __init__(self, input_channels, output_channels,input_size,output_size):
#利用预期输入输出特征图的通道数和尺寸构造过渡块
        super(TransitionModule, self).__init__()
        self.layer=None
        if not (input_size%output_size==0 or \
            output_size%input_size==0):
            raise Exception("Activation Map Size Mismatch!")
        #输入输出特征特征图的尺寸不成整数倍关系则抛出异常
        if input_size>=output_size:
            stride=input_size//output_size
            self.layer=nn.Sequential(
                nn.Conv2d(input_channels,output_channels,\
                    kernel_size=5,stride=stride,padding=2),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
                )
            #输入尺寸大于输出尺寸时，利用大卷积核、大步长卷积进行下采样
        else:
            k=output_size//input_size
            self.layer=nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels,\
                    out_channels=output_channels,
                kernel_size=k,padding=0,\
                    stride=k,output_padding=0),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
                )
            #输入尺寸小于输出尺寸时，利用反卷积进行上采样
    def forward(self, x):
        return self.layer(x)

#特征图自适应模块，用于特征图尺寸、通道数的匹配；
#特征的同步变换，语义信息的自适应匹配
class ActivationMapAdaptiveModule(nn.Module):
    def __init__(self, input_channels, \
        output_channels,input_size,output_size):
        super(ActivationMapAdaptiveModule, self).__init__()
        self.require_grad=True
        self.layer=None
        if input_size>=output_size:
            self.layer=nn.Sequential(
                TransitionModule(input_channels,\
                    output_channels,input_size,output_size),
                #尺寸和通道数的匹配
                BasicBlock(output_channels,output_channels),
                #特征的同步变换，语义信息的自适应匹配，下同
                )
        else:
            self.layer=nn.Sequential(
                BasicBlock(input_channels,input_channels),
                TransitionModule(input_channels,\
                    output_channels,input_size,output_size),
                )

    def forward(self, x):
        return self.layer(x)





from torch import nn

def get_modules(net,module_name_list):
    seq=nn.Sequential()
    for layer,name in zip(net.children(),net.named_children()):
        print(type(name),name)
        if name[0] in module_name_list:
            seq.add_module(name[0],layer)    
    return seq

from torchsummary import summary

def build_adaptive_teacher(teacher_net):
    print("teacher_net")
    summary(teacher_net,(3,32,32))


    root=get_modules(teacher_net,['conv_stem'])
    blocks=get_modules(teacher_net,['blocks'])
    head=get_modules(teacher_net,['conv_head','bn2','global_pool'])
    classifier=get_modules(teacher_net,['classifier'])
    pre_block=get_modules(blocks[0],['0','1','2','3','4','5','6','7','8','9'])
    post_block=get_modules(blocks[0],['12','13','14','15','16','17'])
    mid_block=get_modules(blocks[0],['10','11'])
    T_front=nn.Sequential(
        root,
        pre_block,
        )
    T_back=nn.Sequential(
        post_block,
        head,
        nn.Flatten(),
        classifier
        )
    print("front")
    print(T_front)
    summary(T_front,(3,32,32))#80*2*2
    print("middle")
    print(mid_block)
    summary(mid_block,(80,2,2))#112*2*2
    print("back")
    print(T_back)
    summary(T_back,(112,2,2))#80*2*2

    #STUDENT_SIZE=(64,2,2)#假设学生网络输出监督的尺寸为64*8*8
    pre_adaptor=ActivationMapAdaptiveModule(80,64,8,8).cuda()
    post_adaptor=ActivationMapAdaptiveModule(64,112,8,8).cuda()
    adaptive_teacher=nn.Sequential(
        T_front,
        pre_adaptor,
        post_adaptor,
        T_back,
        )
    print("adaptive teacher")
    print(adaptive_teacher)
    summary(adaptive_teacher,(3,32,32))

    return adaptive_teacher
    



from torchvision.datasets import CIFAR10
import torch.utils.data as Data
import torch
import torch as t
import numpy as np
from torchvision import transforms
import random

def train_and_eval(loss_fn,optim,net,train_loader,test_loader):
    train_accs,test_accs=[],[]
    for epoch in range(EPOCHES):
        net.train()#切换到训练模式
        
        for step, data in enumerate(train_loader, start=0):
            
            images, labels = data
            images,labels=images.cuda(),labels.cuda()

            optim.zero_grad()#将优化器的梯度清零
            logits = net(images)#网络推断的输出
            #print(logits.shape,labels.long())
            loss = loss_fn(logits, labels.long())#计算损失函数
            loss.backward()#反向传播求梯度
            optim.step()#优化器进一步优化

            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
        print()

        net.eval()#切换到测试模式

        train_acc=eval_on_dataloader("train",train_loader,50000,net)
        test_acc=eval_on_dataloader("test",test_loader,10000,net)
        print("epoch:",epoch,"train_acc:",train_acc," test_acc:",test_acc)
        torch.save(net,"adaptive_teacher"+"epoch "+str(epoch)+" "+
                       str(int(train_acc*10000)/10000.)+";"+str(int(test_acc*10000)/10000.)+".pkl")
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
    return train_accs,test_accs    

def train_activation_map_adapt_teacher(adaptive_teacher,train_loader,test_loader):
    #adaptive_teacher=torch.load(xxx)

    for layer,name in zip(adaptive_teacher.children(),adaptive_teacher.named_children()):
        #print(name[0],layer)
        if name[0] not in ['1','2']:
            layer.require_grad=False
        print(name[0],layer.require_grad)

    loss_fn = nn.CrossEntropyLoss()
    #optim = torch.optim.SGD(filter(lambda p: p.requires_grad, adaptive_teacher.parameters()),
    #                        lr = 0.01,weight_decay=3e-4)
    optim= torch.optim.Adam(filter(lambda p: p.requires_grad, adaptive_teacher.parameters()),
                            lr = 0.01,weight_decay=3e-4)

    train_accs,test_accs=train_and_eval(loss_fn,optim,adaptive_teacher,train_loader,test_loader)


def eval_on_dataloader(name,loader,len,net):
    acc = 0.0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images,labels=images.cuda(),labels.cuda()
            outputs = net(images)
            predict_y = torch.max(outputs, dim=1)[1]#torch.max返回两个数值，一个是最大值，一个是最大值的下标
            acc += (predict_y == labels).sum().item()
        accurate = acc / len
        return accurate

def test_fun():
    print("HelloWorld")