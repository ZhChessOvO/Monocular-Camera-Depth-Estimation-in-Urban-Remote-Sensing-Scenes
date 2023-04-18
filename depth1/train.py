import argparse

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import util
import numpy as np
import sobel
from models import my_module, net, resnet, densenet, senet
import os

# 设置系统中可见的显卡，这里申请到了0,1,2,3号共四张显卡，实际使用需要根据实际情况调整
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# 显存爆炸的时候用于约束性能的设置
# max_split_size_mb = 10

# parser里的参数设置，为图方便一般直接在代码这里改
parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
# epoch
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
# epoch开始值，仅用于中断后恢复进度
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
# 学习率
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
# 以下两项参考了J Hu等人2019年的设置，未做修改
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


# 和demo中的一样，设置的主要作用是前期试错，实际训练中几乎只会用到senet部分
def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained=True)
        Encoder = my_module.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = my_module.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = my_module.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])

    return model


# 主函数
def main():
    global args
    args = parser.parse_args()

    # 设置模型基本信息
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    # 如果显卡不同，以下两项需要修改
    # 如果只有一张显卡，就不能再用dataparallel这个函数
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    batch_size = 12

    # 创建optimizer对象
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # 创建train_loader对象，这里调用到loaddata.py中内容，会加载csv中信息做好批量导入准备
    train_loader = loaddata.getTrainingData(batch_size)

    # 学习
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)

    # 保存检查点，文件名需改底下的设置
    save_checkpoint({'state_dict': model.state_dict()})


# 训练本身
def train(train_loader, model, optimizer, epoch):
    # 一些pytorch的八股部分，几乎用在哪里都不用修改
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel().cuda()

    # 用于在日志中输出耗时
    end = time.time()

    # 具体训练过程
    for i, sample_batched in enumerate(train_loader):
        # 加载图片与真值
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        # 初始化
        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()
        # 预测值
        output = model(image)
        # 根据预测值和真实值计算损失
        # 这里看起来多，但是有之前用于试错的损失函数
        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        # 这四行是梯度损失，也就是论文里做出创新的那个损失的前置条件
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
        #进一步计算
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        # depth_normal = F.normalize(depth_normal, p=2, dim=1)
        # output_normal = F.normalize(output_normal, p=2, dim=1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        # loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        # loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()

        # 最终损失
        # loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        loss = loss_depth + loss_normal
        # loss = loss_depth

        # 调优
        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        # 记录时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 更新batchsize，防止无法整除的情况
        batchSize = depth.size(0)

        # 在日志中输出信息，包括当前进度，当前损失，平均损失和当轮耗时
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
        # print(
        #     'Epoch: [{0}][{1}/{2}]\tlamda: {3}'.format(epoch, i, len(train_loader), (loss - loss_depth) / loss_normal))


# 细微调整学习率，因为别人都这么用所以加上了
# 实际上取得的效果微乎其微
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 用来给训练过程调用的简单计算
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 保存模型，实际训练时需要在这里改文件名
def save_checkpoint(state, filename='save/lamda_test.pth.tar'):
    torch.save(state, filename)


if __name__ == '__main__':
    main()
