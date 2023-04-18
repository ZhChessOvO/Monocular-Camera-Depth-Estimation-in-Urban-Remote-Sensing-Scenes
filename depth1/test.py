import argparse
import torch
import torch.nn as nn
import torch.nn.parallel

from models import my_module, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel
import os

# 设置系统中可见的显卡，这里申请到了0,1,2,3号共四张显卡，实际使用需要根据实际情况调整
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    # 加载模型
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    # 因为用到多个显卡合作，保存的时候默认值会多一层state_dict，所以这里需要多读一层state_dict
    # 如果只有一个显卡，中括号应当去掉
    # 具体应不应该去其实挺复杂的，反正如果读不出来一开始就会报错，根据提示调整即可
    model.load_state_dict(torch.load('save1/our_model.pth.tar')['state_dict'])

    test_loader = loaddata.getTestingData(1)
    test(test_loader, model, 0.25)


# 并非所有数都有用，有一些是试错的时候用来找问题的，几乎不影响性能
def test(test_loader, model, thre):
    model.eval()

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    # 改自train.py，基本逻辑是一样的，只不过这个在测出误差之后只记录数据，不进行优化
    for i, sample_batched in enumerate(test_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        depth = depth.cuda()
        image = image.cuda()

        image = torch.autograd.Variable(image, volatile=True)
        depth = torch.autograd.Variable(depth, volatile=True)

        output = model(image)
        output = torch.nn.functional.upsample(output, size=[depth.size(2), depth.size(3)], mode='bilinear')

        depth_edge = edge_detection(depth)
        output_edge = edge_detection(output)

        batchSize = depth.size(0)
        totalNumber = totalNumber + batchSize
        errors = util.evaluateError(output, depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)

        edge1_valid = (depth_edge > thre)
        edge2_valid = (output_edge > thre)

        nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
        A = nvalid / (depth.size(2) * depth.size(3))

        nvalid2 = np.sum(((edge1_valid + edge2_valid) == 2).float().data.cpu().numpy())
        P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
        # if np.sum(edge1_valid.data.cpu().numpy()) == 0:
        #     R = 0.5
        # else:
        R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))
        F = (2 * P * R) / (P + R)

        Ae += A
        Pe += P
        Re += R
        Fe += F

        # print(averageError)

    Av = Ae / totalNumber
    Pv = Pe / totalNumber
    Rv = Re / totalNumber
    Fv = Fe / totalNumber
    print('PV', Pv)
    print('RV', Rv)
    print('FV', Fv)

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    print(averageError)


# 同demo和train
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


# 用于在测量边缘误差的时候调整大小，为图方便把计算也放进来了
# 虽然这么放耦合度有点高，但是最终论文中并没有用到这个，因此也无所谓了
# 想想还是没敢删，调用的地方有点多，万一一删就全盘崩了呢
def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
                 torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
