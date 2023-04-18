import argparse
import torch
import torch.nn.parallel

from models import my_module, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt

plt.set_cmap("jet")


# 此函数主要用于对比测试，用来设置不同种类的网络下的参数，实际demo中几乎只用到senet的部分
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
    # 加载保存好的模型
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./save/our_model.pth.tar')['state_dict'])
    model.eval()

    # 创建dataloader对象，使用对象读取指定RGB图片
    nyu2_loader = loaddata.readNyu2('./demo/RGB0.jpg')

    # 将图片过一遍模型，将输出结果保存在本地
    # 这里为了方便，套用了test.py的test函数略作修改，区别在于这里的dataloader只会加载一张图片
    test(nyu2_loader, model)


# 改自test.py，区别在于这里的dataloader只会加载一张图片，且直接输出图片而不是进一步计算误差
def test(nyu2_loader, model):
    for i, image in enumerate(nyu2_loader):
        image = torch.autograd.Variable(image).cuda()
        out = model(image)

        matplotlib.image.imsave('./demo/test20220616.png', out.view(out.size(2), out.size(3)).data.cpu().numpy())


# 调用主函数
if __name__ == '__main__':
    main()
