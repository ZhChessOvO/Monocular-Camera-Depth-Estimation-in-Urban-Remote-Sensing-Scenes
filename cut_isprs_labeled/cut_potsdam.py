# Python程序
import cv2
import numpy

for k in range(7, 14):
    # img1 = cv2.imread('/user-data/GNN_RemoteSensor/2_Ortho_RGB/top_potsdam_7_' + str(k) + '_RGB.tif')  # 读取RGB原图像
    if k < 10:
        pic_name = '0' + str(k)
    else:
        pic_name = str(k)
    img2 = cv2.imread(
        '/mnt/data/isprs/Potsdam/1_DSM_normalisation/dsm_potsdam_07_' + pic_name + '_normalized_lastools.jpg')  # 读取Labels图像
    # 因为数据集中图片命名不规律，所以需要一批一批的分割
    # cv2.imread函数会把图片读取为（B，G，R）顺序，一定要注意！！！
    # cv2.imwrite函数最后会将通道调整回来，所以成对使用cv2.imread与cv2.imwrite不会改变通道顺序
    # 因为6000/10 = 600，所以6000x6000的图像可以划分为10x10个600x600大小的图像
    for i in range(10):
        for j in range(10):
            # img1_ = img1[600 * i: 600 * (i + 1), 600 * j: 600 * (j + 1), :]
            img2_ = img2[600 * i: 600 * (i + 1), 600 * j: 600 * (j + 1), :]
            # 注意下面name的命名，2400 + k * 100需要一批一批的调整，自己看到数据集中的图片命名就能知道什么意思了
            # name = i * 10 + j + 2400 + k * 100
            name = (k - 7) * 100 + i * 10 + j + 3100
            # 让RGB图像和标签图像的文件名对应
            name = str(name)
            # cv2.imwrite('./datasets/images/' + name + '.jpg', img1_)  # 所有的RGB图像都放到jpg文件夹下
            cv2.imwrite('/mnt/data/isprs/Potsdam/labeled_pic/' + name + '.png', img2_)  # 所有的标签图像都放到png文件夹下

    print('截止到现在的最大name为:', name)
