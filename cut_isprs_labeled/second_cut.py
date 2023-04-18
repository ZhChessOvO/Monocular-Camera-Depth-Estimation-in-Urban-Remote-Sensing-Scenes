from PIL import Image

image_dir = '/mnt/data/caozehao/Potsdam/RGB/'
depth_dir = '/mnt/data/caozehao/Potsdam/labeled/'
dist_dir = '/mnt/data/caozehao/Potsdam/'

box = (0, 0, 304, 228)

# for i in range(3800):
#     filename = image_dir + str(i) + '.jpg'
#     distname = dist_dir + 'RGB1/' + str(i) + '.jpg'
#     with Image.open(filename) as image:
#         img = image.crop(box)
#         img.save(distname)
#         print(i)

for i in range(3800):
    filename = depth_dir + str(i) + '.png'
    distname = dist_dir + 'labeled1/' + str(i) + '.png'
    with Image.open(filename) as image:
        img = image.crop(box)
        img.save(distname)
        print(i)
