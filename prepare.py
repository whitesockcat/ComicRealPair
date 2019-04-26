import os
from glob import glob
import shutil

train_mix_path = 'train_mix/'
if not os.path.exists(train_mix_path):
    os.mkdir(train_mix_path)


rootdir = 'af2019-ksyun-training/images/'

cat_num = 1000
for i in range(cat_num):
    image_lists = glob(rootdir + str(i) + '/*/*.jpg')
    for img_path in image_lists:
        targetDir = train_mix_path + img_path[-36:]
        shutil.copy(img_path, targetDir)
    