import pandas as pd
import cv2
import os
import numpy as np
test_csv = 'af2019-ksyun-testB-20190424/list.csv'

test = pd.read_csv(test_csv, sep=',', header=0)

# print(test.iat[0, 1])

img1list = test.ix[:, 1]
img2list = test.ix[:, 2]
# print(img1list)
outpath = 'testB_show/'
if not os.path.exists(outpath):
    os.mkdir(outpath)

img_root = 'af2019-ksyun-testB-20190424/images/'

for i, (img1name, img2name) in enumerate(zip(img1list, img2list)):
    img1path = img_root + img1name
    img2path = img_root + img2name
    img1 = cv2.imread(img1path)
    img2 = cv2.imread(img2path)
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    h, w = max(h1, h2), w1+w2
    shape = (h, w, 3)
    img = np.zeros(shape, np.uint8)
    img[:h1, :w1, :] = img1
    img[:h2, w1:w, :] = img2
    imgpath = outpath + str(i) + '.jpg'
    cv2.imwrite(imgpath, img)
    # break