import os
from PIL import Image
import random

img_1_dir = 'A_img/'
img_2_dir = 'B_img/'

data_1_list = os.listdir(img_1_dir)
data_2_list = os.listdir(img_2_dir)

for i in data_1_list:
    img = Image.open(img_1_dir + i)
    img = img.resize((2048, 256))
    img.save(img_1_dir + i)

for i in data_2_list:
    img = Image.open(img_2_dir + i)
    img = img.resize((2048, 256))
    img.save(img_2_dir + i)