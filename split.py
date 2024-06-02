import os
import random
import shutil

img_1_dir = 'A_img/'
img_2_dir = 'B_img/'

text_1_dir = 'A_txt/'
text_2_dir = 'B_txt/'

for i in os.listdir(img_1_dir):
    if random.random()>0.05:
        shutil.copy(img_1_dir + i, 'trainA_img/' + i)
        shutil.copy(img_2_dir + i, 'trainB_img/' + i)

        shutil.copy(text_1_dir + i[:-4] + '.txt', 'trainA_txt/' + i[:-4] + '.txt')
        shutil.copy(text_2_dir + i[:-4] + '.txt', 'trainB_txt/' + i[:-4] + '.txt')

    else:
        shutil.copy(img_1_dir + i, 'testA_img/' + i)
        shutil.copy(img_2_dir + i, 'testB_img/' + i)

        shutil.copy(text_1_dir + i[:-4] + '.txt', 'testA_txt/' + i[:-4] + '.txt')
        shutil.copy(text_2_dir + i[:-4] + '.txt', 'testB_txt/' + i[:-4] + '.txt')
