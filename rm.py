import os
import shutil

img_1_dir = 'A_img/'
img_2_dir = 'B_img/'

print(os.listdir(img_1_dir))
print(os.listdir(img_2_dir))
num = 0

for i in os.listdir(img_1_dir):
    os.rename(img_1_dir + i, img_1_dir + str(num) + '.png')
    num = num + 1

num = 0
for i in os.listdir(img_2_dir):
    os.rename(img_2_dir + i, img_2_dir + str(num) + '.png')
    num = num + 1