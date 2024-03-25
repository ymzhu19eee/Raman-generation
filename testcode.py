# -*- coding: utf-8 -*-
"""testcode.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fJAkgGNA91SVPZjLp_GTvIHx-l_0UZ95
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install pytorch-msssim

!pip install pytorch_ssim

!pip install pytorch-msssim
!pip install pytorch_ssim

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import io
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from IPython.display import Image as IMG
from IPython.display import display
from skimage.metrics import structural_similarity as ssim
from math import log10
import pytorch_ssim
from pytorch_msssim import ssim, MS_SSIM

# 如果您想要使用单尺度SSIM
def ssim_loss(input_images, target_images):
    return 1 - ssim(input_images, target_images, data_range=1, size_average=True) # data_range取决于您图像的取值范围

criterion_ssim = ssim_loss  # 或者 ms_ssim_loss，如果您使用多尺度SSIM

def convert_txt_to_image(file_path, image_folder, image_width, image_height):
    # 确保输出图像文件夹存在
    os.makedirs(image_folder, exist_ok=True)

    # 获取文件名（不包含扩展名）
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # 读取数据，假设文件中的数据以逗号分隔
    data = np.loadtxt(file_path, delimiter=',')

    # 假定数据的第一列是波长（不需要转换为图像），第二列是强度
    # 如果有多列数据，请根据实际情况进行调整
    spectrum = data[:, 1]

    # 标准化光谱数据到0-1之间
    spectrum_normalized = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))

    # 将一维光谱数据转换为二维图像格式
    image = np.tile(spectrum_normalized, (image_height, 1))

    # 绘制图像并保存为PNG格式
    plt.imsave(os.path.join(image_folder, filename + '.png'), image, cmap='gray')

def process_folder(folder_path, image_folder, image_width, image_height):
    # 遍历文件夹中的所有文件
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            convert_txt_to_image(file_path, image_folder, image_width, image_height)

# NIR数据和Raman数据的文件夹路径
nir_folder = '/content/drive/MyDrive/Raman generation/NIR_3d' # 更新为你的NIR数据文件夹路径
raman_folder = '/content/drive/MyDrive/Raman generation/Raman_processed_3d' # 更新为你的Raman数据文件夹路径

# 目标图像文件夹
nir_image_folder = '/content/drive/MyDrive/Raman generation/NIR_3d_images'
raman_image_folder = '/content/drive/MyDrive/Raman generation/Raman_processed_3d_images'

# 定义图像的宽度和高度
image_width = 256  # 假设每个光谱有256个数据点
image_height = 256  # 你可以根据需要设置不同的高度

# 处理NIR文件夹中的所有文件
process_folder(nir_folder, nir_image_folder, image_width, image_height)

# 处理Raman文件夹中的所有文件
process_folder(raman_folder, raman_image_folder, image_width, image_height)


# 自定义数据集
class SpectraDataset(Dataset):
    def __init__(self, nir_image_folder, raman_image_folder, transform=None):
        self.nir_image_folder = nir_image_folder
        self.raman_image_folder = raman_image_folder
        self.transform = transform

        # 获取文件夹中的所有图像文件名并按字母顺序排序
        self.nir_image_filenames = sorted([f for f in os.listdir(nir_image_folder) if f.endswith('.png')])
        self.raman_image_filenames = sorted([f for f in os.listdir(raman_image_folder) if f.endswith('.png')])

        # 检查两个列表的长度是否相同
        if len(self.nir_image_filenames) != len(self.raman_image_filenames):
            raise ValueError("Number of NIR and Raman images must be the same.")

    def __len__(self):
        # 两个列表的长度应该是相同的
        return len(self.nir_image_filenames)

    def __getitem__(self, idx):
        # 根据排序后的索引获取文件名
        nir_image_path = os.path.join(self.nir_image_folder, self.nir_image_filenames[idx])
        raman_image_path = os.path.join(self.raman_image_folder, self.raman_image_filenames[idx])

        # 加载图像
        nir_image = Image.open(nir_image_path)
        raman_image = Image.open(raman_image_path)

        # # 打印单个文件的形状
        # print(f"NIR image shape: {nir_image.size}")
        # print(f"Raman image shape: {raman_image.size}")

        # 如果有变换，则应用它
        if self.transform:
            nir_image = self.transform(nir_image)
            raman_image = self.transform(raman_image)

            # # 打印变换后的形状
            # print(f"NIR image shape after transform: {nir_image.shape}")
            # print(f"Raman image shape after transform: {raman_image.shape}")

        return {'A': nir_image, 'B': raman_image}

# 数据转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 实例化数据集
nir_image_folder = '/content/drive/MyDrive/Raman generation/NIR_3d_images/'
raman_image_folder = '/content/drive/MyDrive/Raman generation/Raman_processed_3d_images/'
dataset = SpectraDataset(nir_image_folder, raman_image_folder, transform=transform)

# 计算数据集大小
dataset_size = len(dataset)
print(f"Dataset size: {dataset_size}")

# 计算训练集、验证集和测试集的大小
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

# 随机分割数据集
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 创建数据加载器
batch_size = 1  # Pix2Pix通常使用batch_size为1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练循环
num_epochs = 1
lambda_L1 = 200
best_val_loss = float('inf')

# 记录loss
train_G_losses = []
train_D_losses = []
val_G_losses = []
val_D_losses = []

# # 创建数据加载器
# batch_size = 1  # Pix2Pix通常使用batch_size为1
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 接下来，可以使用上面创建的dataloader在Pix2Pix网络中训练模型

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 定义鉴别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
            nn.Sigmoid(),
            # nn.tanh(),
        )

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        outputs = self.model(inputs)
        return outputs

# 实例化生成器和鉴别器
netG = Generator().to(device)
netD = Discriminator().to(device)

# 定义损失函数和优化器
criterion_GAN = nn.BCELoss().to(device)
criterion_L1 = nn.L1Loss().to(device)

optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    train_G_loss = 0.0
    train_D_loss = 0.0
    val_G_loss = 0.0
    val_D_loss = 0.0

    # 训练模式
    netG.train()
    netD.train()

    for i, data in enumerate(train_loader):
        # 加载数据并移动到GPU
        nir_images = data['A'].to(device)
        raman_images = data['B'].to(device)

        # 训练discriminator和generator
        # 生成器前向传播
        fake_raman = netG(nir_images)

        # 训练鉴别器
        netD.zero_grad()
        real_output = netD(nir_images, raman_images)
        fake_output = netD(nir_images, fake_raman.detach())

        real_label = torch.ones_like(real_output,device=device)
        fake_label = torch.zeros_like(fake_output,device=device)

        loss_D_real = criterion_GAN(real_output, real_label)
        loss_D_fake = criterion_GAN(fake_output, fake_label)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        fake_output = netD(nir_images, fake_raman)
        loss_G_GAN = criterion_GAN(fake_output, real_label)
        loss_G_L1 = criterion_L1(fake_raman, raman_images)
        # 计算SSIM损失
        loss_ssim = criterion_ssim(fake_raman, raman_images)
        lambda_ssim = 0.3
        loss_G = loss_G_GAN + lambda_L1 * loss_G_L1 + lambda_ssim * loss_ssim
        loss_G.backward()
        optimizerG.step()

        # 累计训练loss
        train_D_loss += loss_D.item()
        train_G_loss += loss_G.item()

    # 计算每个epoch的平均训练loss
    train_D_loss /= len(train_loader)
    train_G_loss /= len(train_loader)
    train_D_losses.append(train_D_loss)
    train_G_losses.append(train_G_loss)

    # 验证模式
    netG.eval()
    netD.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # 加载数据并移动到GPU
            nir_images = data['A'].to(device)
            raman_images = data['B'].to(device)

            # 生成器前向传播
            fake_raman = netG(nir_images)

            # 计算验证loss
            real_output = netD(nir_images, raman_images)
            fake_output = netD(nir_images, fake_raman)
            real_label = torch.ones_like(real_output, device=device)
            fake_label = torch.zeros_like(fake_output, device=device)
            loss_D_real = criterion_GAN(real_output, real_label)
            loss_D_fake = criterion_GAN(fake_output, fake_label)
            loss_D_val = (loss_D_real + loss_D_fake) * 0.5
            loss_G_val = criterion_GAN(fake_output, real_label) + lambda_L1 * criterion_L1(fake_raman, raman_images)

            # 累计验证loss
            val_D_loss += loss_D_val.item()
            val_G_loss += loss_G_val.item()

    # 计算每个epoch的平均验证loss
    val_D_loss /= len(val_loader)
    val_G_loss /= len(val_loader)
    val_D_losses.append(val_D_loss)
    val_G_losses.append(val_G_loss)

    # 打印训练和验证信息
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss D: {train_D_loss:.4f}, Train Loss G: {train_G_loss:.4f}, "
          f"Val Loss D: {val_D_loss:.4f}, Val Loss G: {val_G_loss:.4f}")

    # 保存最佳模型权重
    if val_G_loss < best_val_loss:
        best_val_loss = val_G_loss
        torch.save(netG.state_dict(), 'generator_best.pth')
        torch.save(netD.state_dict(), 'discriminator_best.pth')

# 在测试集上评估最佳模型
netG.load_state_dict(torch.load('generator_best.pth'))
netD.load_state_dict(torch.load('discriminator_best.pth'))
netG.eval()
netD.eval()

test_G_loss = 0.0
test_D_loss = 0.0
ssim_scores = []
psnr_scores = []
mse_scores = []

# 创建用于保存生成图像的文件夹
os.makedirs("test_results", exist_ok=True)

with torch.no_grad():
    for i, data in enumerate(test_loader):
        # 加载数据并移动到GPU
        nir_images = data['A'].to(device)
        raman_images = data['B'].to(device)

        # 生成器前向传播
        fake_raman = netG(nir_images)

        # 计算测试loss
        real_output = netD(nir_images, raman_images)
        fake_output = netD(nir_images, fake_raman)
        real_label = torch.ones_like(real_output, device=device)
        fake_label = torch.zeros_like(fake_output, device=device)
        loss_D_real = criterion_GAN(real_output, real_label)
        loss_D_fake = criterion_GAN(fake_output, fake_label)
        loss_D_test = (loss_D_real + loss_D_fake) * 0.5
        loss_G_test = criterion_GAN(fake_output, real_label) + lambda_L1 * criterion_L1(fake_raman, raman_images)

        # 累计测试loss
        test_D_loss += loss_D_test.item()
        test_G_loss += loss_G_test.item()

        # 保存生成图像、输入图像和目标图像
        vutils.save_image(fake_raman.detach(), f"test_results/generated_{i}.png", normalize=True)
        vutils.save_image(nir_images.detach(), f"test_results/input_{i}.png", normalize=True)
        vutils.save_image(raman_images.detach(), f"test_results/target_{i}.png", normalize=True)

        # 计算SSIM、PSNR和MSE
        ssim_score = ssim(fake_raman.cpu().squeeze().permute(1, 2, 0).numpy(), raman_images.cpu().squeeze().permute(1, 2, 0).numpy(), multichannel=True, data_range=1.0)
        ssim_scores.append(ssim_score)

        mse = F.mse_loss(fake_raman, raman_images)
        mse_scores.append(mse.item())

        psnr = 10 * log10(1 / mse.item())
        psnr_scores.append(psnr)

        # 输出当前样本的SSIM、PSNR和MSE
        print(f"Sample {i+1}: SSIM={ssim_score:.4f}, PSNR={psnr:.4f}, MSE={mse.item():.4f}")

        # 显示生成图像、输入图像和目标图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(fake_raman.cpu().squeeze().permute(1, 2, 0))
        axes[0].set_title("Generated Raman")
        axes[1].imshow(nir_images.cpu().squeeze().permute(1, 2, 0))
        axes[1].set_title("Input NIR")
        axes[2].imshow(raman_images.cpu().squeeze().permute(1, 2, 0))
        axes[2].set_title("Target Raman")
        plt.show()

# 计算平均测试loss
test_D_loss /= len(test_loader)
test_G_loss /= len(test_loader)

# 计算SSIM、PSNR和MSE的统计信息
ssim_mean = np.mean(ssim_scores)
ssim_std = np.std(ssim_scores)
ssim_best = np.max(ssim_scores)
ssim_worst = np.min(ssim_scores)

psnr_mean = np.mean(psnr_scores)
psnr_std = np.std(psnr_scores)
psnr_best = np.max(psnr_scores)
psnr_worst = np.min(psnr_scores)

mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)
mse_best = np.min(mse_scores)
mse_worst = np.max(mse_scores)

# 绘制SSIM、PSNR和MSE曲线
plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 1)
plt.plot(ssim_scores)
plt.axhline(y=ssim_mean, color='r', linestyle='-', label=f'Mean: {ssim_mean:.4f}')
plt.axhline(y=ssim_best, color='g', linestyle='--', label=f'Best: {ssim_best:.4f}')
plt.axhline(y=ssim_worst, color='b', linestyle='--', label=f'Worst: {ssim_worst:.4f}')
plt.title(f'SSIM (Std: {ssim_std:.4f})')
plt.ylabel('Score')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(psnr_scores)
plt.axhline(y=psnr_mean, color='r', linestyle='-', label=f'Mean: {psnr_mean:.2f}')
plt.axhline(y=psnr_best, color='g', linestyle='--', label=f'Best: {psnr_best:.2f}')
plt.axhline(y=psnr_worst, color='b', linestyle='--', label=f'Worst: {psnr_worst:.2f}')
plt.title(f'PSNR (Std: {psnr_std:.2f})')
plt.ylabel('Score (dB)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(mse_scores)
plt.axhline(y=mse_mean, color='r', linestyle='-', label=f'Mean: {mse_mean:.4f}')
plt.axhline(y=mse_best, color='g', linestyle='--', label=f'Best: {mse_best:.4f}')
plt.axhline(y=mse_worst, color='b', linestyle='--', label=f'Worst: {mse_worst:.4f}')
plt.title(f'MSE (Std: {mse_std:.4f})')
plt.ylabel('Score')
plt.xlabel('Sample Index')
plt.legend()

plt.tight_layout()
plt.savefig('evaluation_plots.png')

# # 计算平均SSIM、PSNR和MSE
# avg_ssim = sum(ssim_scores) / len(ssim_scores)
# avg_psnr = sum(psnr_scores) / len(psnr_scores)
# avg_mse = sum(mse_scores) / len(mse_scores)

print(f"Test Loss D: {test_D_loss:.4f}, Test Loss G: {test_G_loss:.4f}")
# print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}, Average MSE: {avg_mse:.4f}")