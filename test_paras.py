import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, peak_widths
from torchvision.utils import save_image

# parameters normalization
def n_seq(seq):
    normalized_to_range = []
    for i in range(len(seq)):
        normalized_to_range.append((np.array(seq)[i] - seq_min[i]) / (seq_max[i] - seq_min[i]))
    return np.array(normalized_to_range)

#physical prior parameters calculation
def compute_params(txt_path):
    data = np.loadtxt(txt_path, delimiter=',')
    spectrum = data[:, 0]

    peaks, properties = find_peaks(spectrum, prominence=0.02)
    peak_heights = spectrum[peaks]
    widths, height, left_ips, right_ips = peak_widths(spectrum, peaks, rel_height=0.5)


    params_sequence = np.concatenate((peak_heights, widths), axis=0)
    return params_sequence

#Dataset preparation
class CustomDataset(Dataset):
    def __init__(self, img_dir1, img_dir2, txt_dir1, txt_dir2, transform1=None):
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.txt_dir1 = txt_dir1
        self.txt_dir2 = txt_dir2
        self.transform1 = transform1
        self.img_names = [f for f in os.listdir(img_dir1) if os.path.isfile(os.path.join(img_dir1, f))]
        self.txt_names = [f for f in os.listdir(txt_dir1) if os.path.isfile(os.path.join(txt_dir1, f))]
        self.img_names.sort()
        self.txt_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        txt_name = self.txt_names[idx]
        img_path1 = os.path.join(self.img_dir1, img_name)
        img_path2 = os.path.join(self.img_dir2, img_name)
        txt_path1 = os.path.join(self.txt_dir1, txt_name)
        txt_path2 = os.path.join(self.txt_dir2, txt_name)

        image1 = Image.open(img_path1).convert('L')
        image2 = Image.open(img_path2).convert('L')
        if self.transform1:
            image1 = self.transform1(image1)
            image2 = self.transform1(image2)

        params_sequence1 = n_seq(compute_params(txt_path1))
        params_sequence2 = n_seq(compute_params(txt_path2))

        return image1, image2, params_sequence1, params_sequence2

class SpectroGen(nn.Module):
    def __init__(self):
        super(SpectroGen, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(16*128*256, 512)
        self.fc_mu_img = nn.Linear(512, 256)
        self.fc_logvar_img = nn.Linear(512, 256)
        self.fc1_seq = nn.Linear(2, 512)
        self.bn_1 = nn.BatchNorm1d(512)

        # Decoder
        self.fc2 = nn.Linear(256, 16*128*256)
        self.dropout6 = nn.Dropout(0.25)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.fc2_seq = nn.Linear(16*128*256, 2)

    def encode(self, x, params_sequence):
        params_sequence = torch.tensor(params_sequence, dtype=torch.float32)
        x_seq = F.relu(self.fc1_seq(params_sequence))


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = torch.add(x, x_seq)
        return self.fc_mu_img(x), self.fc_logvar_img(x), x_seq

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = self.dropout6(x)
        x_seq = self.fc2_seq(x)
        x = x.view(-1, 256, 16, 128)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x, x_seq

    def forward(self, x, params_sequence):
        mu_img, logvar_img, mu_seq = self.encode(x, params_sequence)
        z_img = self.reparameterize(mu_img, logvar_img)
        recon_img, recon_seq = self.decode(z_img)
        return recon_img, recon_seq, mu_img, logvar_img, mu_seq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectroGen().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()


transform = transforms.Compose([
    transforms.Resize((256, 2048)),
    transforms.ToTensor(),
])

seq_mean = (3613.7053984374998, 845.4835775222828, 2.017627638546281)
seq_std = (1061.5666450890458, 246.4933204514553, 0.007412496169852322)

seq_max = (4001.569, 941.4534680476987, 2.037972656054459)
seq_min = (700.0332, 168.96417954190701, 2.014925493215356)


dataset = CustomDataset('testA_img', 'testB_img', 'testA_txt', 'testB_txt', transform1=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

def test(model, dataloader, save_dir):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (image1, image2, params_sequence1, params_sequence2) in enumerate(dataloader):
            image1, image2, params_sequence1, params_sequence2 = image1.cuda(), image2.cuda(), params_sequence1.cuda(), params_sequence2.cuda()

            # Save input images
            for idx in range(len(image1)):
                input_img_path = os.path.join(save_dir, f'input_image_{batch_idx * dataloader.batch_size + idx}.png')
                save_image(image1[idx], input_img_path)

    test_loss /= len(dataloader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

# Assuming dataloader and save_dir are already defined
test(model, dataloader, 'testA_img_output/')
