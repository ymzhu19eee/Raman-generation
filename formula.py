import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# 文件夹路径
folder_path = 'A_txt/'  # 修改为你的文件夹路径
save_path = '/content/drive/MyDrive/Raman generation/训练集结果txt/generatedA_eval/'

# 获取文件夹中所有.txt文件
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 初始化存储统计数据的列表
avg_peak_heights = []
avg_fwhms = []
avg_snrs = []

# 遍历文件列表
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data = np.loadtxt(file_path, delimiter=',')
    spectrum = data[:, 0]
    # print(spectrum)

    # 在 CPU 上找到峰值
    peaks, properties = find_peaks(spectrum, prominence=0.02)
    peak_heights = spectrum[peaks]

    # 计算FWHM
    widths, height, left_ips, right_ips = peak_widths(spectrum, peaks, rel_height=0.5)

    # 计算噪声和SNR
    snrs = []
    for i, peak in enumerate(peaks):
        noise = np.std(spectrum[max(0, peak - 10):peak + 10])
        snrs.append(peak_heights[i] / noise)
        # print(snrs)

    # 计算平均值并存储
    avg_peak_heights.append(peak_heights)
    avg_fwhms.append(widths)
    avg_snrs.append(snrs)
print(np.min(avg_peak_heights))
print(np.min(avg_fwhms))
print(np.min(avg_snrs))