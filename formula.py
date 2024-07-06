import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# folder path
folder_path = 'A_txt/'  # change to your folder path
save_path = '/content/drive/MyDrive/Raman generation/训练集结果txt/generatedA_eval/'

# collect all .txt files from the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# initializing the list of stored statistics 
avg_peak_heights = []
avg_fwhms = []
avg_snrs = []

# iterate through the list of files
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data = np.loadtxt(file_path, delimiter=',')
    spectrum = data[:, 0]
    # print(spectrum)

    # find peaks in CPU
    peaks, properties = find_peaks(spectrum, prominence=0.02)
    peak_heights = spectrum[peaks]

    # calculate FWHM
    widths, height, left_ips, right_ips = peak_widths(spectrum, peaks, rel_height=0.5)

    # calculate noise and SNR
    snrs = []
    for i, peak in enumerate(peaks):
        noise = np.std(spectrum[max(0, peak - 10):peak + 10])
        snrs.append(peak_heights[i] / noise)
        # print(snrs)

    # calculate mean and save
    avg_peak_heights.append(peak_heights)
    avg_fwhms.append(widths)
    avg_snrs.append(snrs)
print(np.min(avg_peak_heights))
print(np.min(avg_fwhms))
print(np.min(avg_snrs))
