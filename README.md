# Generative Physical-informed Spectroscopy (GPS) Model

This is the official code repository for the paper "Universal Spectroscopy Transfer with Physical Prior-Informed Deep Generative Learning".
This project is from [Tadesse Lab](https://tadesselab.mit.edu/), Mechanical Engineering Department, Massachusetts Institute of Technology.

## Overview
GPS transforms spectra through the priors of spectral broadening distributions of vibrations and absorptions and couples with deep generative models. We establish a probabilistic encoder $q_ϕ (z|x)$ to learn the prior probability distribution of Spectrum A, capturing the physical constraints inherent in the spectral transformation process, including the complex dependencies of line broadening, superposition, and wavenumber shifts. We compute the Kullback-Leibler (KL) divergence loss between the generated and input spectra to guild the spectra transformation. Additionally, we fully leverage the network's fitting capabilities by constructing an upsampling autoencoder that compensates for non-uniform broadening effects, including collisional broadening, transit-time broadening, saturation broadening, and environmental broadening.

## Usage
1. Do preparation for your dataset.
   Here is the steps:
   (1) If your file is .txt, first transfer it to .png file;
   (2) Resize to (256, 2048);
   (3) Rename your files with rename.py;
   (4) Seperate to train and test set.
2. Feed your prepared data to model by train_paras.py
3. You can test the performance by test_paras.py

## Data requirements
To transfer two kinds of spectra, we need one-to-one spectra pairs for the same sample/material. 

## License
This project is available under the MIT license.

## Reference
If you use this code, please give the citation of the paper.


