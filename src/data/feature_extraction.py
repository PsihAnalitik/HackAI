from typing import List
import cv2
from scipy.stats import entropy, skew, kurtosis
import numpy as np

def blur_coefficient(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def calculate_image_entropy(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _bins = 128

    hist, _ = np.histogram(gray_img.ravel(), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    
    return entropy(prob_dist, base=2)

def kurtosis_(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return kurtosis(img.flatten())

def skew_(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return skew(img.flatten())

def get_features(img_path):
    img = cv2.imread(img_path)
    img = img[100:img.shape[0] - 100, :, :]
    img = cv2.resize(img, (256, 256))
    per_channel_mean = img.mean(axis=(0,1))
    per_channel_std = img.std(axis=(0,1))
    laplacian_var = blur_coefficient(img)
    entropy = calculate_image_entropy(img)
    kurtosis = kurtosis_(img)
    skew = skew_(img)

    return [per_channel_mean[0], per_channel_mean[1], per_channel_mean[2], per_channel_std[0], per_channel_std[1], per_channel_std[2], kurtosis, skew, laplacian_var, entropy]
    