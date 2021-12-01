import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

img1 = imread('output/median_baseline.jpg')
img2 = imread('output/4_rounds_median.jpg')
img2 = imread('input/1.jpg')
# img2 = imread('output/median20.jpg')

print(ssim(img1, img2, multichannel=True))