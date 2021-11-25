import os
import imageio
from PIL import Image
import argparse
from glob import glob
import re
import cv2
import numpy as np
from skimage.io import imread
from cv2 import resize

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',         type=str,  required=False, default='input')
parser.add_argument('--input_cull_factor', type=int,  required=False, default=1)
parser.add_argument('--resize_img_factor', type=float,required=False, default=1.0)
parser.add_argument('--ext',               type=str,  required=False, default='.JPG')
parser.add_argument('--output_dir',        type=str,  required=False, default='output')
args = parser.parse_args()

def resize_image_set(images, resize_factor):
    for i in range(len(images)):
        img = images[i]
        resize_dim = (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor))
        images[i] = resize(img, resize_dim)
    return images

# Function to produce the input image set
def get_input_image_set(input_dir, ext, input_cull_factor):
    input_imgs = glob(input_dir + '/*' + ext)
    input_imgs.sort()
    input_imgs_culled = []
    for i in range(len(input_imgs)):
        if i % input_cull_factor == 0:
            input_imgs_culled.append(input_imgs[i])
    input_imgs = input_imgs_culled

    # Function to help sort files with numbers
    def natural_keys(text):
        def atoi(text):
            return int(text) if text.isdigit() else text
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]

    input_imgs.sort(key=natural_keys)
    return input_imgs

# Function for aligning two images using OpenCV libraries
def alignImages(images):

    def alignImagesSub(im1, im2):
        # Function parameters
        MAX_FEATURES = 20000
        GOOD_MATCH_PERCENT = 0.25

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

        return im1Reg, h

    base_img = len(images)//2
    for i in range(len(images)):
        if i != base_img:
            im    = images[i]
            imReg, h = alignImagesSub(im, images[base_img])
            images[i] = imReg
    return images

# Function to output an image set as a GIF
def frames_to_gif(images, output_path, filenames=False):
    read_images = images
    if filenames:
        read_images = []
        for image in images:
            read_images.append(imread(image))

    imageio.mimsave(output_path, read_images)

# Find and cull down input set
input_imgs = get_input_image_set(args.input_dir, args.ext, args.input_cull_factor)
read_input_imgs = [imread(img) for img in input_imgs]
images = resize_image_set(read_input_imgs, args.resize_img_factor)

output_path = args.output_dir + '/animation.gif'

frames_to_gif(read_input_imgs, args.output_dir + '/unaligned.gif')
imave = np.average(np.array([np.array(im) for im in read_input_imgs]),axis=0)
cv2.imwrite(args.output_dir + '/unaligned_mean.png', imave)

alignImages(read_input_imgs)

frames_to_gif(read_input_imgs, args.output_dir + '/aligned.gif')
imave = np.average(np.array([np.array(im) for im in read_input_imgs]),axis=0)
cv2.imwrite(args.output_dir + '/aligned_mean.png', imave)
