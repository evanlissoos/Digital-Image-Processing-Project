import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from math import ceil
from skimage.io import imread
import matplotlib.pyplot as plt

from utils import warp
from model import SBMENet, ABMRNet, SynthesisNet

from torch.backends import cudnn
from helpers import *
from window import *
cudnn.benchmark = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',         type=str,  required=False, default='input')
parser.add_argument('--input_cull_factor', type=int,  required=False, default=1)
parser.add_argument('--resize_img_factor', type=float,required=False, default=1.0)
# parser.add_argument('--exposure_length',   type=float,required=False, default=1.0)
# Add this functionality to allow for resizing of the exposure length
parser.add_argument('--output_dir',        type=str,  required=False, default='output')
parser.add_argument('--output_all_frames', type=bool, required=False, default=False)
parser.add_argument('--output_mean_med',   type=bool, required=False, default=True)
parser.add_argument('--ext',               type=str,  required=False, default='.jpg')
parser.add_argument('--num_rounds',        type=int,  required=False, default=1)
parser.add_argument('--mask',              type=bool, required=False, default=True)
args = parser.parse_args()

args.DDP = False

SBMNet = SBMENet()
ABMNet = ABMRNet()
SynNet = SynthesisNet(args)

SBMNet.load_state_dict(torch.load('Best/SBME_ckpt.pth', map_location='cpu'))
ABMNet.load_state_dict(torch.load('Best/ABMR_ckpt.pth', map_location='cpu'))
SynNet.load_state_dict(torch.load('Best/SynNet_ckpt.pth', map_location='cpu'))

for param in SBMNet.parameters():
    param.requires_grad = False 
for param in ABMNet.parameters():
    param.requires_grad = False
for param in SynNet.parameters():
    param.requires_grad = False
   
SBMNet.cuda()
ABMNet.cuda()
SynNet.cuda()

# Cull down input set
# input_imgs = glob(args.input_dir + '/*' + args.ext)
# input_imgs.sort()
# input_imgs_culled = []
# for i in range(len(input_imgs)):
#     if i % args.input_cull_factor == 0:
#         input_imgs_culled.append(input_imgs[i])
# input_imgs = input_imgs_culled
input_imgs = get_input_image_set(args.input_dir, args.ext, args.input_cull_factor)

print('Using ' + str(len(input_imgs)) + ' source frames')

read_input_imgs = resize_image_set([imread(img) for img in input_imgs], args.resize_img_factor)
# read_input_imgs = alignImages(read_input_imgs)

k = 10
thresh = 3
motion_mask = mask_gen(k, thresh, read_input_imgs)
motion_mask = np.array([motion_mask, motion_mask, motion_mask])
motion_mask_inv = np.invert(motion_mask) & 1
import cv2
motion_mask = mask_gen(k, thresh, read_input_imgs) * 255
cv2.imshow('Mask', motion_mask.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
quit()

print('Image size ' + str(read_input_imgs[0].shape))
frames = [TF.to_tensor(img).unsqueeze(0) for img in read_input_imgs]

# Iterate through number rounds to generate interpolated images
for round in range(args.num_rounds):
    print('Starting interpolation round ' + str(round))
    frames_new = frames.copy()
    for i in range(len(frames)-1):
        print('Interpolating frames ' + str(i) + ' and ' + str(i+1))
        gpu_frame0 = frames[i].cuda()
        gpu_frame1 = frames[i+1].cuda()
        result = run_frame(gpu_frame0, gpu_frame1, SBMNet, ABMNet, SynNet).cpu()
        frames_new.insert(i+1, result)
    frames = frames_new

# Output all frames into output directory
if args.output_all_frames:
    for i in range(len(frames)):
        output_img(frames[i], args.output_dir + '/' + str(i) + args.ext)

# Compute and output mean and median images
if args.output_mean_med:
    # Processing
    # To convert to numpy array: frames[0].numpy()
    base_frame = frames[-1].numpy()[0] * motion_mask_inv
    stack = torch.stack(frames)
    del frames
    if args.mask:
        print('Generating median image')
        median = torch.median(stack, 0)[0]
        median = median.numpy()[0] * motion_mask + base_frame
        median = np.swapaxes(np.swapaxes(median, 0, 2), 0, 1)
        median = median * 255
        median = median.astype(np.uint8)
        print('Generating mean image')
        mean = torch.mean(stack, 0)
        mean = mean.numpy()[0] * motion_mask + base_frame
        mean = np.swapaxes(np.swapaxes(mean, 0, 2), 0, 1)
        mean = mean * 255
        mean = mean.astype(np.uint8)
        plt.imsave(args.output_dir + '/median' + args.ext, median)
        plt.imsave(args.output_dir + '/mean' + args.ext, mean)
    else:
        print('Generating median image')
        median = torch.median(stack, 0)[0]
        print('Generating mean image')
        mean = torch.mean(stack, 0)
        output_img(median, args.output_dir + '/median' + args.ext)
        output_img(mean, args.output_dir + '/mean' + args.ext)