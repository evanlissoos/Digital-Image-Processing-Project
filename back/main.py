import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from math import ceil
from skimage.io import imread
from glob import glob

from utils import warp
from model import SBMENet, ABMRNet, SynthesisNet

from torch.backends import cudnn
from helpers import *
cudnn.benchmark = True

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',  type=str, required=False, default='input')
parser.add_argument('--output_dir', type=str, required=False, default='output')
parser.add_argument('--ext',        type=str, required=False, default='.png')
parser.add_argument('--num_rounds', type=int, required=False, default=1)
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

input_imgs = glob(args.input_dir + '/*' + args.ext)
input_imgs.sort()
print(input_imgs)
frames = [TF.to_tensor(imread(img)).unsqueeze(0).cuda() for img in input_imgs]


# Iterate through number rounds to generate interpolated images
for round in range(args.num_rounds):
    print('Starting interpolation round ' + str(round))
    frames_new = frames.copy()
    for i in range(len(frames)-1):
        print('Interpolating frames ' + str(i) + ' and ' + str(i+1))
        result = run_frame(frames[i], frames[i+1], SBMNet, ABMNet, SynNet)
        frames_new.insert(i+1, result)
    frames = frames_new

# Output all frames into output directory
for i in range(len(frames)):
    output_img(frames[i], args.output_dir + '/' + str(i) + args.ext)