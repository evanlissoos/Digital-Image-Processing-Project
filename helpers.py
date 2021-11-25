import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from math import ceil
from skimage.io import imread
from cv2 import resize

from utils import warp
from model import SBMENet, ABMRNet, SynthesisNet

from torch.backends import cudnn

def run_frame(first_frame, second_frame, SBMNet, ABMNet, SynNet):
    with torch.no_grad():
        frame1 = first_frame
        frame3 = second_frame
        
        H = frame1.shape[2]
        W = frame1.shape[3]

        # 4K video requires GPU memory of more than 24GB. We recommend crop it into 4 regions with some margin.
        if H < 512:
            divisor = 64.
            D_factor = 1.
        else:
            divisor = 128.
            D_factor = 0.5
        
        H_ = int(ceil(H / divisor) * divisor * D_factor)
        W_ = int(ceil(W / divisor) * divisor * D_factor)

        frame1_ = F.interpolate(frame1, (H_, W_), mode='bicubic')
        frame3_ = F.interpolate(frame3, (H_, W_), mode='bicubic')

        SBM = SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
        SBM_= F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0

        frame2_1, Mask2_1 = warp(frame1_, SBM_ * (-1),  return_mask=True)
        frame2_3, Mask2_3 = warp(frame3_, SBM_       ,  return_mask=True)

        frame2_Anchor_ = (frame2_1 + frame2_3) / 2
        frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1-Mask2_1) + frame2_1 * (1-Mask2_3))

        Z  = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
        Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)
        
        ABM_bw, _ = ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM*(-1), Z_.exp())
        ABM_fw, _ = ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

        SBM_     = F.interpolate(SBM, (H, W), mode='bilinear')   * 20.0
        ABM_fw   = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
        ABM_bw   = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

        SBM_[:, 0, :, :] *= W / float(W_)
        SBM_[:, 1, :, :] *= H / float(H_)
        ABM_fw[:, 0, :, :] *= W / float(W_)
        ABM_fw[:, 1, :, :] *= H / float(H_)
        ABM_bw[:, 0, :, :] *= W / float(W_)
        ABM_bw[:, 1, :, :] *= H / float(H_)

        divisor = 8.
        H_ = int(ceil(H / divisor) * divisor)
        W_ = int(ceil(W / divisor) * divisor)
        
        Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)
        
        Syn_inputs = F.interpolate(Syn_inputs, (H_,W_), mode='bilinear')
        Syn_inputs[:, 6, :, :] *= float(W_) / W
        Syn_inputs[:, 7, :, :] *= float(H_) / H
        Syn_inputs[:, 8, :, :] *= float(W_) / W
        Syn_inputs[:, 9, :, :] *= float(H_) / H
        Syn_inputs[:, 10, :, :] *= float(W_) / W
        Syn_inputs[:, 11, :, :] *= float(H_) / H 

        result = SynNet(Syn_inputs)
        
        result = F.interpolate(result, (H,W), mode='bicubic')

        return result


def output_img(img, path):
    fn, ext = os.path.splitext(path)
    
    if ext in ['.jpg','.png','.jpeg','.bmp']:
        save_image(img, path)
        print('\'%s\' saved!'%path)
    else:
        raise ValueError('Change [\'%s\'] to [\'.jpg\',\'.png\',\'.jpeg\',\'.bmp\']'% ext)


# Function to resize images
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