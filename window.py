import cv2
#from cv2 import imread
import numpy as np
from glob import glob
import argparse

#NUMERIC KEY DEF
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
#https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python

#MASK GEN DEF
def mask_gen(k,THRESH,input_imgs):
    #RETURNS A MASK WITH VALUES OF 0 - 1
    #k size of the block for temporal average
    #THRESH theshold of movement
    #input_imgs array of directions to files
    # input_imgs= sorted(input_imgs,key=numericalSort)
    
    #READ EACH BLOCK OF IMAGES
    mask=np.zeros(input_imgs[0].shape[0:2])
    for i in range(len(input_imgs)-k) :
        frames = [img for img in input_imgs[0+i:k+i]]

        gray_bg=np.array(np.zeros(frames[0].shape[0:2]))
        #AVRG THE FRAMES
        for j in range(len(frames)-1):
            gray_frame = cv2.cvtColor(frames[j],cv2.COLOR_BGR2GRAY)
            gray_bg = gray_bg + gray_frame

        gray_bg = gray_bg / (len(frames)-1)

        #TURN TO GRAYSCALE
        gray_img = cv2.cvtColor(frames[-1],cv2.COLOR_BGR2GRAY)

        size = 10
        kernel = np.ones((size,size),np.float32)/size**2

        #GAUSS FILTERING
        gauss_bg =  cv2.filter2D(gray_bg,-1,kernel)
        gauss_img =  cv2.filter2D(gray_img,-1,kernel)

        #ACUMULATE MASKS
        diff = abs(gauss_bg.astype(np.int0)-gauss_img.astype(np.int0))
        diff = (diff > THRESH).astype(np.uint8)
        mask = mask + diff

        #STATUS UPDATE
        #if (i % 20 == 0):
        #    print(i)

    #LIMIT MASK TO 0 - 1 VALUES
    mask = np.clip(mask,0,1).astype(np.uint8)

    #GENERATE KERNEL
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #MORPHOLOGY OPERATIONS
    mask_test =  cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,iterations=2)
    mask_test2 = cv2.dilate(mask_test, kernel,iterations=1)
    mask_test3 = cv2.erode(mask_test2, kernel,iterations=1)

    return mask_test3

#FOR TESTING
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',         type=str,  required=False, default='input')
parser.add_argument('--input_cull_factor', type=int,  required=False, default=1)
parser.add_argument('--ext',               type=str,  required=False, default='.jpg')
args = parser.parse_args()
args.DDP = False


#Cull down input set
input_imgs = glob(args.input_dir + '/*' + args.ext)
input_imgs= sorted(input_imgs,key=numericalSort)

input_imgs_culled = []
for i in range(len(input_imgs)):
    if i % args.input_cull_factor == 0:
        input_imgs_culled.append(input_imgs[i])
input_imgs = input_imgs_culled

mask = mask_gen(10,6,input_imgs)

cv2.imshow("IMG",cv2.imread(input_imgs[0]))
cv2.imshow("MASK",mask*255)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""