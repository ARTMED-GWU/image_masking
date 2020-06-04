#!/usr/bin/env python
 
'''
Welcome to the Image Masking Program!
 
This program allows users to highlight a specific 
object within an image by masking it.
 
Usage:
  image_masking.py [-h] [-v]
 
Keys:
  r     - mask the image
  SPACE - reset the inpainting mask
  ESC   - exit
'''
 
# Python 2/3 compatibility
from __future__ import print_function
 
import argparse
import logging

import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt
from common import Sketcher
from os.path import join
from os import listdir, mkdir
 
# Based on Image Masking Using OpenCV project by Addison Sears-Collins
# Python version: 3.7
# Description: This program allows users to highlight a specific 
# object within an image by masking it. Takes all images from the specified
# images directory.
 
dir_img = "data/imgs/"
dir_outmask = "data/masks/"

def createMask(image, mask, kernel):
    debug = False
    
    kernel2 = np.ones((3,3),np.uint8)
    bg = cv2.dilate(mask, kernel2, iterations=20)

    fg = cv2.erode(mask, None)
    
    fg = np.uint8(fg)
    unknown = cv2.subtract(bg,fg)
    
    ret, markers = cv2.connectedComponents(fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    if (debug):
        plt.imshow(markers,cmap='jet')
        plt.show()
    
    finalmask = cv2.watershed(image,markers)
    finalmask[finalmask == 1] = 0
    finalmask[finalmask == -1] = 0
    finalmask[finalmask != 0] = 255
    
    return finalmask.astype(np.uint8)

def main(viz):
    
    try:
        mkdir(dir_outmask)
        logging.info("Created {} directory".format(dir_outmask))
    except OSError:
        pass
    
    for img_file in listdir(dir_img):
        if  img_file.startswith('.'):
            continue
        
        n = False;
 
        # Load the image and store into a variable
        image = cv2.imread(join(dir_img, img_file))
     
        # Create an image for sketching the mask
        image_mark = image.copy()
        sketch = Sketcher('Image', [image_mark], lambda : ((255, 255, 255), 255))
     
        # Sketch a mask
        while True:
            ch = cv2.waitKey(100)
            if ch == 27: # ESC - exit
                n = True;
                break
            if ch == ord('r'): # r - mask the image
                break
            if ch == ord(' '): # SPACE - reset the inpainting mask
                image_mark[:] = image
                sketch.show()
            if cv2.getWindowProperty('Image',cv2.WND_PROP_VISIBLE) < 1:
                n = True
                break
            
        if n:
            cv2.destroyAllWindows()
            continue
       
        # define range of white color
        lower_white = np.array([255,255,255])
        upper_white = np.array([255,255,255])
     
        # Create base mask
        usermask = cv2.inRange(image_mark, lower_white, upper_white)
    
        # Remove the small white regions
        kernel = np.ones((3,3),np.uint8)
        usermask = cv2.morphologyEx(usermask, cv2.MORPH_OPEN, kernel)
        
        cv2.imshow('mask', usermask)
        
        out = createMask(image, usermask, kernel)
        cv2.imshow('test', out)
        
        #Save the mask
        cv2.imwrite(join(dir_outmask, img_file), out)
     
        # Display images, used for debugging
        if (viz):

            # Create the inverted mask
            mask_inv = cv2.bitwise_not(out)
         
            # Convert to grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         
            # Extract the dimensions of the original image
            rows, cols, channels = image.shape
            image = image[0:rows, 0:cols]
         
            # Bitwise-OR mask and original image
            colored_portion = cv2.bitwise_or(image, image, mask = out)
            colored_portion = colored_portion[0:rows, 0:cols]
         
            # Bitwise-OR inverse mask and grayscale image
            gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
            gray_portion = np.stack((gray_portion,)*3, axis=-1)
            
            # Combine the two images
            overlay = colored_portion + gray_portion
            
            # Create a table showing input image, mask, and overlay
            mask = np.stack((out,)*3, axis=-1)
            usermask = np.stack((usermask,)*3, axis=-1)
            
            table_of_images = np.concatenate((image, usermask, mask, overlay), axis=1)
            cv2.imshow('Input Image / User Mask / Watershed Mask / Overlay', table_of_images)
            
            while True:
                cv2.waitKey(100) # Wait for a keyboard event
                if cv2.getWindowProperty('Input Image / User Mask / Watershed Mask / Overlay',cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser(description='Image masking',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--viz', action='store_true',
                        help='Display image and created mask', dest='viz', default=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print(__doc__)
    main(viz=args.viz)