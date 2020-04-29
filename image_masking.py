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
     
        # Create the mask
        mask = cv2.inRange(image_mark, lower_white, upper_white)
    
        # Remove the small white regions
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        #Save the mask
        cv2.imwrite(join(dir_outmask, img_file), mask)
     
        # Display images, used for debugging
        if (viz):

            # Create the inverted mask
            mask_inv = cv2.bitwise_not(mask)
         
            # Convert to grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         
            # Extract the dimensions of the original image
            rows, cols, channels = image.shape
            image = image[0:rows, 0:cols]
         
            # Bitwise-OR mask and original image
            colored_portion = cv2.bitwise_or(image, image, mask = mask)
            colored_portion = colored_portion[0:rows, 0:cols]
         
            # Bitwise-OR inverse mask and grayscale image
            gray_portion = cv2.bitwise_or(gray, gray, mask = mask_inv)
            gray_portion = np.stack((gray_portion,)*3, axis=-1)
            
            # Combine the two images
            overlay = colored_portion + gray_portion
            
            # Create a table showing input image, mask, and overlay
            mask = np.stack((mask,)*3, axis=-1)
            table_of_images = np.concatenate((image, mask, overlay), axis=1)
            
            cv2.imshow('Table of Images', table_of_images)
            
            while True:
                cv2.waitKey(100) # Wait for a keyboard event
                if cv2.getWindowProperty('Table of Images',cv2.WND_PROP_VISIBLE) < 1:
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