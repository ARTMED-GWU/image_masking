#!/usr/bin/env python
 
'''
Welcome to the Image Masking Program!
 
This program allows users to highlight a specific 
object within an image by masking it.
 
Keys:
  f     - consider as final mask
  w     - mask the image and apply watershed
  n     - go to next image
  SPACE - reset the inpainting mask
  ESC   - exit
'''
 
# Python 2/3 compatibility
from __future__ import print_function
 
import argparse
import logging
import pickle

import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt
from common import Sketcher
from os.path import join, splitext, exists
from os import listdir, mkdir, remove
 
# Based on Image Masking Using OpenCV project by Addison Sears-Collins
# Python version: 3.7
# Description: This program allows users to highlight a specific 
# object within an image by masking it. 
#Takes all images from the specified images directory. If a mask exists for such
#image it will use this as the base mask which user can modify.
 
dir_img = "../data/imgs/"
dir_outmask = "../data/masks/"
state_file = "state.data"

def createMask(image, mask, debug, ws):
    
    if not(ws):
      return mask.astype(np.uint8)         
    
    kernel = np.ones((3,3),np.uint8)
    bg = cv2.dilate(mask, kernel, iterations=15)

    fg = cv2.erode(mask, None)
    
    fg = np.uint8(fg)
    unknown = cv2.subtract(bg,fg)
    
    markers = fg.astype('int32')
    markers[markers==255] = 1
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

def sketchMask(image, image_name, mask = None):
    
    orig_mask = mask if mask is not None else np.zeros((image.shape[0], image.shape[1]), image.dtype)
    image_mask = orig_mask.copy() 
    
    # Sketch a mask
    sketch = Sketcher(image_name, [image, image_mask])
    ws = False
    n = False
    while True:
        ch = cv2.waitKey(100)
        if ch == 27: # ESC - exit
            exit()
        if ch == ord('f'): # f - consider as final mask for the image
            break
        if ch == ord('w'): # w - mask the image and apply watershed
            ws = True
            break
        if ch == ord('s'): # s - save current sketching state
            orig_mask[:] = image_mask
            sketch.show()
        if ch == ord(' '): # SPACE - reset the inpainting mask
            image_mask[:] = orig_mask
            sketch.show()
        if ch == ord('n') or cv2.getWindowProperty(image_name,cv2.WND_PROP_VISIBLE) < 1:
            n = True
            break
    
    cv2.destroyAllWindows()
    
    return image_mask, n, ws

def displayTable(image, mask, usermask = None):
    
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
    
    # Create a table showing input image, usermask if exist, mask, and overlay
    mask = np.stack((mask,)*3, axis=-1)
    
    if usermask is not None:
        usermask = np.stack((usermask,)*3, axis=-1)
        table_of_images = np.concatenate((image, usermask, mask, overlay), axis=1)
        window_name = 'Input Image / User Mask / Watershed Mask / Overlay'
        cv2.imshow(window_name, table_of_images)
        return window_name
    else:
        table_of_images = np.concatenate((image, mask, overlay), axis=1)
        window_name = 'Input Image / Mask / Overlay'
        cv2.imshow(window_name, table_of_images)
        return window_name

def main(proc_imgs, debug=False, skip=False):
    
    try:
        mkdir(dir_outmask)
        masks = []
        logging.info("Created {} directory".format(dir_outmask))
    except OSError:
        masks =  [file for file in listdir(dir_outmask)
                    if not file.startswith('.')]
        pass
    
    img_files = [file for file in listdir(dir_img)
                    if not file.startswith('.') and (skip or not file in proc_imgs) ]
    
    for img_file in img_files:
        
        mask = None
        if masks and img_file in masks:
            mask = cv2.imread(join(dir_outmask, img_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512,512))
            
        # Load the image and store into a variable
        image = cv2.imread(join(dir_img, img_file))
        image = cv2.resize(image, (512,512)) #Use scaling instead of constant.
        f_n = splitext(img_file)[0]
     
        image_mask, n, ws = sketchMask(image, f_n, mask = mask)
        
        if n:
            continue

        out = createMask(image, image_mask, debug, ws)
        
        if debug:
            cv2.imshow('Mask', out)
     
        # Display images, used for refining the mask
        window_name = displayTable(image, out, image_mask) if ws else displayTable(image, out) 
        
        print("If wish to refine mask press the 'r' key. Otherwise press 'n' for next image")
        while True:
            ch = cv2.waitKey(100) # Wait for a keyboard event
            if ch == 27: # ESC - exit
                exit()
            if ch == ord('r'):
                out, n, ws = sketchMask(image, f_n, mask = out)
                out = createMask(image, out, debug, ws)                
                window_name = displayTable(image, out)
            if ch == ord('n') or cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        
        #Save the mask
        cv2.imwrite(join(dir_outmask, img_file), out)
        
        #Save state of processed images
        proc_imgs.add(img_file)
        with open(state_file, 'wb') as fp:
            pickle.dump(proc_imgs,fp)

def get_args():
    parser = argparse.ArgumentParser(description='Image masking',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enables debugging mode', dest='debug', default=False)
    parser.add_argument('-s', '--skip_state', action='store_true',
                        help='Ignores list of processed images', dest='skip', default=False)
    parser.add_argument('-r', '--reset_state', action='store_true',
                        help='Resets state of processed images', dest='reset', default=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    img_list = set()
    if exists(state_file):
        if args.reset:
            remove(state_file)
        else:
            with open(state_file, 'rb') as fr:
                img_list = pickle.load(fr)
    
    print(__doc__)
    
    main(img_list, debug=args.debug, skip=args.skip)