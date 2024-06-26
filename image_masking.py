#!/usr/bin/env python
 
'''
Welcome to the Image Masking Program!
 
This program allows users to highlight a specific 
object within an image by masking it.
 
Keys:
  f     - consider as final mask
  w     - mask the image and apply watershed
  s     - save current sketching state
  n     - go to next image
  SPACE - reset the inpainting mask
  ESC   - exit
'''
 
# Python 2/3 compatibility
from __future__ import print_function
 
import argparse
import logging
import pickle
import cv2
import fnmatch
import yaml
import numpy as np

from common import Sketcher, DilatedSketcher
from prediction import NerveMask
from os.path import join, splitext, exists
from os import listdir, mkdir, remove
from sys import exit
 
# Based on Image Masking Using OpenCV project by Addison Sears-Collins
# Python version: 3.7
# Description: This program allows users to highlight a specific 
# object within an image by masking it. 
#Takes all images from the specified images directory. If a mask exists for such
#image it will use this as the base mask which user can modify.
 
state_file = "state.data"

def main(proc_imgs, dir_img, dir_outmask, debug=False, skip=False, ffilter=None, window_size=None, predict=True):
    
    assert dir_img is not None and dir_outmask is not None, f'Both dir_img and dir_outmask should not be set to None. Check config.yaml'
    
    dir_img_jet = join(dir_img, "jet")
    dir_img_rgb = join(dir_img, "rgb")
    
    global size
    if window_size is not None:
        size = window_size
    else:
        size = {'width': 512, 'height':512}
    
    try:
        mkdir(dir_outmask)
        masks = []
        logging.info("Created {} directory".format(dir_outmask))
    except OSError:
        masks =  [file for file in listdir(dir_outmask)
                    if not file.startswith('.')]
        pass
    
    img_files = [file for file in listdir(dir_img_jet)
                    if not file.startswith('.')
                    and (skip or not file in proc_imgs)
                    and (ffilter is None or fnmatch.fnmatch(file,ffilter))]
    
    if predict:
        #Inititate network for inference
        nerve_mask = NerveMask()

        
    for img_file in img_files:
        
        mask = None
        if masks and img_file in masks:
            mask = cv2.imread(join(dir_outmask, img_file), cv2.IMREAD_GRAYSCALE)
            
            
        # Load the images and store into a variables
        image = cv2.imread(join(dir_img_jet, img_file))
        image2 = cv2.imread(join(dir_img_rgb, img_file))
        
        if image is None or image2 is None:
            print(f'Both the RGB and Jet images need to exist for image masking. Please ensure this is the case for {img_file}. Moving on to next file.')
            continue
        
        if predict:
            #Make sure images are passed in RGB format
            mask = nerve_mask.get_mask(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)) * np.uint8(255) #birefrigence and rgb images passed
            
        elif mask is not None:
            img_height, img_width, __ = image.shape
            m_height, m_width = mask.shape
        
            if img_height != m_height or img_width != m_width:
                mask = cv2.resize(mask,(img_width,img_height))
                
            __, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) #Ensure maintains binary
        
        f_n = splitext(img_file)[0]
     
        image_mask, n, ws = sketchMask(image, image2, f_n, mask = mask)
        
        if n:
            continue

        out = createMask(image, image2, image_mask, debug, ws)
        
        kernel = np.ones((3,3),np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel) # removing noise
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel) #closing holes
        
        if debug:
            cv2.imshow('Mask', out)
     
        # Display images, used for refining the mask
        window_name = displayTable(image, out, image_mask) if ws else displayTable(image, out) 
        
        logging.info("If wish to refine mask press the 'r' key. Otherwise press 'n' for next image")
        while True:
            ch = cv2.waitKey(100) # Wait for a keyboard event
            if ch == 27: # ESC - exit
                cv2.destroyAllWindows()
                exit()
            if ch == ord('r'):
                out, n, ws = sketchMask(image, f_n, mask = out)
                out = createMask(image, out, debug, ws)              
                window_name = displayTable(image, out)
            if ch == ord('n'): #  or cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        
        #Save the mask
        cv2.imwrite(join(dir_outmask, img_file), out) #Maybe look at having it save as PNG.
        
        #Save state of processed images
        proc_imgs.add(img_file)
        with open(state_file, 'wb') as fp:
            pickle.dump(proc_imgs,fp)
    
    logging.info("No further images to be processed.\n"
                     "If wish to reset state run the program with -r argument or to ignore state with -s argument.")

def createMask(image, image2, mask, debug, ws):
    
    if not(ws):
      return mask.astype(np.uint8)         
    
    bg = cv2.dilate(mask, None, iterations=15)
    fg = cv2.erode(mask, None)
    bg2 = sketchMask(image, image2, "DilatedMask", bg, fg)[0]
    
    unknown = cv2.subtract(bg2,fg)
    
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

def sketchMask(image, image2, image_name, mask = None, fg = None):
    
    orig_mask = mask if mask is not None else np.zeros((image.shape[0], image.shape[1]), image.dtype)
    image_mask = orig_mask.copy()
    
    # Sketch a mask
    if fg is None:
        sketch = Sketcher(image_name, size, [image, image2, image_mask])
    # Sketch dilated mask
    else:
        sketch = DilatedSketcher(image_name, size, [image, image2, image_mask, fg])
        
    ws = False
    n = False
    while True:
        ch = cv2.waitKey(100)
        if ch == 27: # ESC - exit
            cv2.destroyAllWindows()
            exit()
        if ch == ord('f'): # f - consider as final mask for the image
            break
        if ch == ord('w') and fg is None: # w - mask the image and apply watershed
            ws = True
            break
        if ch == ord('s'): # s - save current sketching state
            orig_mask[:] = image_mask
            sketch.show()
        if ch == ord(' '): # SPACE - reset the inpainting mask
            image_mask[:] = orig_mask
            sketch.show()
        if ch == ord('n'): #or cv2.getWindowProperty(image_name,cv2.WND_PROP_VISIBLE) < 1:
            n = True
            break
    
    cv2.destroyAllWindows()
    
    return sketch.dests[1], n, ws

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
        showWindowTable(table_of_images, window_name, 4)
        return window_name
    else:
        table_of_images = np.concatenate((image, mask, overlay), axis=1)
        window_name = 'Input Image / Mask / Overlay'
        showWindowTable(table_of_images, window_name, 3)
        return window_name

def showWindowTable(table_of_images, window_name, num_contents):
    cv2.namedWindow(window_name,flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, size['width']*num_contents, size['height'])
    cv2.imshow(window_name, table_of_images)

def get_args():
    parser = argparse.ArgumentParser(description='Image masking',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enables debugging mode', dest='debug', default=False)
    parser.add_argument('-s', '--skip_state', action='store_true',
                        help='Ignores list of processed images', dest='skip', default=False)
    parser.add_argument('-r', '--reset_state', action='store_true',
                        help='Resets state of processed images', dest='reset', default=False)
    parser.add_argument('-c', '--configfile', type=str,
                        help='Configuration file', dest='config', default='config.yaml')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    if exists(args.config):
        config = yaml.safe_load(open(args.config, 'r'))
    else:
        config = {}
        
    img_list = set()
    if exists(state_file):
        if args.reset:
            remove(state_file)
        else:
            with open(state_file, 'rb') as fr:
                img_list = pickle.load(fr)
    
    print(__doc__)
    
    if args.debug:
        import matplotlib.pyplot as plt
    
    main(img_list, debug=args.debug, skip=args.skip, **config)