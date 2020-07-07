#!/usr/bin/env python
 
'''
Module for sketching
'''

import cv2
import numpy as np

def maskOverlay(mask, img):    
    mask_inv = cv2.bitwise_not(mask)
    bg =  cv2.bitwise_and(img, img, mask = mask_inv)
    output = cv2.add(bg, np.stack((mask,)*3, axis=-1)) 
    return output
        
class Sketcher:
    def __init__(self, windowname, dests, color = 255):
        self.size = 3
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.color = color
        self.show()
        self.sizetrackname = 'Line size'
        cv2.createTrackbar(self.sizetrackname, self.windowname, self.size, 20, self.on_sizetrackbar)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
 
    def show(self):
        output = maskOverlay(self.dests[1], self.dests[0])
        cv2.imshow(self.windowname, output)
 
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                if flags > 0:
                    self.size += 1
                else:
                    self.size -= 1
                cv2.setTrackbarPos(self.sizetrackname, self.windowname, self.size)
            else:
                self.mousewheel_events(flags)                
        elif event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.prev_pt = None
        
        if self.prev_pt:
            if flags == cv2.EVENT_FLAG_LBUTTON:
                color = self.color
            elif flags  == cv2.EVENT_FLAG_RBUTTON:
                color = 0
            else:
                return
            
            cv2.line(self.dests[1], self.prev_pt, pt, color, self.size)
            self.prev_pt = pt
            self.show()
    
    def on_sizetrackbar(self, val):
        self.size = val
        
    def mousewheel_events(self, flags):
        #Placeholder for mouse wheel events that exist on other classes
        return
        
class DilatedSketcher(Sketcher):
    def __init__(self, windowname, dests):
        super().__init__(windowname, dests, color = 128)
        self.dests[0] = maskOverlay(self.dests[2], self.dests[0])
        self.iter_val = 15
        self.diltrackname = 'Dilated Iterations'
        cv2.createTrackbar(self.diltrackname, self.windowname, self.iter_val, 30, self.on_diltrackbar)
        self.show()
    
    def show(self):
        unknown = cv2.subtract(self.dests[1], self.dests[2])
        unknown[unknown > 0] = self.color
        output = maskOverlay(unknown, self.dests[0])
        cv2.imshow(self.windowname, output)

    def on_diltrackbar(self, val):
        if val > self.iter_val:
            self.dests[1] = cv2.dilate(self.dests[1], None, iterations=(val - self.iter_val))
        else:      
            self.dests[1] = cv2.erode(self.dests[1], None, iterations=(self.iter_val - val))

        self.iter_val = val
        self.show()
    
    def mousewheel_events(self, flags):
        if flags & cv2.EVENT_FLAG_ALTKEY:     
            if flags > 0:
                self.dests[1] = cv2.dilate(self.dests[1], None, iterations=1)
                self.iter_val += 1
            else:
                self.dests[1] = cv2.erode(self.dests[1], None, iterations=1)
                self.iter_val -= 1
            cv2.setTrackbarPos(self.diltrackname, self.windowname, self.iter_val)