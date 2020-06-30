#!/usr/bin/env python
 
'''
Module for sketching
'''

import cv2
import numpy as np
        
class Sketcher:
    def __init__(self, windowname, dests):
        self.max_slider_size = 20
        self.size = 3
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.show()
        cv2.createTrackbar('Line thickness', self.windowname, self.size, self.max_slider_size, self.on_trackbar)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
 
    def show(self):
        mask_inv = cv2.bitwise_not(self.dests[1])
        bg =  cv2.bitwise_and(self.dests[0], self.dests[0], mask = mask_inv)
        output = cv2.add(bg, np.stack((self.dests[1],)*3, axis=-1))      
        cv2.imshow(self.windowname, output)
 
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.prev_pt = None
        
        if self.prev_pt:
            if flags == cv2.EVENT_FLAG_LBUTTON:
                color = 255
            elif flags  == cv2.EVENT_FLAG_RBUTTON:
                color = 0
            else:
                print("On mouse flag not recognized")
                return
            
            cv2.line(self.dests[1], self.prev_pt, pt, color, self.size)
            self.prev_pt = pt
            self.show()
    
    def on_trackbar(self, val):
        self.size = val