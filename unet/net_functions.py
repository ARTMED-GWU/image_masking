# -*- coding: utf-8 -*-
"""
Network functions
"""

import torch.nn as nn
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
        if m.bias is not None:          
            '''fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)'''
            init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1.0)
        init.constant_(m.bias, 0.0)