#
#  Nerve Segmentation
#  Code desgined by: Gary Milam Jr.
#  Modified Date: 02/23/2021
#  Affiliation: ART-Med Lab. (PI: Chung Hyuk Park), BME Dept., SEAS, GWU
#
#  Inspired by: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
#

""" Full assembly of the parts to form the complete network """

import torch.nn as nn
import importlib
import logging
import torch

from .unet_parts import Initial, Down, Up, OutConv, Fuse, MultiAttention, TransformerBlock

class BaseEncoder(nn.Module):
    def __init__(self, n_filter, n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl):
        super(BaseEncoder, self).__init__()
        self.inc = Initial(n_channels, n_filter[0], res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[0])
        self.down1 = Down(n_filter[0], n_filter[1], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[1])
        self.down2 = Down(n_filter[1], n_filter[2], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[2])
        self.down3 = Down(n_filter[2], n_filter[3], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[3])
        self.down4 = Down(n_filter[3], n_filter[4], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[4])
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return x1,x2,x3,x4,x5
    
class BaseDecoder(nn.Module):
    def __init__(self, n_filter, n_classes, upsampling, groups, att, t, norm_type, hwl):
        super(BaseDecoder, self).__init__()
        self.up1 = Up(n_filter[4], n_filter[3], upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[3])
        self.up2 = Up(n_filter[3], n_filter[2], upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[2])
        self.up3 = Up(n_filter[2], n_filter[1], upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[1])
        self.up4 = Up(n_filter[1], n_filter[0], upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[0])
        self.outc = OutConv(n_filter[0], n_classes, norm_type)
        
    def forward(self, x1,x2,x3,x4,x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class BaseUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_f, upsampling='bilinear', drop_rate=0, groups=1, att=0, 
                 res=False, t=0, deep_supervision=False, split_gpu=False, ibt=0, norm_type=0, hwd=256):
        super(BaseUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upsampling = upsampling
        self.n_f = n_f
        self.split_gpu = split_gpu
        assert not deep_supervision, 'Deep supervision only available in UNet2P'
        self.deep_supervision = deep_supervision
        
        n_filter = [self.n_f, self.n_f*2, self.n_f*4, self.n_f*8, self.n_f*16]
        hwl = [int(hwd), int(hwd/2), int(hwd/4), int(hwd/8), int(hwd/16)]
        
        if self.split_gpu:
            self.encoder = BaseEncoder(n_filter, self.n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl).to('cuda:0')
            self.decoder = BaseDecoder(n_filter, self.n_classes, self.upsampling, groups, att, t, norm_type, hwl).to('cuda:1')
        else:
            self.encoder = BaseEncoder(n_filter, self.n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl)
            self.decoder = BaseDecoder(n_filter, self.n_classes, self.upsampling, groups, att, t, norm_type, hwl)
        
        #self.apply(init_weights) #To be used for testing once settled with a model.
        
    def forward(self, x):
        if self.split_gpu:
            x = [t.to('cuda:1') for t in self.encoder(x)]
        else:
            x = self.encoder(x)
        
        logits = self.decoder(x[0],x[1],x[2],x[3],x[4])
        return logits

class UNet(BaseUNet):
    def __init__(self, n_channels, n_classes, n_f, upsampling='bilinear', drop_rate=0, groups=1, res=False, t=0, 
                 att=0, split_gpu=False, ibt=0, norm_type=0, hwd=256, **kwargs):
        super().__init__(n_channels, n_classes, n_f, upsampling=upsampling, drop_rate=drop_rate, groups=groups, 
                         res=res, t=t, att=att, split_gpu=split_gpu, ibt=ibt, norm_type=norm_type, hwd=hwd)

class AttUNet(BaseUNet):
    def __init__(self, n_channels, n_classes, n_f, upsampling='bilinear', drop_rate=0, groups=1, res=False, t=0,
                 att=1, split_gpu=False, ibt=0, norm_type=0, hwd=256, **kwargs):
        assert att > 0, f'Config pass att as {att}. This should be set to an integer greater than 0 for AttUNet.'
        super().__init__(n_channels, n_classes, n_f, upsampling=upsampling, drop_rate=drop_rate, groups=groups,
                         res=res, t=t, att=att, split_gpu=split_gpu, ibt=ibt, norm_type=norm_type, hwd=hwd)
    
class ResUNet(BaseUNet):
    def __init__(self, n_channels, n_classes, n_f, upsampling='bilinear', drop_rate=0, groups=1, res=True, t=0,
                 att=0, split_gpu=False, ibt=0, norm_type=0, hwd=256, **kwargs):
        if not res:
            logging.warn(f'Config pass res as {res}. This should be set to True for ResUNet.'
                         ' This will be ignored but best to correct config.')
        super().__init__(n_channels, n_classes, n_f, upsampling=upsampling, drop_rate=drop_rate, groups=groups,
                         res=True, t=t, att=att, split_gpu=split_gpu, ibt=ibt, norm_type=norm_type, hwd=hwd)

class R2UNet(BaseUNet):
    def __init__(self, n_channels, n_classes, n_f, upsampling='bilinear', drop_rate=0, groups=1, res=False, t=2, 
                 att=0, split_gpu=False, ibt=0, norm_type=0, hwd=256, **kwargs):
        assert t > 0, f'Config pass t as {t}. For R2Unet this should be set with a value greater than 0, correct config with intended value.'
        super().__init__(n_channels, n_classes, n_f, upsampling=upsampling, drop_rate=drop_rate, groups=groups,
                         res=res, t=t, att=att, split_gpu=split_gpu, ibt=ibt, norm_type=norm_type, hwd=hwd)

#Need to update for configurable normalization
class UNet2P(nn.Module):
    def __init__(self, n_channels, n_classes, n_f, upsampling='transpose', att=0, deep_supervision=False, groups=1, **kwargs):
        super(UNet2P, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upsampling = upsampling
        self.deep_supervision = deep_supervision
        self.n_f = n_f
        
        #Not currently handled.
        self.split_gpu = False 
        
        n_filter = [n_f, n_f*2, n_f*4, n_f*8, n_f*16]
        
        self.inc = Initial(n_channels, n_filter[0])
        self.down1_0 = Down(n_filter[0], n_filter[1], groups=groups)
        self.up0_1 =  Up(n_filter[1], n_filter[0], 1, upsampling=upsampling, att=att, groups=groups)
        
        
        self.down2_0 = Down(n_filter[1], n_filter[2])
        self.up1_1 = Up(n_filter[2], n_filter[1], 1, upsampling=upsampling, att=att, groups=groups)
        self.up0_2 =  Up(n_filter[1], n_filter[0], 2, upsampling=upsampling, att=att, groups=groups)
        
        self.down3_0 = Down(n_filter[2], n_filter[3], groups=groups)
        self.up2_1 = Up(n_filter[3], n_filter[2], 1, upsampling=upsampling, att=att, groups=groups)
        self.up1_2 = Up(n_filter[2], n_filter[1], 2, upsampling=upsampling, att=att, groups=groups)
        self.up0_3 = Up(n_filter[1], n_filter[0], 3, upsampling=upsampling, att=att, groups=groups)
        
        self.down4_0 = Down(n_filter[3], n_filter[4])
        self.up3_1 = Up(n_filter[4], n_filter[3], 1, upsampling=upsampling, att=att, groups=groups)
        self.up2_2 = Up(n_filter[3], n_filter[2], 2, upsampling=upsampling, att=att, groups=groups)
        self.up1_3 = Up(n_filter[2], n_filter[1], 3, upsampling=upsampling, att=att, groups=groups)
        self.up0_4 = Up(n_filter[1], n_filter[0], 4, upsampling=upsampling, att=att, groups=groups)
        
        if self.deep_supervision:
            self.outc1 = OutConv(n_filter[0], n_classes)
            self.outc2 = OutConv(n_filter[0], n_classes)
            self.outc3 = OutConv(n_filter[0], n_classes)
            self.outc4 = OutConv(n_filter[0], n_classes)
        else:   
            self.outc = OutConv(n_filter[0], n_classes)
            
        #self.apply(init_weights) #To be used for testing once settled with a model.
        
    def forward(self, x):
        x0_0 = self.inc(x)
        x1_0 = self.down1_0(x0_0)
        x0_1 = self.up0_1(x1_0, x0_0)
        
        x2_0 = self.down2_0(x1_0)
        x1_1 = self.up1_1(x2_0, x1_0)
        x0_2 = self.up0_2(x1_1, x0_1, x0_0)
        
        x3_0 = self.down3_0(x2_0)
        x2_1 = self.up2_1(x3_0, x2_0)
        x1_2 = self.up1_2(x2_1, x1_1, x1_0)
        x0_3 = self.up0_3(x1_2, x0_2, x0_1, x0_0)
        
        x4_0 = self.down4_0(x3_0)
        x3_1 = self.up3_1(x4_0, x3_0)
        x2_2 = self.up2_2(x3_1, x2_1, x2_0)
        x1_3 = self.up1_3(x2_2, x1_2, x1_1, x1_0)
        x0_4 = self.up0_4(x1_3, x0_3, x0_2, x0_1, x0_0)
        
        if self.deep_supervision:
            out1 = self.outc1(x0_1)
            out2 = self.outc2(x0_2)
            out3 = self.outc3(x0_3)
            out4 = self.outc4(x0_4)
            return [out1, out2, out3, out4]
            
        logits = self.outc(x0_4)
        return logits

class DualBaseEncoder(nn.Module):
    def __init__(self, n_filter, n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl):
        super(DualBaseEncoder, self).__init__()
        self.inc = Initial(n_channels, n_filter[0], res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[0])
        self.down1 = Down(n_filter[0], n_filter[1], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[1])
        self.down2 = Down(n_filter[1], n_filter[2], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[2])
        self.down3 = Down(n_filter[2], n_filter[3], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[3])
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        return x1,x2,x3,x4

class DualFuseAllEncoder(nn.Module):
    def __init__(self, n_filter, n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl):
        super(DualFuseAllEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        
        self.inca = Initial(n_channels, n_filter[0], res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[0])
        self.incb = Initial(n_channels, n_filter[0], res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[0])
        
        self.downa_1 = Down(n_filter[0]*2, n_filter[1], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[1], add_maxpool=False)
        self.downa_2 = Down(n_filter[1]*2, n_filter[2], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[2], add_maxpool=False)
        self.downa_3 = Down(n_filter[2]*2, n_filter[3], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[3], add_maxpool=False)
        
        self.downb_1 = Down(n_filter[0]*2, n_filter[1], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[1], add_maxpool=False)
        self.downb_2 = Down(n_filter[1]*2, n_filter[2], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[2], add_maxpool=False)
        self.downb_3 = Down(n_filter[2]*2, n_filter[3], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[3], add_maxpool=False)

    def forward(self, x, x2):
        xa_1 = self.inca(x)
        xb_1 = self.incb(x2)
        
        xa_1m = self.maxpool(xa_1)
        xb_1m = self.maxpool(xb_1)
        
        xa_2 = self.downa_1(torch.cat((xa_1m,xb_1m), dim=1))
        xb_2 = self.downb_1(torch.cat((xb_1m,xa_1m), dim=1))
        
        xa_2m = self.maxpool(xa_2)
        xb_2m = self.maxpool(xb_2)
        
        xa_3 = self.downa_2(torch.cat((xa_2m,xb_2m), dim=1))
        xb_3 = self.downb_2(torch.cat((xb_2m,xa_2m), dim=1))
        
        xa_3m = self.maxpool(xa_3)
        xb_3m = self.maxpool(xb_3)
        
        xa_4 = self.downa_3(torch.cat((xa_3m,xb_3m), dim=1))
        xb_4 = self.downb_3(torch.cat((xb_3m,xa_3m), dim=1))
        
        return (xa_1,xa_2,xa_3,xa_4), (xb_1,xb_2,xb_3,xb_4)

class DualBaseDecoder(nn.Module):
    def __init__(self, n_filter, n_classes, upsampling, groups, att, t, norm_type, hwl):
        super(DualBaseDecoder, self).__init__()
        self.up1 = Up(n_filter[4], n_filter[3], l=2, upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[3])
        self.up2 = Up(n_filter[3], n_filter[2], l=2, upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[2])
        self.up3 = Up(n_filter[2], n_filter[1], l=2, upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[1])
        self.up4 = Up(n_filter[1], n_filter[0], l=2, upsampling=upsampling, groups=groups, att=att, t=t, norm_type=norm_type, hw=hwl[0])
        self.outc = OutConv(n_filter[0], n_classes, norm_type)
        
    def forward(self, x1a, x1b, x2a, x2b, x3a, x3b, x4a, x4b, x5):
        x = self.up1(x5, x4b, x4a)
        x = self.up2(x, x3b, x3a)
        x = self.up3(x, x2b, x2a)
        x = self.up4(x, x1b, x1a)
        logits = self.outc(x)
        return logits

class CoLearnDecoder(DualBaseDecoder):
    def __init__(self, n_filter, n_classes, upsampling, groups, att, t, norm_type, hwl):
        super().__init__(n_filter, n_classes, upsampling, groups, att, t, norm_type, hwl)
        
    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, *_split(x4))
        x = self.up2(x, *_split(x3))
        x = self.up3(x, *_split(x2))
        x = self.up4(x, *_split(x1))
        logits = self.outc(x)
        return logits
    
class DualUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_f, upsampling='bilinear', drop_rate=0, groups=1, att=0, 
                 res=False, t=0, deep_supervision=False, split_gpu=False, co_learn=0, ibt=0, norm_type=0, 
                 hwd=256, multiatt=0, num_heads=4, tfb=0, tnh=4, tr=False, tfbm=False, tfbres=False, 
                 interconnect=False, **kwargs):
        super(DualUNet, self).__init__()
        
        i = 0
        if co_learn > 0:
            i = i + 1
        if multiatt > 0:
            i = i + 1
        if tfb:
            i = i + 1
            
        assert i < 2, f'Only a single fusion approach can be used at a given time. Currently using {i} approaches. \
            Following are fusion approach values. co_learn: {co_learn}, multiatt: {multiatt}, TransformBlock (tfb) {tfb}.'
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upsampling = upsampling
        self.n_f = n_f
        self.split_gpu = split_gpu  #Still needed as used in main training script
        assert not self.split_gpu, 'Split GPU not implemented for DualUNet'
        self.co_learn = co_learn
        assert not deep_supervision, 'Deep supervision only available in UNet2P'
        self.deep_supervision = deep_supervision
        self.multiatt = multiatt
        self.tfb = tfb
        
        #Following is used when tfb == 2
        self.tfbm = tfbm 
        self.tfbres = tfbres 
        
        self.interconnect = interconnect
        
        assert ibt < 4, 'Only three variants of inception block are available. Change ibt value {ibt} to one that exists.'
       
        n_filter = [self.n_f, self.n_f*2, self.n_f*4, self.n_f*8, self.n_f*16]
        hwl = [int(hwd), int(hwd/2), int(hwd/4), int(hwd/8), int(hwd/16)]

        if self.interconnect:
            self.encoder = DualFuseAllEncoder(n_filter, self.n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl)
        else:
            self.encoder1 = DualBaseEncoder(n_filter, self.n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl)
            self.encoder2 = DualBaseEncoder(n_filter, self.n_channels, drop_rate, groups, res, t, ibt, norm_type, hwl)
    
        in_size = n_filter[3]*2 #input size for bridge
        
        if self.co_learn == 2: #Applying co_learn module to all layers
            self.fuse1 = Fuse(n_filter[0], n_filter[0]*2)
            self.fuse2 = Fuse(n_filter[1], n_filter[1]*2)
            self.fuse3 = Fuse(n_filter[2], n_filter[2]*2)
            self.fuse4 = Fuse(n_filter[3], n_filter[3]*2)
            self.decoder = CoLearnDecoder(n_filter, self.n_classes, self.upsampling, groups, att, t, norm_type, hwl)
        else:
            if self.co_learn == 1: #Only adding co_learn module prior to bottleneck
                self.fuse4 = Fuse(n_filter[3], n_filter[3]*2, norm_type)
            elif self.tfb == 1:
                self.fuse4 = TransformerBlock(n_filter[3], tnh)
                self.reverse = tr #Which modality is attending the other. 
                in_size = n_filter[3]
            elif self.tfb == 2:
                self.tfuse = TransformerBlock(n_filter[3], tnh)
                self.tfuse2 = TransformerBlock(n_filter[3], tnh)
                if self.tfbm:
                    in_size = n_filter[3]
            elif self.multiatt > 0:
                self.multi_attention1 = MultiAttention(n_filter[3],num_heads)
                self.multi_attention2 = MultiAttention(n_filter[3],num_heads)
                if self.multiatt > 1:
                    self.multi_attention3 = MultiAttention(n_filter[3],num_heads)
                    in_size = n_filter[3]
            self.decoder = DualBaseDecoder(n_filter, self.n_classes, self.upsampling, groups, att, t, norm_type, hwl)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.bridge =  Down(in_size, n_filter[4], drop_rate=drop_rate, groups=groups, res=res, t=t, ibt=ibt, norm_type=norm_type, hw=hwl[4], add_maxpool=False)
        #self.apply(init_weights) #To be used for testing once settled with a model.

    def forward(self, x_a, x_b):

        if self.interconnect:
            x_a, x_b = self.encoder(x_a, x_b)
        else:
            x_a = self.encoder1(x_a)
            x_b = self.encoder2(x_b)

        #If improves make change to rest of fusion
        x_a3m = self.maxpool(x_a[3])
        x_b3m = self.maxpool(x_b[3])

        if self.co_learn == 2:
            #Incorrect implementation. Should be done after max pooling
            x1 = self.fuse1(x_b[0],x_a[0])
            x2 = self.fuse2(x_b[1],x_a[1])
            x3 = self.fuse3(x_b[2],x_a[2])
            x4 = self.fuse4(x_b[3],x_a[3])
        elif self.co_learn == 1:
            #x4 = self.fuse4(x_b[3],x_a[3]) #Originally
            x4 = self.fuse4(x_a3m,x_b3m)
        elif self.tfb == 1:
            if self.reverse:
                x4 = self.fuse4(x_b[3],x_a[3])
            else:
                x4 = self.fuse4(x_a[3],x_b[3])
        elif self.tfb == 2:
            y = self.tfuse(x_a3m,x_b3m)
            y2 = self.tfuse2(x_b3m,x_a3m)
            if self.tfbres:
                y = y + x_b3m
                y2 = y2 + x_a3m
            if self.tfbm:
                x4 = torch.mul(y,y2)
            else:
                x4 = torch.cat((y,y2), dim=1)
        elif self.multiatt > 0:
            x_m1 = self.multi_attention1(x_a[3])
            x_m2 = self.multi_attention2(x_b[3])
            if self.multiatt > 1:
                x4 = self.multi_attention3(x_m1, x_m2)
            else:
                x4 = torch.cat((x_m2,x_m1), dim=1)
        else:
            x4 = torch.cat((x_a3m,x_b3m), dim=1)
            #x4 = torch.cat((x_b[3],x_a[3]), dim=1) #Previous Implementation

        xb = self.bridge(x4)
        
        if self.co_learn == 2:
            logits = self.decoder(x1, x2, x3, x4, xb)
        else:
            logits = self.decoder(x_a[0],x_b[0],x_a[1],x_b[1],x_a[2],x_b[2],x_a[3],x_b[3],xb)
            
        return logits

def _split(x):
    return(torch.split(x,x.shape[1]//2,dim=1))

def get_model(config):
    def _model(classname):
        return getattr(importlib.import_module('unet.unet_model'), classname)
    
    assert 'model' in config, 'Model configuration not defined.'
    model_config = config['model']
    model = _model(model_config['arch'])
    imgd = config['loaders'].get('img_size', 256)
    model_config.update({'hwd': imgd})
    return model(**model_config)