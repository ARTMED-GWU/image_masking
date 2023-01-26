#
#  Nerve Segmentation
#  Code modified by: Gary Milam Jr.
#  Modified Date: 02/23/2021
#  Affiliation: ART-Med Lab. (PI: Chung Hyuk Park), BME Dept., SEAS, GWU
#
#  Inspired by: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
#

""" Parts of the models """

import torch
import torch.nn as nn

def _get_normalization(norm_type, channels, hw=None):
    if norm_type == 0:
        return nn.BatchNorm2d(channels)
    elif norm_type == 1:
        return nn.InstanceNorm2d(channels)
    elif norm_type == 2:
        assert hw is not None, "Must provide height and width dimension of input"
        return nn.LayerNorm((channels,hw,hw))
    else:
        raise ValueError(f"Norm_type value {norm_type} not supported. Only 0 for BatchNorm, 1 for InstanceNorm, 2 for LayerNorm")

'''From https://github.com/josedolz/IVD-Net/blob/490d9b6c4f9a6a662bbcda6d7b2593bf0b3222c4/Blocks.py'''
def _conv_block_Asym_Inception(in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=0, hw=None):
    if norm_type == -1:
        model = nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size,1], padding=tuple([padding,0]), dilation = (dilation,1))),
        nn.ReLU(),
        nn.utils.weight_norm(nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0,padding]), dilation = (1,dilation))),
        nn.ReLU(inplace=True)
        )
    else:          
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=[kernel_size,1], padding=tuple([padding,0]), dilation = (dilation,1)),
            _get_normalization(norm_type,out_dim,hw),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=[1, kernel_size], padding=tuple([0,padding]), dilation = (1,dilation)),
            _get_normalization(norm_type,out_dim,hw),
            nn.ReLU(inplace=True)
        )
    return model

'''From https://github.com/josedolz/IVD-Net/blob/490d9b6c4f9a6a662bbcda6d7b2593bf0b3222c4/Blocks.py'''
def _conv_block(in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=0, hw=None):
    if norm_type == -1:
        model = nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation )),
        nn.ReLU(inplace=True)
        )
    else:    
        model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation ),
            _get_normalization(norm_type,out_dim,hw),
            nn.ReLU(inplace=True)
        )
        
    return model

'''Based from https://github.com/josedolz/IVD-Net/blob/490d9b6c4f9a6a662bbcda6d7b2593bf0b3222c4/IVD_Net.py'''
class InceptionBlock(nn.Module):

    def __init__(self,in_dim,out_dim,norm_type=0,hw=None,ibt=0):
        super(InceptionBlock,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ibt = ibt
        
        self.conv_1 = _conv_block(self.in_dim,self.out_dim, norm_type=norm_type, hw=hw)
        
        self.conv_2_1 = _conv_block(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0, dilation=1, norm_type=norm_type, hw=hw)
        
        self.conv_2_2 = _conv_block_Asym_Inception(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=norm_type, hw=hw)
        self.conv_2_3 = _conv_block_Asym_Inception(self.out_dim, self.out_dim, kernel_size=5, stride=1, padding=2, dilation=1, norm_type=norm_type, hw=hw)
        
        if self.ibt==1:
            self.max_seq = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
                _conv_block(self.out_dim, self.out_dim, kernel_size=1, stride=1, padding=0, dilation=1, norm_type=norm_type, hw=hw)
            )
            n = 4
        elif self.ibt==2:
            self.conv_2_4 = _conv_block_Asym_Inception(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=2, dilation=2, norm_type=norm_type, hw=hw)
            self.conv_2_5 = _conv_block_Asym_Inception(self.out_dim, self.out_dim, kernel_size=3, stride=1, padding=4, dilation=4, norm_type=norm_type, hw=hw)
            n = 5
        else:
            n = 3

        self.conv_2_output = _conv_block(self.out_dim*n, self.out_dim, kernel_size=1, stride=1, padding=0, dilation=1, norm_type=norm_type, hw=hw)
        
        self.conv_3 = _conv_block(self.out_dim,self.out_dim, norm_type=norm_type, hw=hw)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        
        conv_2_1 = self.conv_2_1(conv_1)
        conv_2_2 = self.conv_2_2(conv_1)
        conv_2_3 = self.conv_2_3(conv_1)
        
        if self.ibt==1:
            max_seq = self.max_seq(conv_1)
            out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3,max_seq], 1)
        elif self.ibt==2:
            conv_2_4 = self.conv_2_4(conv_1)
            conv_2_5 = self.conv_2_5(conv_1)
            out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3, conv_2_4, conv_2_5], 1)
        else:
            out1 = torch.cat([conv_2_1, conv_2_2, conv_2_3], 1)

        out1 = self.conv_2_output(out1)
        
        conv_3 = self.conv_3(out1+conv_1)
        return conv_3

class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_channels, out_channels, norm_type=0, hw=None, groups=1):
        super().__init__()

        if norm_type == -1:
            self.downsample = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=groups)),
            )
    
            self.conv = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)),
            )
            
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=groups),
                _get_normalization(norm_type,out_channels,hw)
            )
    
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                _get_normalization(norm_type,out_channels,hw),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                _get_normalization(norm_type,out_channels,hw)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_id = self.downsample(x)
        out = self.conv(x)
        out = self.relu(x_id + out)
        return out  

class AttentionBlock(nn.Module):
    """Attention Block"""
    def __init__(self, g_channels, l_channels, int_channels, norm_type=0, hw=None):
        super().__init__()

        if norm_type == -1:
             self.W_g = nn.utils.weight_norm(nn.Conv2d(g_channels, int_channels, kernel_size=1, padding=0))
    
             self.W_x = nn.utils.weight_norm(nn.Conv2d(l_channels, int_channels, kernel_size=1, padding=0))
    
             self.psi = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(int_channels, 1, kernel_size=1, padding=0)),
                nn.Sigmoid())
            
        else:
            self.W_g = nn.Sequential(
                nn.Conv2d(g_channels, int_channels, kernel_size=1, padding=0),
                _get_normalization(norm_type,int_channels,hw)
            )
    
            self.W_x = nn.Sequential(
                nn.Conv2d(l_channels, int_channels, kernel_size=1, padding=0),
                _get_normalization(norm_type,int_channels,hw)
            )
    
            self.psi = nn.Sequential(
                nn.Conv2d(int_channels, 1, kernel_size=1, padding=0),
                _get_normalization(norm_type,1,hw),
                nn.Sigmoid()
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class RecurrentBlock(nn.Module):
    """Recurrent block for R2U-Net model"""
    def __init__(self, out_channels, t=2, groups=1, norm_type=0, hw=None):
        super().__init__()
        
        self.t = t
        
        if norm_type == -1:
            self.conv = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)),
                nn.ReLU(inplace=True)
            )
            
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                _get_normalization(norm_type,out_channels,hw),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            out = self.conv(x + x1)
        return out
 
class RRCNNBlock(nn.Module):
    """Recurrent Residual CNN block"""
    def __init__(self, in_channels, out_channels, t=2, groups=1, norm_type=0, hw=None):
        super().__init__()
        
        self.RCNN = nn.Sequential(
                RecurrentBlock(out_channels, t=t, groups=groups, norm_type=norm_type, hw=hw),
                RecurrentBlock(out_channels, t=t, groups=groups, norm_type=norm_type, hw=hw),
                )
        
        if norm_type == -1:
            self.conv1 = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=groups))
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, groups=groups)
        
    def forward(self,x):
        x = self.conv1(x)
        x1 = self.RCNN(x)
        return x+x1

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, groups=1, norm_type=0, hw=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels;

        if norm_type == -1:
            self.double_conv = nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups)),
                nn.ReLU(inplace=True)
            )        
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups),
                _get_normalization(norm_type,mid_channels,hw),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups),
                _get_normalization(norm_type,out_channels,hw),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)

class Initial(nn.Module):
    """Initial Part"""

    def __init__(self, in_channels, out_channels, t=0, groups=1, res=False, ibt=0, norm_type=0, hw=None):
        super().__init__()
        if t > 0:
            self.initial_conv = RRCNNBlock(in_channels, out_channels, t=t, groups=groups, norm_type=norm_type, hw=hw)
        elif res:
            self.initial_conv = ResBlock(in_channels, out_channels, groups=groups, norm_type=norm_type, hw=hw)
        elif ibt > 0:
            self.initial_conv = InceptionBlock(in_channels,out_channels, norm_type=norm_type, hw=hw, ibt=ibt)
        else:
            self.initial_conv = DoubleConv(in_channels, out_channels, groups=groups, norm_type=norm_type, hw=hw)
        
    def forward(self, x):
        return self.initial_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then conv"""

    def __init__(self, in_channels, out_channels, drop_rate=0, t=0, groups=1, res=False, ibt=0, norm_type=0, hw=None, add_maxpool=True):
        super().__init__()
        layers = []
        
        if drop_rate > 0:
            layers.append(nn.Dropout2d(drop_rate))
        
        if add_maxpool:
            layers.append(nn.MaxPool2d(2))
        
        if t > 0:
            layers.append(RRCNNBlock(in_channels, out_channels, t=t, groups=groups, norm_type=norm_type, hw=hw))
        elif res:
            layers.append(ResBlock(in_channels, out_channels, groups=groups, norm_type=norm_type, hw=hw))
        elif ibt > 0:
            layers.append(InceptionBlock(in_channels,out_channels, norm_type=norm_type, hw=hw, ibt=ibt))
        else:
            layers.append(DoubleConv(in_channels, out_channels, groups=groups, norm_type=norm_type, hw=hw))
        
        self.maxpool_conv = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, l = 1, drop_rate=0, t=0, upsampling='bilinear', att=0, groups=1, norm_type=0, hw=None):
        super().__init__()

        self.att = att
        layers = []
    
        if drop_rate > 0:
            layers.append(nn.Dropout2d(drop_rate))

        if upsampling.lower() == 'transpose':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, groups=groups))
        else:
            layers.append(nn.Upsample(scale_factor=2, mode=upsampling.lower(), align_corners=True))
            if norm_type == -1:
                layers.append(nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
                layers.append(_get_normalization(norm_type,out_channels,hw))
            layers.append(nn.ReLU(inplace=True))            
            
        self.up = nn.Sequential(*layers)
        
        if att == 1:
            self.attblock = AttentionBlock(out_channels, out_channels, out_channels // 2, norm_type, hw)
        elif att == 2:
            self.attblock = MultiHeadCrossAttention(out_channels, in_channels, groups, norm_type, hw, num_heads=1)
        
        if t > 0:
            self.conv = RRCNNBlock(in_channels, out_channels, t=t, groups=groups, norm_type=norm_type, hw=hw)
        elif l > 1:
            self.conv = DoubleConv(out_channels * (l+1), out_channels, groups=groups, norm_type=norm_type, hw=hw)
        else:
            self.conv = DoubleConv(in_channels, out_channels, groups=groups, norm_type=norm_type, hw=hw)       

    def forward(self, x1, x2, x3=None, x4=None, x5=None):
        
        x = []
        x.append(self.up(x1))
        
        for j in [x2, x3, x4, x5]:
            if (j == None):
                break
            elif self.att == 1:
                x.append(self.attblock(x[0], j))
            elif self.att == 2:
                x.append(self.attblock(x1, j))
            else:
                x.append(j)
        
        x.reverse()
        
        x = torch.cat(x, dim=1)
            
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type=0):
        super(OutConv, self).__init__()
        
        if norm_type == -1:
            self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Fuse(nn.Module):
    def __init__(self,in_channels, out_channels, norm_type):
        super(Fuse, self).__init__()
        
        if norm_type == -1:
            self.conv3d = nn.utils.weight_norm(nn.Conv3d(in_channels, out_channels, kernel_size=(2,3,3), padding=(0,1,1)))
        else:
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(2,3,3), padding=(0,1,1))
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2):
        z = torch.stack([x1,x2],2)
        z = self.conv3d(z)
        z = z.squeeze(dim=2)
        self.relu(z)
        x = torch.cat([x1,x2], dim=1)
        return torch.mul(z,x)

class MultiAttention(nn.Module):
    def __init__(self,in_channels, num_heads=4):
        super(MultiAttention, self).__init__()
        
        assert in_channels % num_heads == 0, f'In_channels {in_channels} should be divisible by num_heads {num_heads}'
        
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads)
        
    def forward(self, x, x2=None):
        s = x.shape
        x = x.reshape(s[0],s[1],s[2]*s[3])
        x = x.permute(2,0,1)
        if x2==None:
            attn_output, attn_weights = self.multihead_attn(x, x, x)  #query, key, value
        else:
            s2 = x2.shape
            x2 = x2.reshape(s2[0],s2[1],s2[2]*s2[3])
            x2 = x2.permute(2,0,1)
            attn_output, attn_weights = self.multihead_attn(x, x, x2) #query, key, value
        
        attn_output = attn_output.permute(1,2,0)
        attn_output = attn_output.reshape(s[0],s[1],s[2],s[3])
            
        return attn_output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, s_channels, y_channels, groups, norm_type, hw, num_heads=4):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.sconv = nn.Sequential(
            nn.MaxPool2d(2),
            _conv_block(s_channels, s_channels, kernel_size=1, padding=0, norm_type=norm_type,hw=hw)
            )
            
        self.yconv = _conv_block(y_channels, s_channels, kernel_size=1, padding=0, norm_type=norm_type,hw=hw)
        
        self.mhsa = MultiAttention(in_channels=s_channels, num_heads=num_heads)
        
        self.mhsaconv = nn.Sequential(
            nn.Conv2d(s_channels, s_channels, kernel_size = 1),
            _get_normalization(norm_type,s_channels,hw),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
        
    def forward(self, y, s):
        sm = self.sconv(s)
        y = self.yconv(y)
        out = self.mhsa(y,sm) #Y is used for Q and K. S as V
        out = self.mhsaconv(out)
        out = s * out
        
        return out
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, forward_expansion=4):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
       
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion*embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
       
    def forward(self, q, kv):
        s = q.shape
        q = q.reshape(s[0],s[1],s[2]*s[3])
        q = q.permute(2,0,1)
        
        s2 = kv.shape
        kv = kv.reshape(s2[0],s2[1],s2[2]*s2[3])
        kv = kv.permute(2,0,1)
        
        attention,_ = self.attention(q, kv, kv)
        x = self.dropout(self.norm1(attention + q))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        out = out.permute(1,2,0)
        out = out.reshape(s[0],s[1],s[2],s[3])
        
        return out