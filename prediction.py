#
#  Mask Nerve Prediction
#  Code modified by: Gary Milam Jr.
#  Created/Modified Date: 01/25/2023
#  Affiliation: ART-Med Lab. (PI: Chung Hyuk Park), BME Dept., SEAS, GWU
#

import torch 
import yaml

from os.path import join
from unet import get_model
from torchvision import transforms    

class NerveMask():
    def __init__(self, threshold = 0.5, x=1024, y=1224):
        super(NerveMask, self).__init__()
              
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        main_dir = '.'
        config_dir = join(main_dir,'data')
        fn_config = join(config_dir,'config_net.yaml')
        config = yaml.safe_load(open(fn_config,'r'))
        self.net = self._get_network(config, config_dir, self.device)
        self.net.eval()
        
        self.dt = torch.float32
        self.tf1, self.tf2 = self._get_tf_imgsize(config)
        
        #For mask creation
        self.thresh = threshold
        self.tv = transforms.Resize((x,y),interpolation=transforms.InterpolationMode.BILINEAR)
    
    @staticmethod
    def _get_network(config, config_dir, device):
        net = get_model(config)
        net.to(device=device)   
        pred_config = config['prediction']       
        model = join(config_dir,pred_config['model'])   
        state_dict = torch.load(model, map_location=device);   
        net.load_state_dict(state_dict)
        return net

    @staticmethod
    def _get_mean_std(config):
        mean = config.get('mean', None)
        std = config.get('std', None)
        return mean, std

    def _get_tf_imgsize(self,config):
        mean, std = self._get_mean_std(config['stats'])    
        mean2, std2 = self._get_mean_std(config.get('stats2', None))
        img_size = config.get('img_size', 256)
        
        #Using bicubic interpolation, same as in training. However, to speed
        #it up test with other interpolations if make too much difference.
        tf1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size,img_size),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
            transforms.Normalize(mean=mean,std=std)]
            )

        tf2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size,img_size),interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
            transforms.Normalize(mean=mean2,std=std2)]
            )
        
        return tf1, tf2
                    
    @torch.no_grad()
    def get_mask(self, bfm_img, rgb_img):
        
        rgb = self.tf2(rgb_img).to(device=self.device, dtype=self.dt).unsqueeze(0)
        bfm = self.tf1(bfm_img).unsqueeze(0)
        
        outputs = torch.sigmoid(self.net(bfm,rgb))
        
        mask = self.tv(outputs) #Required to transform mask to original image sizes
        mask = ((mask.squeeze() > self.thresh)).cpu().numpy()
        
        return mask