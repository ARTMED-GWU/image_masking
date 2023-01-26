# Image Masking Utility

The program allows users to mask images from a specified 'data/imgs' directory. Created masks will output to 'data/masks' directory. 

If an image has a corresponding mask when running the program, that mask will be the base from which the user can modify it as needed.

## Requirements
- Python >= 3.9
- OpenCV
- Numpy
- Matplotlib (only for debugging)
- torch >= 1.9.0
- torchvision >= 0.10.0
- Pyyaml	

It is recommended to install anaconda to facilitate the management of packages. https://www.anaconda.com/distribution/. Furthermore, create and segregate environments to avoid package dependency conflicts between projects.

### Steps
1. Open the anaconda command line prompt
2. Creating new environment (optional): 'conda create --name new_env_name python=3.9'
3. Activate environment (optional): 'conda activate new_env_name'
4. Install Python on a new environment (if step 2 was executed, this is not required): 'conda install -c anaconda python'
5. Install OpenCV: 'conda install -c conda-forge opencv'
6. Install Numpy (not needed if step 5 was executed): 'conda install -c conda-forge numpy'
7. Install Matplotlib: 'conda install -c conda-forge matplotlib'
8. Install pyyaml: 'conda install -c anaconda pyyaml'

## Usage

1. Before running the script, place jet images under 'data/imgs/jet' and RGB images under 'data/imgs/rgb'
   1. (Optional) Set corresponding masks on 'data/masks'. ***Note: images and masks should be named the same.***
   2. (Optional) For mask predictions, set the config_net.yaml file and the model file the configuration is referencing.

2. Execute: `python image_masking.py`

```shell script
> python image_masking.py [-h] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           Enables debugging mode (default: False)
  -s, --skip_state      Ignores list of processed images (default: False)
  -r, --reset_state     Resets state of processed images (default: False)
  -c CONFIG, --configfile CONFIG
                        Configuration file (default: config.yaml)
```

3. A popup window with an image will appear. A single image will be available at a time.
	1. When the predict option is set to 'true' in the config file, the program will try to predict the mask and overlay it on the image.
	2. If the predict option is set 'false' and a mask exists for the image (step 1.i), this mask is considered base and will overlay on the image.

4. Depending on the operating system, a trackbar will appear on the top or bottom of the window. The trackbar will allow you to change the thickness of the line (both for masking or erasing the mask) dynamically

5. To mask an area of interest, hold down the left mouse button; to erase a portion of the mask, hold down the right mouse button 

6. Click on the middle mouse button to switch between the jet (birefringence) image and the RGB image. 

7. Four options are available when creating masks:
   1. To finalize the created mask (e.g. final mask), press the 'f' key.
   2. To run the watershed algorithm on a created mask that may help fill in the object's boundaries (e.g. nerve), press the 'w' key. **Note: Depending on the image, over-segmentation may occur.**
   3. To reset the inpainting mask to the base mask (e.g. background if no mask previously existed, step 1.i), the user should press the space key.
   4. To skip and go to the next image available, press the 'ESC' key. **Any mask creation/modification will be lost and not saved**

8. Finalizing the mask opens a new window that will include the original image, the user-created mask, the final watershed mask (only if watershed is chosen), and an overlay of the mask on the greyscale image.

9. To refine the mask, press the 'r' key and the program will go back to step 3. If completely satisfied with the mask, press the 'n' key or close the window.

10. The final mask is saved in the masks directory. The mask will be saved with the same name as the image used.

11. Next image from the imgs directory is opened. Go back to step 3.
