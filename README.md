# Image Masking Utility

Program provides the ability for users to mask a set of images from a specified directory 'data/imgs'. Created masks will be outputed to 'data/masks' directory. 

If an image has a corresponding mask when running program, that mask will be used as the base from which user can modify as needed.

## Requirements
- Python3
- OpenCV
- Numpy
- Matplotlib

It is recommended to install anaconda to facilitate the management of packages. https://www.anaconda.com/distribution/. Furthermore, you may create and segregate environments to avoid package dependency conflicts between projects.

### Steps
1) Open anaconda command line prompt
2) Creating new environment (optional): 'conda create --name new_env_name python=3.7'
3) Activate environment (optional): 'conda activate new_env_name'
4) Install Python on new environment (not needed if step 2 executed): 'conda install -c anaconda python'
5) Install OpenCV: 'conda install -c conda-forge opencv'
6) Install Numpy (not needed if step 5 executed): 'conda install -c conda-forge numpy'
7) Install Matplotlib: 'conda install -c conda-forge matplotlib'

## Usage

1) Before running the script place all images in the 'data/imgs' directory.
  a) (Optional) Set corresponding masks on 'data/masks' directory. ***Note: image and mask should be named the same.***

2) Execute: `python image_masking.py`

```shell script
> python image_masking.py [-h] [-d]

optional arguments:
  -h, --help  show this help message and exit
  -d, --debug  Enables debugging mode (default: False)
```

3) A pop up window with image will appear. A single image will be available at a time.
  a) If a mask existed for the image (step 1.a), this mask will be considered as the base and will be overlayed on the image.

4) Depending on the operating system, a trackbar will appear on the top or bottom of the window. This allows the user to change dynamically the thickness of the line (both for masking or erasing mask.)

4) To mask an area of interest, hold down left mouse button; to erase a portion of the mask, hold down right mouse button 

5) Four options are available when creating masks:
  a) If no further modifications to be performend on created mask (e.g. final mask), user should press 'f' key.
  b) To run the watershed algorithm on a created mask which may help fill in the boundaries of the object (e.g. nerve), user should press 'w' key. **Note: Depending on image, over segmentation may occur.**
  c) To reset the inpainting mask to the base mask (e.g. background if no mask previously existed, step 1.a), user should press the space key.
  d) To skip and go to next image in imgs directory, user press 'ESC' key. **Any mask creation/modification will be lost and not saved**

6) Once a mask is chosen, a new window will appear that will include the original image, the user created mask, the watershed final mask (only if chose to run watershed) and an overlay of the mask on the greyscale image.

7) To refine the mask, user should press the 'r' key and will go back to step 3. If user completely satisfied with mask, user should press the 'n' key or close window.

8) Final mask is saved on the masks directory. Mask will be saved with the same name as the image used.

9) Next image from the imgs directory is opened. Go back to step 3.
