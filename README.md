# Image Masking Utility

Program provides the ability for users to mask a set of images from a specified directory 'data/imgs'. Created masks will be outputed to 'data/masks' directory. 

## Requirements
- Python3
- OpenCV
- Numpy

It is recommended to install anaconda to facilitate the management of packages. https://www.anaconda.com/distribution/. Furthermore, you may create and segregate environments to avoid package dependency conflicts between projects.

### Steps
1) Open anaconda command line prompt
2) Creating new environment (optional): 'conda create --name new_env_name'
3) Activate environment (optional): 'conda activate new_env_name'
4) Install Python on new environment: 'conda install -c anaconda python'
5) Install OpenCV: 'conda install -c conda-forge opencv'
6) Install Numpy (needed only if had not been done as part of step 5): 'conda install -c conda-forge numpy'

## Usage

1) Before running the script place all images in the 'data/imgs' directory.

2) Execute: `python image_masking.py`

```shell script
> python image_masking.py [-h] [-v]

optional arguments:
  -h, --help  show this help message and exit
  -v, --viz   Display image and created mask (default: False)
```

3) A pop up window with image will appear. A single image will be available at a time. 

4) While left clicking on mouse, mask the area of interest.

5) If satisfied with mask hit the 'r' key to save the mask (output will be saved under 'data/masks' directory). If wish to clear mask and reset image hit the space bar.

6) In the event wish to pass an image, either hit the escape key or close the window.

7) If using the viz option and wish to proceed to next image hit any key or close window.
