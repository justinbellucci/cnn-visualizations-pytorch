# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_07_2020                                  
# REVISED DATE: 

import torch
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt

### ----------------------------------------------
def process_image(image, dim=224):
    """ Scales, crops (224 x 224 px), and normalizes a PIL image for a 
        Pytorch model. Accepts both a jpg or radom nois np.ndarray. Converts
        np.ndarray to a PIL image with shape (3, 224, 224).
       
        Arguments:
        - jpg image path or np.ndarray radom image
        - dim (desired pixel size)
       
        Returns:
        - Normalized Pytorch Tensor (image) 
    """
    # NOTE: Check ability to accept jpg images.
    if type(image) != Image.Image: 
        im = Image.fromarray(image)
    else:
        im = Image.open(image)  
        
    # resize image 
    width, height = im.size
    if width > height:
        ratio = width/height
        im.thumbnail((ratio*256, 256))
    elif height > width:
        ratio = height/width
        im.thumbnail((256, ratio*256))
    new_width, new_height = im.size
    
    # crop image around center
    left = (new_width - dim)/2
    top = (new_height - dim)/2
    right = (new_width + dim)/2
    bottom = (new_height + dim)/2
    im = im.crop((left, top, right, bottom))
    
    # convert to a np.array and divide by the color channel (int max)
    np_image = np.array(im)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (np_image - mean)/std
    # convert to a Tensor - reorder color channel so it is first. Torch requirement
    image = torch.FloatTensor(image.transpose(2, 0, 1))
    return image
### ----------------------------------------------

def rebuild_image(tensor):
    """ Rebuilds a Pytorch Tensor with dimensions (1, 3, w, h) and converts
        it to the necessary format for visualization. Reverses the normalization
        step using the mean and std from the ImageNet dataset. 

        Arguments:
            - tensor (torch.tensor with shape = (1, 3, w, h)
        Returns:
            - image (np.ndarray)
    """
    np_image = tensor.detach().numpy() # convert tensor to nparray
    np_image = np_image.squeeze(0) # reduce size of tensor
    np_image = np_image.transpose(1, 2, 0) # reorder color channel 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image * std + mean)  # 

    return image


def save_image(img, path):
    """ Displays and saves the processsed image from the 
        given layer/filter number.

        Arguments:
            - image (np.ndarray)
            - path (string) save path
    """
    plt.figure(figsize=[2,2])
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    plt.savefig(path, dpi=150)
    plt.show()  
    