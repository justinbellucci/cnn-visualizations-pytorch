# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_07_2020                                  
# REVISED DATE: 

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### ----------------------------------------------
def process_image(image, dim=224):
    """Scales, crops (224 x 224 px), and normalizes a PIL image for a 
       Pytorch model. Accepts both a jpg or radom nois np.ndarray. Converts
       np.ndarray to a PIL image with shape (3, 224, 224).
       
       Arguments:
       - jpg image path or np.ndarray radom image
       - dim (desired pixel size)
       
       Returns:
       - Normalized Pytorch Tensor (image) 
    """
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
#     if torch.cuda.is_available():
#         image = torch.cuda.FloatTensor(image.transpose(2, 0, 1))
#     else:
#         image = torch.FloatTensor(image.transpose(2, 0, 1))
    image = torch.FloatTensor(image.transpose(2, 0, 1))
    return image

### ----------------------------------------------
def show_img(img_path):
    im = Image.open(img_path)
    np_image = np.array(im)
#     plt.title('{} - {}'.format(class_pred[0], dog_name).title())
    plt.axis('off')
    plt.imshow(np_image)
    plt.show()