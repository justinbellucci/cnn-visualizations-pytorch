# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_09_2020                                  
# REVISED DATE: 

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import optim
from helper_functions import process_image, rebuild_image, save_image

class ActivationMaximizationVis():
    """ 
    
    """
    
    def __init__(self, model, epochs, cnn_layer, cnn_filter):
        self.model = model
        self.model.eval() # set model to evaluation mode
        self.epochs = epochs
        self.cnn_layer = cnn_layer
        self.cnn_filter = cnn_filter
        self.conv_output = 0 # initialize the output of the model for loss
        
        if not os.path.exists('activ_max_imgs'):
            os.makedirs('/activ_max_imgs')
            
    def hook_cnn_layer(self):
        """ Initiates a forward hook function to save the gradients
            from the selected cnn layer.
            
            Arguments:
                - self
        """
        def hook_fn(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.cnn_filter]
        self.model[self.cnn_layer].register_forward_hook(hook_fn)
        
    def vis_cnn_layer(self):
        # initiate hook function
        self.hook_cnn_layer()
        # create a noisy image
        noisy_img = np.random.randint(125, 190, (224, 224, 3), dtype='uint8')
        # add dimension and activate requires_grad on tensor
        processed_image = process_image(noisy_img).unsqueeze_(0).requires_grad_()
        # define optimizer
        optimizer = optim.Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for e in range(1, self.epochs):
            optimizer.zero_grad() # zero out gradients
            x = processed_image
            
            # iterate through each layer of the model
            for idx, layer in enumerate(self.model):
                # pass processed image through each layer
                x = layer(x)
                # activate hook when image gets to the layer in question
                if idx == self.cnn_layer:
                    break
            # loss function according to Qin et. al
            loss = -torch.mean(self.conv_output)
            loss.backward() # calculate gradients
            optimizer.step() # update weights
            self.layer_img = rebuild_image(processed_image) # reconstruct image
            # print(type(self.layer_img))
            # print(self.layer_img.shape)
            print('Epoch {}/{} --> Loss {:.3f}'.format(e+1, self.epochs, loss.data.numpy()))
            
            if e % 10 == 0:
                img_path = 'activ_max_imgs/am_layer_' + str(self.cnn_layer) + \
                    '_f_' + str(self.cnn_filter) + '_iter' + str(e+1) + '.jpg'
                plt.imshow(self.layer_img)
                plt.axis('off')
                plt.savefig(img_path, dpi=150)
                # save_image(self.layer_img, img_path)
                # imsave(img_path, self.layer_img)
                plt.show()
    