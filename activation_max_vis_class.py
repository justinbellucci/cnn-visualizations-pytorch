# PROGRAMMER: Justin Bellucci 
# DATE CREATED: 06_09_2020                                  
# REVISED DATE: 

import torch
from torch import optim
# from torchvision import models, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from helper_functions import process_image, rebuild_image, save_image

class ActivationMaximizationVis():
    """ Activation Maximization Class based on Erhan et al. (2009) 
        paper for visualizing filters in a CNN. 

        Attributes:
            - model (CNN model imported from torchvision)
            - epochs (int) number of iterations
            - cnn_layer (int) layer to hook gradients
            - cnn_filter (int) filter number  
    """
    
    def __init__(self, model, epochs, cnn_layer, cnn_filter):
        self.model = model
        self.model.eval() # set model to evaluation mode
        self.epochs = epochs
        self.cnn_layer = cnn_layer
        self.cnn_filter = cnn_filter
        self.conv_output = 0 # initialize the output of the model for loss
        
        if not os.path.exists('activ_max_imgs'):
            os.makedirs('activ_max_imgs')
            
    def hook_cnn_layer(self):
        """ Initiates a forward hook function to save the gradients
            from the selected cnn layer.
            
            Arguments:
                - self
        """
        def hook_fn(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.cnn_filter]
            print('---- conv out -----')
            print(type(self.conv_output))
            print(self.conv_output.shape)
            print(grad_out.shape)
            print(grad_out[0, self.cnn_filter])
            print('\n----- grad in ------')
            print(grad_in[0][0][0])
            # saving the number of filters in that layer
            self.num_filters = grad_out.shape[1]
        self.model[self.cnn_layer].register_forward_hook(hook_fn)
        
    def vis_cnn_layer(self):
        """ Method to visualize selected filter (activation map) from 
            a CNN layer. Creates a random image as input. 
        """
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
            # print(self.num_filters)
            # self.conv_output = x[0, self.cnn_filter]
            # loss function according to Erhan et al. (2009)
            loss = -torch.mean(self.conv_output)
            print(loss.shape)
            print(loss.data)
            loss.backward() # calculate gradients
            optimizer.step() # update weights
            self.layer_img = rebuild_image(processed_image) # reconstruct image
            
            print('Epoch {}/{} --> Loss {:.3f}'.format(e+1, self.epochs, loss.data.numpy()))

            if e % 5 == 0:
                img_path = 'activ_max_imgs/am_vis_l' + str(self.cnn_layer) + \
                    '_f' + str(self.cnn_filter) + '_iter' + str(e+1) + '.jpg'
                save_image(self.layer_img, img_path)

    
