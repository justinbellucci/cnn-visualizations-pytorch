# Convolutional Neural Network Visualizations 

This repository is an attempt for me to visually represent the inner workings of convolutional neural networks. This work is by no means revolutionary, however, I am trying to illustrate various methods for representing how a CNN makes decisions. In this effort I hope to understand the fine details of CNNs. Deep neural networks do not have to be black boxes. It may seem that it is some miracle that a model can identify a cat in an image, but believe me, it's not. It's just really complicated math under the hood. I believe that every ML engineer should understand how their model makes decisions, which ultimatly should answer questions related to bias. I'm new at this so bare with me...

### Navigation
* [Running Notebook Locally](#installing_locally)
* [Filter Visualization](#filter_vis)
* [Feature Map Visualization](#feature_map_visualization)
* [References](#referances)

<a id='installing_locally'></a>
## Installing Locally
If you are new to this, feel free to install locally and make it your own.
1. Install dependencies (I recommended that you create an environment in Conda-Python3)  
>`pip install requirments.txt`
2. The following Jupyter notebooks outline various visualization methods:
    * `cnn_filter_visualizations.ipynb` Jupyter notebook 

<a id='filter_vis'></a>
## Filter Visualization

Generally speaking, filters in a CNN are used to extract information from an image that is then passed through the network to make predictions. These filters are called kernels. Mathmatically they perform operations on pixels that reduce an image to basic features. Each CNN layer can have hundreds of layers (kernels). These layers make up the depth of a CNN. The following gif <sup>[1](#1)</sup> illustrates how a filter is applied to an an image:

<p align="center">
<img width="250" src = "images/padding_strides.gif">
</p>

### Model Architecture

In order to visualize the various filters and feature maps of a neural netork we first need to load a pre-trained network from Pytorch. We will use the VGG16<sup>[2](#1)</sup> neural network and extract each corresponding convolutional layer. We not performing backpropagation so the 

<a id='references'></a>
## References
[1]<a id='1'></a> https://github.com/vdumoulin/conv_arithmetic\
[2]<a id='2'></a> *Very Deep Convolutional Networks for Large-Scale Image Recognition.* Simonyan, K.,
Zisserman, A. 2015.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://arxiv.org/abs/1409.1556
