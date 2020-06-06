# Convolutional Neural Network Visualizations 

This repository is an attempt for me to visually represent the inner workings of convolutional neural networks. This work is by no means revolutionary, however, I am trying to illustrate various methods for representing how a CNN makes decisions. In this effort I hope to understand the fine details of CNNs. Deep neural networks do not have to be black boxes. It may seem that it is some miracle that a model can identify a cat in an image, but believe me, it's not. It's just really complicated math under the hood. I believe that every ML engineer should understand how their model makes decisions, which ultimatly should answer questions related to bias. I'm new at this so bare with me...

### Navigation
* [Running Notebook Locally](#installing_locally)
* [Filter Visualization](#filter_vis)
* [Feature Map Visualization](#feature_map_vis)
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

Generally speaking, filters in a CNN are used to extract information from an image that is then passed through the network to make predictions. These filters are called kernels. Mathmatically they perform operations on pixels that reduce an image to basic features. Each CNN layer can have hundreds of layers (kernels). These layers make up the depth of a CNN. The following gif<sup>[1](#1)</sup> illustrates how a filter is applied to an an image:

<p align="center">
<img width="250" src = "images/padding_strides.gif">
</p>

### Model Architecture

In order to visualize the various filters and feature maps of a neural netork we first need to load a pre-trained network from Pytorch. We will use the VGG16<sup>[2](#1)</sup> neural network and extract each corresponding convolutional layer. We will not performing backpropagation. Instead, we will use each layer's weights to help visualize the filters used and the resulting image processing.

### Filter Layers

Taking a look at 3 of the 13 convolutional layers in the VGG16 model we see that there is increased depth as we move through the model. The following images illustrate each filter in the respective layers. **Note:** The filters are displayed in grayscale for readability.
<!---
<p align="center">
<img width="250" src = "images/conv_layer_1_filter.jpg">
</p>
<p align="center">
<td align="left"> Layer 1 Conv2d filters Kernel 3 x 3 - Depth 64</td>
</p>

<p align="center">
<img width="250" src = "images/conv_layer_5_filter.jpg">
</p>
<p align="center">
<td align="left"> Layer 5 Conv2d filters Kernel 3 x 3 - Depth 256</td>
</p>

<p align="center">
<img width="250" src = "images/conv_layer_10_filter.jpg">
</p>
<p align="center">
<td align="left"> Layer 10 Conv2d filters Kernel 3 x 3 - Depth 512</td>
</p
--->

<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="20%" align="center"> Layer 1: <strong>Conv2d</strong> </td>
			<td width="20%" align="center"> Layer 5: <strong>Conv2d</strong> </td>
			<td width="20%" align="center"> Layer 10: <strong>Conv2d</strong> </td>
		</tr>
		<tr>
			<td width="20%" align="center"> <img src="images/conv_layer_1_filter.jpg"> </td>
			<td width="20%" align="center"> <img src="images/conv_layer_5_filter.jpg"> </td>
			<td width="20%" align="center"> <img src="images/conv_layer_10_filter.jpg"> </td>
		</tr>
	</tbody>
</table>


<a id='feature_map_vis'></a>
## Feature Map Visualization

When we pass an image into the pre-trained network we process it at each layer and save the respective image representation. This is essentially what the image looks like after each filter is applied. First we will pass in an adorable picture of a black lab. Yea, I know. 

<p align="center">
<img width="250" src = "images/Labrador_retriever_01.jpg">
</p>

When we pass the image through the first convolutional layer we would essentially get 64 corresponding filtered images. Let's take a look at when kernel 17 is applied to the image. **Note:** There is some preprocessing that was done which is why the image looks squished. 

<p align="center">
<img width="500" src = "images/lab_layer_1.jpg">
</p>

<a id='references'></a>
## References
[1]<a id='1'></a> https://github.com/vdumoulin/conv_arithmetic  

[2]<a id='2'></a> *Very Deep Convolutional Networks for Large-Scale Image Recognition.* Simonyan, K.,
Zisserman, A. 2015.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://arxiv.org/abs/1409.1556
