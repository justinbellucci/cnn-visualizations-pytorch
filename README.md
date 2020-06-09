# Convolutional Neural Network Visualizations 

This repository is an attempt for me to visually represent the inner workings of convolutional neural networks. This work is by no means revolutionary, however, I am trying to illustrate various methods for representing how a CNN makes decisions. In this effort I hope to understand the fine details of CNNs. Deep neural networks do not have to be black boxes. It may seem that it is some miracle that a model can identify a cat in an image, but believe me, it's not. It's just really complicated math under the hood. I believe that every ML engineer should understand how their model makes decisions, which ultimatly should answer questions related to bias. I'm new at this so bare with me...

### Navigation
* [Running Notebook Locally](#installing_locally)
* [Filter Visualization](#filter_vis)
* [Activation Map Visualization](#activation_map_vis)
* [Activation Maximization](#max_activations)
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

Generally speaking, filters in a CNN are used to extract information from an image that is then passed through the network to make predictions. These filters are called kernels. Mathmatically they perform operations on pixels that reduce an image to basic features. Each CNN layer can have hundreds of layers (kernels). These layers make up the depth of a CNN. The following gif<sup>[[1]](#1)</sup> illustrates how a filter is applied to an an image:

<p align="center">
<img width="250" src = "images/padding_strides.gif">
</p>

### Model Architecture

In order to visualize the various filters and feature maps of a neural netork we first need to load a pre-trained network from Pytorch. We will use the VGG16<sup>[[2]](#1)</sup> neural network and extract each corresponding convolutional layer. We will not performing backpropagation. Instead, we will use each layer's weights to help visualize the filters used and the resulting image processing.

### Filter Layers

Taking a look at 3 of the 13 convolutional layers in the VGG16 model we see that there is increased depth as we move through the model. The following images illustrate each filter in the respective layers. **Note:** The filters are displayed in grayscale for readability.

<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="20%" align="center"> Layer 1: 3x3 Kernel: Depth 64 </td>
			<td width="20%" align="center"> Layer 5: 3x3 Kernel: Depth 256 </td>
			<td width="20%" align="center"> Layer 10: 3x3 Kernel: Depth 512 </td>
		</tr>
		<tr>
			<td width="20%" align="center"> <img src="filter_imgs/conv_layer_1_filter.jpg"> </td>
			<td width="20%" align="center"> <img src="filter_imgs/conv_layer_5_filter.jpg"> </td>
			<td width="20%" align="center"> <img src="filter_imgs/conv_layer_10_filter.jpg"> </td>
		</tr>
	</tbody>
</table>

<a id='activation_map_vis'></a>
## Activation Map Visualization

When we pass an image into the pre-trained network we process it at each layer and save the respective image representation. This is essentially what the image looks like after each filter is applied. First we will pass in an adorable picture of a black lab. Yea, I know. 

<p align="center">
<img width="250" src = "images/Labrador_retriever_01.jpg">
</p>

When we pass the image through the first convolutional layer we will essentially get 64 corresponding activation maps. Let's take a look at when kernel 17 is applied to the image on layer 1. **Note:** There is some preprocessing that was done which is why the image looks squished. 

<p align="center">
<img width="500" src = "filter_imgs/lab_layer_1.jpg">
</p>

### Processing Through Mulitple Layers
After some pre-processing the below block of code takes an image and applies it to each `torch.nn.Conv2d` layer. The output of one layer is the input to the next. 

```python
    # Pass image through the first convolutional layer 
    # save the outpue
    conv_out = [conv_layers[0](image)]
    # Iteratively pass image through all convolutional layers
    for i in range(1, len(conv_layers)):
        conv_out.append(conv_layers[i](conv_out[-1]))
```
The depth of Layer 1 is 64. You can see how each filter extracts different details from the image. Layer 1 feature maps are fairly clear. As we move deeper into the model we can see how the detail in the image starts to degrade. Can you pick out what the feature maps are representing? Sometimes the outline of the image is clear, sometimes dark colors are emphesized, and sometimes it is hard to tell it what the image is originally of. 

<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="20%" align="center"> Layer 1: 3x3 Kernel </td>
			<td width="20%" align="center"> Layer 1: Filtered Images </td>
		</tr>
		<tr>
			<td width="20%" align="center"> <img src="filter_imgs/conv_layer_1_filter.jpg"> </td>
			<td width="20%" align="center"> <img src="filter_imgs/conv_layer_1_output.jpg"> </td>
		</tr>
	</tbody>
</table>
<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="20%" align="center"> Layer 2</td>
			<td width="20%" align="center"> Layer 4</td>
            <td width="20%" align="center"> Layer 6</td>
		</tr>
		<tr>
			<td width="20%" align="center"> <img src="filter_imgs/lab_layer_2.jpg"> </td>
			<td width="20%" align="center"> <img src="filter_imgs/lab_layer_2.jpg"> </td>
            <td width="20%" align="center"> <img src="filter_imgs/lab_layer_6.jpg"> </td>
		</tr>
	</tbody>
</table>
<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="20%" align="center"> Layer 8</td>
			<td width="20%" align="center"> Layer 10</td>
            <td width="20%" align="center"> Layer 12</td>
		</tr>
		<tr>
			<td width="20%" align="center"> <img src="filter_imgs/lab_layer_8.jpg"> </td>
			<td width="20%" align="center"> <img src="filter_imgs/lab_layer_10.jpg"> </td>
            <td width="20%" align="center"> <img src="filter_imgs/lab_layer_12.jpg"> </td>
		</tr>
	</tbody>
</table>

<a id='max_activations'></a>
## Activation Maximization 

Bla bla bla. Write some stuff here.

<p align="center">
<img width="250" src = "filter_imgs/01_noisy_image.jpg">
</p>

### The Algorithm

Write some stuff here.

### Layer Vis
Taking a look at the first few layers you can see...
<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="5%" align="center"> Layer 1 - Filter 1</td>
			<td width="5%" align="center"> Layer 1 - Filter 5</td>
            <td width="5%" align="center"> Layer 1 - Filter 6</td>
			<td width="5%" align="center"> Layer 1 - Filter 6</td>
		</tr>
		<tr>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l1_f1_iter31.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l1_f5_iter21.jpg"> </td>
            <td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l1_f6_iter31.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l1_f55_iter41.jpg"> </td>
		</tr>
	</tbody>
</table>
Now if we take a look more layers you can see...
<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="5%" align="center"> Layer 3 - Filter 1</td>
			<td width="5%" align="center"> Layer 3 - Filter 5</td>
            <td width="5%" align="center"> Layer 3 - Filter 28</td>
			<td width="5%" align="center"> Layer 3 - Filter 38</td>
		</tr>
		<tr>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l3_f1_iter31.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l3_f5_iter41.jpg"> </td>
            <td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l3_f28_iter31.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l3_f38_iter31.jpg"> </td>
		</tr>
	</tbody>
</table>

<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="5%" align="center"> Layer 10 - Filter 5</td>
			<td width="5%" align="center"> Layer 10 - Filter 10</td>
            <td width="5%" align="center"> Layer 10 - Filter 65</td>
			<td width="5%" align="center"> Layer 10 - Filter 165</td>
		</tr>
		<tr>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l10_f5_iter41.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l10_f10_iter51.jpg"> </td>
            <td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l10_f65_iter51.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l10_f165_iter51.jpg"> </td>
		</tr>
	</tbody>
</table>

<table border=0 width="800px" align="center">
	<tbody> 
    <tr>		
            <td width="5%" align="center"> Layer 12 - Filter 5</td>
			<td width="5%" align="center"> Layer 12 - Filter 10</td>
            <td width="5%" align="center"> Layer 12 - Filter 65</td>
			<td width="5%" align="center"> Layer 12 - Filter 165</td>
		</tr>
		<tr>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l12_f28_iter51.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l12_f68_iter51.jpg"> </td>
            <td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l12_f168_iter51.jpg"> </td>
			<td width="5%" align="center"> <img src="activ_max_imgs/am_vis_l12_f178_iter51.jpg"> </td>
		</tr>
	</tbody>
</table>

<a id='references'></a>
## References
[1]<a id='1'></a> https://github.com/vdumoulin/conv_arithmetic  

[2]<a id='2'></a> *Very Deep Convolutional Networks for Large-Scale Image Recognition.* Simonyan, K.,
Zisserman, A. 2015.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://arxiv.org/abs/1409.1556

[3]<a id='3'></a> D. Erhan, Y. Bengio, A. Courville and P. Vincent. *Visualizing higher-layer features of a deep network*  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Technical report, University of Montreal, 1341 (2009), p3.

[4]<a id='4'></a> Z. Qin, F. Yu, C. Liu, and X. Chen, *How convolutional neural network see the world - A survey of   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;convolutional neural network visualization methods*. 2018. https://arxiv.org/abs/1804.11191 