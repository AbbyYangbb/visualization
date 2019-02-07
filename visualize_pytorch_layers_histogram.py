"""
This is modified from: 
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/cnn_layer_visualization.py

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from matplotlib import pyplot as plt
import torchvision.transforms as T
from skimage import io


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, img_path, selected_layers, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layers = selected_layers
        self.selected_filter = selected_filter
        self.conv_output = {}
        # Create the folder to export images if not exists
        s_a = img_path.split('/')[-1].split('.')[0] 
        s_b = '_results'
        self.output_path = ''.join([s_a, s_b])
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        transform = T.Compose([T.ToTensor(), 
                               T.Normalize(mean = [0.485, 0.456, 0.406], 
                                           std = [0.229, 0.224, 0.225])])
        self.image = transform(io.imread(img_path)).unsqueeze_(0)

    def hook_layer(self, layer):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output[layer] = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[layer].register_forward_hook(hook_function)

    def create_and_save_histogram(self, data, bin_num, fname):
        plt.figure(figsize=(4.5, 2.5))
        plt.hist(data, bin_num)
        plt.savefig(os.path.join(self.output_path, fname))

    def visualise_layer_with_hooks(self, bin_num):
        # Hook the selected layer
        [self.hook_layer(i) for i in self.selected_layers]
        # get the forward stop layer
        final_layer = max(self.selected_layers)

        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))

        # Assign create image to a variable to move forward in the model
        x = self.image
        for index, layer in enumerate(self.model):
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == final_layer:
                break

        # create the histograms for weights:
        for layer in self.selected_layers:
            layer_obj = self.model[layer]
            if not isinstance(layer_obj, nn.Conv2d):
                del self.conv_output[layer]
                continue
            weights = layer_obj.weight.data.numpy().flatten()
            self.create_and_save_histogram(weights, bin_num, 'layer_{}_weights_histogram.png'.format(layer))

        # create histograms for features:
        for layer in self.conv_output:
            features = self.conv_output[layer].data.numpy().flatten()
            self.create_and_save_histogram(features, bin_num, 'layer_{}_features_histogram.png'.format(layer))



if __name__ == '__main__':
    # the first 5 Conv2d layer of VGG16
    cnn_layers = [0, 2, 5, 7, 10]
    filter_pos = 5
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, "./starry_night.jpg", cnn_layers, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks(20)
