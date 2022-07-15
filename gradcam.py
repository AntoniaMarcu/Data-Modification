"""
    Implementation based on publicly available code
    (https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)
"""

import numpy as np
import cv2
import math

import torch
import torch.nn.functional as F

from models.resnet import ResNet18


class ResNet_CAM(torch.nn.Module):
    """
    Class used for hooking the gradients.
    Args:
        * net: instance of ResNet18 on which to perform Grad-CAM
        * layer_k: layer from which to grab activations
    """

    def __init__(self, net: ResNet18, layer_k: int):
        super(ResNet_CAM, self).__init__()
        self.resnet = net
        modules = list(net.children())[:-1]
        modules.insert(2, torch.nn.ReLU())
        convs = torch.nn.Sequential(*modules)
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = torch.nn.Sequential(*list(net.children())[-1:])

    # registers hook in the forward function
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.first_part_conv(x)
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view((batch_size, -1))
        x = self.linear(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.first_part_conv(x)


def get_grad_cam(net, images, proportion):
    # get gradcam heatmap
    net.eval()
    pred = net(images)
    device = images.get_device()
    pred[torch.arange(0, pred.shape[0]), pred.argmax(dim=1)].backward(torch.ones(pred.shape[0]).to(device))
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    activations = net.get_activations(images).detach()
    for j in range(activations.size(0)):
        for i in range(activations.size(1)):
            activations[j, i, :, :] *= pooled_gradients[j][i]
    heatmap = torch.mean(activations, dim=1).cpu()
    for j in range(activations.size(0)):
        heatmap[j] = np.maximum(heatmap[j], 0)
        heatmap[j] /= torch.max(heatmap[j])

    # randomly choose to occlude either the most or the least salient pixels
    most = torch.randint(2, (1,)).item()

    # generate masks based on the heatmap
    masks = torch.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))
    for i in range(0, len(images)):
        resized_heatmap = cv2.resize(heatmap[i].numpy(), (images[i].shape[1], images[i].shape[2]))
        resized_heatmap = np.uint8(255 * resized_heatmap)
        idx = resized_heatmap.reshape(-1).argsort()[::-1]
        mask = resized_heatmap.reshape(-1)

        # Invert proportion if keeping the most salient feature
        used_proportion = abs(most - proportion)

        num = math.ceil(used_proportion * mask.size)
        mask[idx[:num]] = most
        mask[idx[num:]] = abs(most - 1)
        mask = mask.reshape((1, 32, 32))
        masks[i] = torch.Tensor(mask).to(device)
    masks = masks.to(device)
    images = images.to(device)
    return images * masks
