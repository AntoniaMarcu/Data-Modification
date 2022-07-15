"""
    Script for computing robustness to occlusion according to 2 measures: cutOcclusion and iOcclusion.
    * cutOcclusion measures the raw model accuracy after masking out rectangular regions of an image.
    * iOcclusion is given by |(A_train_p - A_test_p)/(A_train - A_test)|, where A_{train/test} gives the
    train/test accuracy, while A_{train/test}_p gives the train/test accuracy when a proportion p of
    the image is occluded.

    Note that for masking with iOcclusion we provide 2 variants:
    - random masking (noisier, but less computationally intensive): iOcclusion_random
    - saliency-based masking: iOcclusion_gradcam
"""

import argparse
import ast

import random

import torch
from torch.utils.data import DataLoader

from models.resnet import ResNet18
from datasets.datasets import ds, dsmeta
from gradcam import ResNet_CAM, get_grad_cam
from aug_dataset import AugmentedDataset

parser = argparse.ArgumentParser(description='Compute occlusion measurement of a ResNet18 model')
parser.add_argument('--measurement', type=str, default='iOcclusion_gradcam', choices=['iOcclusion_gradcam',
                                                                                      'cutOcclusion',
                                                                                      'iOcclusion_random', 'none'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'fashion', 'imagenet'])
parser.add_argument('--augment', type=ast.literal_eval, default=False, help='use standard augmentation')
parser.add_argument('--device', default='cuda', type=str, help='Device on which to run')
parser.add_argument('--proportion', default=0.1, type=float, help='proportion of the image to be occluded')
parser.add_argument('--model-path', type=str)
parser.add_argument('--dataset-path', type=str, default=None, help='Optional dataset path')
parser.add_argument('--dataset-proportion', default=1., type=float, help='proportion of the dataset to evaluate on')
args = parser.parse_args()

distortions = {
    'iOcclusion_gradcam': 'none',
    'cutOcclusion': 'cutOut_restricted',
    'iOcclusion_random': 'fout'
}

random.seed(0)
device = args.device if torch.cuda.is_available() else "cpu"
batch_size = 1
data = ds[args.dataset]
meta = dsmeta[args.dataset]
classes, nc, size = meta['classes'], meta['nc'], meta['size']

trainset, valset, _ = data(args)

# For computational reasons, measures can be estimated on a subset of the train and test data.
# The default setting is to use the entire set.
train_subset = torch.utils.data.Subset(trainset,
                                       torch.randperm(len(trainset))[:int(len(trainset) * args.dataset_proportion)])
val_subset = torch.utils.data.Subset(valset,
                                     torch.randperm(len(valset))[:int(len(valset) * args.dataset_proportion)])
trainloader_distorted = DataLoader(AugmentedDataset(train_subset, distortions[args.measurement],
                                                    args.proportion),
                                   batch_size=batch_size, shuffle=True, num_workers=8)
valloader_distorted = DataLoader(AugmentedDataset(val_subset, distortions[args.measurement],
                                                  args.proportion), batch_size=batch_size,
                                 shuffle=True, num_workers=8)
trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=8)


def get_acc(model: torch.nn.Module, loader: DataLoader, proportion: float, cam_net: ResNet_CAM = None) -> float:
    """ Computes the model accuracy on provided data. For iOcclusion, the distortion is created based
    on the saliency map.
    Args:
        * model: model to be evaluated
        * cam_net: model with hooked gradients. Only used for saliency-based masking
        * loader: data loader for distorted data (note distortion can be 'none')
        * proportion: proportion of image to be occluded
    Returns:
        Model accuracy
    """
    model.eval()
    correct = 0
    total = 0

    for image, labels in loader:
        image = image.to(device)
        labels = labels.to(device)
        if cam_net:
            image = get_grad_cam(cam_net, image, proportion)
        outputs = model(image)

        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


model = ResNet18(10).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage)['model'])

if 'iOcclusion' in args.measurement:
    cam_net = ResNet_CAM(model, 6).to(device)
else:
    cam_net = None

val_i = get_acc(model, valloader_distorted, args.proportion, cam_net)
train_i = get_acc(model, trainloader_distorted, args.proportion, cam_net)
val = get_acc(model, valloader, args.proportion, None)
train = get_acc(model, trainloader, args.proportion, None)

if 'iOcclusion' in args.measurement:
    print(abs(train_i - val_i) / abs(train - val))
else:
    print(val_i)