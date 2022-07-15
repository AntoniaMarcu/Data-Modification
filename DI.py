"""
    Script for computing DI index from multiple model instances. The models are expected to follow the naming convention
    'path_{run_number}' where run_number is in [0,n_runs].
"""

import ast
import argparse
import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader

import distortions
from models.models import get_model
from datasets.datasets import ds, dsmeta
from aug_dataset import AugmentedDataset

parser = argparse.ArgumentParser(description='Compute DI index of 5 models for a specified distortion')
parser.add_argument('--distortion', type=str, default='cutOut_restricted', choices=['cutOut_restricted',
                                                                                    'cutOut_unrestricted',
                                                                                    'cutMix_restricted',
                                                                                    'cutMix_unrestricted',
                                                                                    'patch_shuffle'])
parser.add_argument('--device', default='cuda', type=str, help='Device on which to run')
parser.add_argument('--model', default="ResNet18", type=str, help='model type')
parser.add_argument('--model-path', type=str)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'fashion', 'imagenet'])
parser.add_argument('--dataset-path', type=str, default=None, help='Optional dataset path')
parser.add_argument('--augment', type=ast.literal_eval, default=False, help='use standard augmentation')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--n-runs', default=5, type=int, help='number of model runs')
parser.add_argument('--grid-size', default=4, type=int, help='batch size')
args = parser.parse_args()

args.device = 'cpu'
random.seed(0)
data = ds[args.dataset]
meta = dsmeta[args.dataset]
classes, nc, size = meta['classes'], meta['nc'], meta['size']
device = args.device if torch.cuda.is_available() else "cpu"

trainset, valset, _ = data(args)
valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=8)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

if 'cut' in args.distortion:
    valloader_aug = DataLoader(AugmentedDataset(valset, args.distortion, 1., imbalanced=True, sample=True),
                               batch_size=args.batch_size, shuffle=True, num_workers=8)
else:
    valloader_aug = DataLoader(AugmentedDataset(valset, args.distortion, args.grid_size),
                               batch_size=args.batch_size, shuffle=True, num_workers=8)


def get_wrong_predictions(net: torch.nn.Module, loader: DataLoader):
    """ Computes a class-wise histogram of incorrect predictions
    Args:
        * loader of data on which the model is evaluated
    Returns:
        * torch.Tensor of shape [classes] which gives the number of incorrect
        predictions for each class
    """
    net.to(device)
    net.eval()
    cnt = torch.zeros(classes)

    for image, labels in loader:
        image = image.to(device)
        labels = labels.to(device)
        outputs = net(image)
        _, predicted = torch.max(outputs.detach(), 1)
        incorrect = predicted != labels
        cnt[predicted[incorrect]] += 1
    return cnt / torch.sum(cnt)


def get_index(dif: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
    """ Computes the DI index
    Args:
        * dif: tensor of shape [n_runs, classes] which gives
               the difference in incorrect predictions for each run
    Returns:
        * a tuple formed of the DI index and its associated std.
    """
    increase = torch.clamp(dif, 0) * dif.shape[1] / 100
    cls = torch.argmax(increase.mean(axis=0))
    proportion = increase[:, cls] / increase.sum(axis=1)
    max_ratio = increase[:, cls].mean()
    di = proportion * max_ratio
    return di.mean(), di.std()


undistorted = torch.zeros((args.n_runs, classes))
distorted = torch.zeros((args.n_runs, classes))
for run in range(0, args.n_runs):
    net = get_model(args, classes, nc)
    net.load_state_dict(
        torch.load(args.model_path + f'{run}.pt', map_location=lambda storage, loc: storage)['model'])
    undistorted[run] = get_wrong_predictions(net, valloader)
    distorted[run] = get_wrong_predictions(net, valloader_aug)
print(get_index(distorted - undistorted))
