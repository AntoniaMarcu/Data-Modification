from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, ImageNet, MNIST
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, \
    Resize, CenterCrop

from datasets.tiny_imagenet import TinyImageNet
from utils import split


def cifar_transforms(args):
    normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_base = [ToTensor(), normalize]

    if args.augment:
        transform = [RandomCrop(32, padding=4), RandomHorizontalFlip()] + transform_base
    else:
        transform = transform_base

    transform_train = Compose(transform)
    transform_test = Compose(transform_base)
    return transform_train, transform_test


def fashion_transforms(args):
    normalise = Normalize((0.1307,), (0.3081,))
    base = [ToTensor(), normalise]

    if args.augment:
        transform = [RandomCrop(28, padding=4), RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose(base)
    return transform_train, transform_test

def mnist_transforms(args):
    normalise = Normalize((0.1307,), (0.3081,))
    base = [ToTensor(), normalise]

    if args.augment:
        transform = [RandomCrop(28, padding=4), RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose(base)
    return transform_train, transform_test


def gst_transforms(args):
    normalise = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    base = [ToTensor(), normalise]
    transform_train = Compose([RandomResizedCrop(224), RandomHorizontalFlip()] + base)
    transform_test = Compose([Resize(256), CenterCrop(224)] + base)
    return transform_train, transform_test


def imagenet_transforms(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [ToTensor(), normalize]

    if args.augment:
        transform = [RandomResizedCrop(224), RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose([Resize(256), CenterCrop(224)] + base)
    return transform_train, transform_test


def tinyimagenet_transforms(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [ToTensor(), normalize, ]

    if args.augment:
        transform = [RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose([*base])
    return transform_train, transform_test


dstransforms = {
    'cifar10': cifar_transforms,
    'cifar100': cifar_transforms,
    'cifar10h': cifar_transforms,
    'fashion': fashion_transforms,
    'tinyimagenet': tinyimagenet_transforms,
    'imagenet': imagenet_transforms,
    'gst': gst_transforms,
    'mnist': mnist_transforms
}


@split
def cifar(args):
    cifar = CIFAR100 if args.dataset == 'cifar100' else CIFAR10
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = args.dataset_path if args.dataset_path is not None else './data'
    trainset = cifar(root=root, train=True, download=True, transform=transform_train)
    valset = cifar(root=root, train=False, download=True, transform=transform_test)
    return trainset, valset

@split
def gst(args):
    gst = ImageFolder
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = args.dataset_path if args.dataset_path is not None else './data'
    trainset = gst(root=root, transform=transform_train)
    testset = gst(root=root, transform=transform_test)

    return trainset, testset


@split
def mnist(args):
    mnist = MNIST
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = args.dataset_path if args.dataset_path is not None else './data'
    trainset = mnist(root=root, train=True, download=True, transform=transform_train)
    valset = mnist(root=root, train=False, download=True, transform=transform_test)
    return trainset, valset

@split
def fashion(args):
    data = FashionMNIST
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = './data/fashion'
    root = args.dataset_path if args.dataset_path is not None else root
    trainset = data(root=root, train=True, download=True, transform=transform_train)
    valset = data(root=root, train=False, download=True, transform=transform_test)
    return trainset, valset


@split
def imagenet(args):
    data = ImageFolder
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = '/ssd/ILSVRC2012' if args.dataset_path is None else args.dataset_path

    trainset = data(root=root, transform=transform_train)
    testset = data(root=root, transform=transform_test)

    return trainset, testset


@split
def tinyimagenet(args):
    data = TinyImageNet
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = '/ssd/tinyimagenet' if args.dataset_path is None else args.dataset_path

    trainset = data(root=root, transform=transform_train)
    valset = data(root=root, transform=transform_test, train=False)
    return trainset, valset


ds = {
    'cifar10': cifar,
    'cifar100': cifar,
    'fashion': fashion,
    'imagenet': imagenet,
    'tinyimagenet': tinyimagenet,
    'gst': gst,
    'mnist': mnist
}

dsmeta = {
    'cifar10': {'classes': 10, 'nc': 3, 'size': (32, 32)},
    'cifar100': {'classes': 100, 'nc': 3, 'size': (32, 32)},
    'fashion': {'classes': 10, 'nc': 1, 'size': (28, 28)},
    'imagenet': {'classes': 1000, 'nc': 3, 'size': (224, 224)},
    'tinyimagenet': {'classes': 200, 'nc': 3, 'size': (64, 64)},
    'gst': {'classes': 16, 'nc': 3, 'size': (224, 224)},
    'mnist': {'classes': 10, 'nc': 1, 'size': (28, 28)}
}
