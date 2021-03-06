import argparse
import ast
from datetime import datetime

import pandas as pd
import torch
import torchbearer
from torch import nn, optim
from torchbearer import Trial
from torchbearer.callbacks import MultiStepLR, CosineAnnealingLR
from torchbearer.callbacks import TensorBoard, TensorBoardText, Cutout, RandomErase
from torchbearer.callbacks import CutMix as RCutMix
from datasets.datasets import ds, dstransforms, dsmeta
from implementations.torchbearer_implementation import PointNetFMix
from utils.reformulated_fmix_100 import FMix_100
from utils.reformulated_fixedfmix import FixedFMix
from models.models import get_model
from utils import MSDAAlternator, WarmupLR
from utils.reformulated_mixup_all import RMixup
from datasets.toxic import ToxicHelper


# Setup
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'reduced_cifar', 'fashion', 'imagenet', 'tinyimagenet',
                             'commands', 'modelnet', 'toxic', 'mnist'])
parser.add_argument('--other', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'reduced_cifar', 'fashion', 'imagenet', 'tinyimagenet',
                             'commands', 'modelnet', 'toxic', 'mnist'])
parser.add_argument('--dataset-path', type=str, default=None, help='Optional dataset path')
parser.add_argument('--other-dataset-path', type=str, default=None, help='Optional dataset path')
parser.add_argument('--split-fraction', type=float, default=1., help='Fraction of total data to train on for reduced_cifar dataset')
parser.add_argument('--pointcloud-resolution', default=128, type=int, help='Resolution of pointclouds in modelnet dataset')
parser.add_argument('--model', default="ResNet18", type=str, help='model type')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--train-steps', type=int, default=None, help='Number of training steps to run per "epoch"')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-warmup', type=ast.literal_eval, default=False, help='Use lr warmup')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--device', default='cuda', type=str, help='Device on which to run')
parser.add_argument('--num-workers', default=7, type=int, help='Number of dataloader workers')

parser.add_argument('--auto-augment', type=ast.literal_eval, default=False, help='Use auto augment with cifar10/100')
parser.add_argument('--augment', type=ast.literal_eval, default=True, help='use standard augmentation (default: True)')
parser.add_argument('--parallel', type=ast.literal_eval, default=False, help='Use DataParallel')
parser.add_argument('--reload', type=ast.literal_eval, default=False, help='Set to resume training from model path')
parser.add_argument('--verbose', type=int, default=2, choices=[0, 1, 2])
parser.add_argument('--seed', default=0, type=int, help='random seed')

# Augs
parser.add_argument('--random-erase', default=False, type=ast.literal_eval, help='Apply random erase')
parser.add_argument('--cutout', default=False, type=ast.literal_eval, help='Apply Cutout')
parser.add_argument('--msda-mode', default=None, type=str, choices=['fmix', 'fixedfmix', 'cutmix', 'mixup', 'alt_mixup_fmix',
                                                                    'alt_mixup_cutmix', 'alt_fmix_cutmix', 'None'])

# Aug Params
parser.add_argument('--alpha', default=1., type=float, help='mixup/fmix interpolation coefficient')
parser.add_argument('--f-decay', default=3.0, type=float, help='decay power')
parser.add_argument('--cutout_l', default=16, type=int, help='cutout/erase length')
parser.add_argument('--reformulate', default=False, type=ast.literal_eval, help='Use reformulated fmix/mixup')
parser.add_argument('--lam-train', default=False, type=ast.literal_eval)

# Scheduling
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs.')
parser.add_argument('--cosine-scheduler', type=ast.literal_eval, default=False, help='Set to use a cosine scheduler instead of step scheduler')

# Cross validation
parser.add_argument('--fold-path', type=str, default='./data/folds.npz', help='Path to object storing fold ids. Run-id 0 will regen this if not existing')
parser.add_argument('--n-folds', type=int, default=6, help='Number of cross val folds')
parser.add_argument('--fold', type=str, default='test', help='One of [1, ..., n] or "test"')

# Logs
parser.add_argument('--run-id', type=int, default=0, help='Run id')
parser.add_argument('--log-dir', default='./logs/testing', help='Tensorboard log dir')
parser.add_argument('--model-file', default='./saved_models/model.pt', help='Path under which to save model. eg ./model.py')
args = parser.parse_args()


if args.seed != 0:
    torch.manual_seed(args.seed)

print('==> Preparing data..')
data = ds[args.dataset]
meta = dsmeta[args.dataset]
classes, nc, size = meta['classes'], meta['nc'], meta['size']

trainset, valset, testset = data(args)

# Toxic comments uses its own data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if not args.dataset == 'toxic' else trainset
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if not args.dataset == 'toxic' else valset
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if not args.dataset == 'toxic' else testset

trainset_other = trainset
trainloader_other = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) if not args.dataset == 'toxic' else trainset
if args.reformulate:
    # OTHER
    args.dataset_path=args.other_dataset_path
    args.dataset = args.other
    other_data = ds[args.other]
    trainset_other, _, _ = other_data(args)
    trainloader_other = torch.utils.data.DataLoader(trainset_other, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)



print('==> Building model..')
net = get_model(args, classes, nc)
net = nn.DataParallel(net) if args.parallel else net
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

print('==> Setting up callbacks..')
current_time = datetime.now().strftime('%b%d_%H-%M-%S') + "-run-" + str(args.run_id)
tboard = TensorBoard(write_graph=False, comment=current_time, log_dir=args.log_dir)
tboardtext = TensorBoardText(write_epoch_metrics=False, comment=current_time, log_dir=args.log_dir)


@torchbearer.callbacks.on_start
def write_params(_):
    params = vars(args)
    params['schedule'] = str(params['schedule'])
    df = pd.DataFrame(params, index=[0]).transpose()
    tboardtext.get_writer(tboardtext.log_dir).add_text('params', df.to_html(), 1)


modes = {
    'fmix': FMix_100(trainloader_other, decay_power=args.f_decay, alpha=args.alpha, size=size, max_soft=0, reformulate=args.reformulate, fout=False, lam_train=args.lam_train),
    'fixedfmix': FixedFMix(trainloader_other, decay_power=args.f_decay, alpha=args.alpha, size=size, max_soft=0, reformulate=False, fout=False, lam_train=args.lam_train),
    'mixup': RMixup(trainloader_other, alpha=args.alpha, reformulate=args.reformulate, mixout=False, lam_train=args.lam_train),
    'cutmix': RCutMix(args.alpha, classes, True), #RCutMix(trainloader_other, args.alpha, classes, True, reformulate=args.reformulate),
    'pointcloud_fmix': PointNetFMix(args.pointcloud_resolution, decay_power=args.f_decay, alpha=args.alpha, max_soft=0,
                                    reformulate=args.reformulate, lam_train=args.lam_train)
}
modes.update({
    'alt_mixup_fmix': MSDAAlternator(modes['fmix'], modes['mixup']),
    'alt_mixup_cutmix': MSDAAlternator(modes['mixup'], modes['cutmix']),
    'alt_fmix_cutmix': MSDAAlternator(modes['fmix'], modes['cutmix']),
})

# Pointcloud fmix converts voxel grids back into point clouds after mixing
mode = 'pointcloud_fmix' if (args.msda_mode == 'fmix' and args.dataset == 'modelnet') else args.msda_mode

# CutMix callback returns mixed and original targets. We mix in the loss function instead
@torchbearer.callbacks.on_sample
def cutmix_reformat(state):
    state[torchbearer.Y_TRUE] = state[torchbearer.Y_TRUE][0]

cb = [tboard, tboardtext, write_params, torchbearer.callbacks.MostRecent(args.model_file)]
# Toxic helper needs to go before the msda to reshape the input
cb.append(ToxicHelper()) if args.dataset == 'toxic' else []
cb.append(modes[mode]) if args.msda_mode not in [None, 'None'] else []
cb.append(Cutout(1, args.cutout_l)) if args.cutout else []
cb.append(RandomErase(1, args.cutout_l)) if args.random_erase else []
# WARNING: Schedulers appear to be broken (wrong lr output) in some versions of PyTorch, including 1.4. We used 1.3.1
cb.append(MultiStepLR(args.schedule)) if not args.cosine_scheduler else cb.append(CosineAnnealingLR(args.epoch, eta_min=0.))
cb.append(WarmupLR(0.1, args.lr)) if args.lr_warmup else []
cb.append(cutmix_reformat) if args.msda_mode == 'cutmix' else []

# FMix loss is equivalent to mixup loss and works for all msda in torchbearer
if args.msda_mode not in [None, 'None']:
    bce = True if args.dataset == 'toxic' else False
    criterion = modes['fmix'].loss(bce)
elif args.dataset == 'toxic':
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

# from torchbearer.metrics.roc_auc_score import RocAucScore
print('==> Training model..')
trial = Trial(net, optimizer, criterion, metrics=['acc', 'loss', 'lr'], callbacks=cb)
trial.with_generators(train_generator=trainloader, val_generator=valloader, train_steps=args.train_steps, test_generator=testloader).to(args.device)

if args.reload:
    state = torch.load(args.model_file)
    trial.load_state_dict(state)
    trial.replay()
trial.run(args.epoch, verbose=args.verbose)
trial.evaluate(data_key=torchbearer.TEST_DATA)
