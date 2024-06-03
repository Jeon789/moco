# -*- coding: utf-8 -*-
"""moco_cifar10_demo

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

## MoCo Demo: CIFAR-10

This is a simple demo for training MoCo on CIFAR-10. It can be run directly in a Colab notebook using a publicly available GPU.

#### Results

These are the ResNet-18 classification accuracy of a **kNN monitor** on the unsupervised pre-training features.

| config | 200ep | 400ep | 800ep |
| --- | --- | --- | --- |
| Asymmetric | 82.6 | 86.3 | 88.7 |
| Symmetric | 85.3 | 88.5 | 89.7 |

#### Notes

* **Symmetric loss**: the original MoCo paper uses an *asymmetric* loss -- one crop is the query and the other crop is the key, and it backpropagates to one crop (query). Following SimCLR/BYOL, here we provide an option of a *symmetric* loss -- it swaps the two crops and computes an extra loss. The symmetric loss behaves like 2x epochs of the asymmetric counterpart: this may dominate the comparison results when the models are trained with a fixed epoch number.

* **SplitBatchNorm**: the original MoCo was trained in 8 GPUs. To simulate the multi-GPU behavior of BatchNorm in this 1-GPU demo, we provide a SplitBatchNorm layer. We set `--bn-splits 8` by default to simulate 8 GPUs. `--bn-splits 1` is analogous to SyncBatchNorm in the multi-GPU case.

* **kNN monitor**: this demo provides a kNN monitor on the test set. Training a linear classifier on frozen features often achieves higher accuracy. To train a linear classifier (not provided in this demo), we recommend using lr=30, wd=0, epochs=100 with a stepwise or cosine schedule. The ResNet-18 model (kNN 89.7) has 90.7 linear classification accuracy.

#### Disclaimer

This demo aims to provide an interface with a free GPU (thanks to Colab) for understanding how the code runs. We suggest users be careful to draw conclusions from CIFAR, which may not generalize beyond this dataset. We have verified that it is beneficial to have the momentum encoder (disabling it by `--moco-m 0.0` should fail), queue size (saturated at `--moco-k 4096`) and ShuffleBN (without which loses 4% at 800 epochs) on CIFAR, similar to the observations on ImageNet. But new observations made only on CIFAR should be judged with caution.

#### References
This demo is adapted from:
* http://github.com/zhirongw/lemniscate.pytorch
* https://github.com/leftthomas/SimCLR

### Prepare

Check GPU settings. A free GPU in Colab is <= Tesla P100. The log of the demo is based on a Tesla V100 from Google Cloud Platform.
"""

import utils

from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet
from tqdm import tqdm
import argparse
import json
import math
import os
import ast
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import str2bool, seed_everything, print_setting
import time
import random
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import warnings
import builtins



"""### Set arguments"""

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=800, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default="[120, 160]", type=ast.literal_eval, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', default=True, type=ast.literal_eval, help='use cosine lr schedule')

parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

parser.add_argument('--symmetric',default=False, type=ast.literal_eval, help='use a symmetric loss function that backprops to both crops')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# utils
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')


# GCL
parser.add_argument('--seed', default=527, type=int)
parser.add_argument('--GCL', default=True, type=str2bool)
parser.add_argument('--num_patch', default=8, type=int)
parser.add_argument('--sim', default='cos', type=str)
parser.add_argument('--multi_gpu', default=None, type=ast.literal_eval)
parser.add_argument('--fm_method', default=1, type=int)


parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)


'''
args = parser.parse_args('')  # running in ipynb
'''
args = parser.parse_args()  # running in command line
if args.results_dir == '':
    args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

print(args)
seed_everything(args.seed)

"""### Define data loaders"""

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2
    
class CIFAR10Grpoup(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root, train, transform, download, num_patch=8):
        super().__init__(root, train, transform, download)
        self.num_patch = num_patch

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        images1 = []
        if self.transform:
            for _ in range(self.num_patch):
                images1.append(self.transform(img))

        images2 = []
        if self.transform:
            for _ in range(self.num_patch):
                images2.append(self.transform(img))

        return torch.stack(images1), torch.stack(images2)
    

"""### Define base encoder"""

# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

"""### Define MoCo wrapper"""
class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True, num_patch=8, sim='cos'):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.num_patch = num_patch
        self.sim = sim

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
    
    
    def get_centroid(self, images_q, sim='cos'):
        """ images_q, images_k (B,G/2,3,32,32) -> im_q, im_k (B,3,32,32) """

        def _get_centroid1(tensors, sim=sim):
            if sim == 'dot':
                return torch.mean(tensors, dim=0)
            elif sim == 'cos' or sim == 'angle':
                temp_cen = tensors[0]
                for n in range(len(tensors)) :
                    if n != 0 :
                        add_vec = tensors[n]
                        inner = torch.dot(temp_cen, add_vec)
                        theta = torch.arccos(inner)
                        
                        y= torch.cos((n * theta) / (n + 1)) - torch.cos(theta) * torch.cos(theta / (n + 1))
                        y= y / (1 - (torch.cos(theta)) ** 2)
                        
                        x = torch.cos(theta / (n + 1)) - y
                        
                        temp_cen = x * temp_cen + y * add_vec
                        temp_cen = temp_cen / torch.linalg.norm(temp_cen, 2)
                return temp_cen


        def _get_centroid2(tensors, sim=sim):
            """ tensors =  tensor of size (num_gp,3,32,32)"""
            """ angle n:1 """
            if sim == 'dot':
                return torch.mean(tensors, dim=0)
            elif sim == 'cos' or sim == 'angle':
                temp_cen = tensors[0]
                for n in range(len(tensors)) :
                    if n != 0 :
                        next_vec = tensors[n]
                        theta = torch.arccos(torch.dot(temp_cen, next_vec))

                        tan_vec  = next_vec - torch.dot(next_vec, temp_cen) * temp_cen
                        tan_vec  = tan_vec / (torch.linalg.norm(tan_vec, 2) + 1e-9)
                        temp_cen = torch.cos( (1/n+1)*theta ) * temp_cen + torch.sin( (1/n+1)*theta ) * tan_vec
                        temp_cen = temp_cen / torch.linalg.norm(temp_cen, 2)
                return temp_cen
            
        def _get_centroid3(tensors, sim='cos'):
            """ tensors =  tensor of size (num_gp,3,32,32)"""
            """ Recursive one """
            if tensors.size()[0] == 1:
                return tensors[0]
            elif tensors.size()[0] == 2:
                return torch.nn.functional.normalize(tensors[0] + tensors[1], dim=0)
            elif tensors.size()[0] > 2 :
                mid = int(tensors.size()[0] / 2)
                temp1 = _get_centroid2(tensors[:mid], sim=sim)
                temp2 = _get_centroid2(tensors[mid:], sim=sim)
                return torch.nn.functional.normalize(temp1 + temp2, dim=0)

        batch_size = int(images_q.size()[0]/self.num_patch)
        q = []
        for i in range(batch_size):
            if args.fm_method == 1:
                q.append(_get_centroid1(images_q[i*self.num_patch:(i+1)*self.num_patch], sim=sim))
            elif args.fm_method == 2:
                q.append(_get_centroid2(images_q[i*self.num_patch:(i+1)*self.num_patch], sim=sim))
            elif args.fm_method == 3:
                q.append(_get_centroid3(images_q[i*self.num_patch:(i+1)*self.num_patch], sim=sim))

        return torch.stack(q)


    def group_contrastive_loss(self, images_q, images_k, sim='cos'):
        # compute query features
        q = self.encoder_q(images_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        q = self.get_centroid(q, sim=sim)


        # start = time.time()
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(images_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle) #여기서 저장
            k = self.get_centroid(k, sim=sim)
        # print(f"k : {time.time()-start:.4f}")

        # compute logits
        if sim == 'dot':
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        elif sim == 'cos' :
            l_pos = torch.einsum('nc,nc->n', [F.normalize(q, dim=1), F.normalize(k, dim=1)]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [F.normalize(q, dim=1), F.normalize(self.queue.clone().detach(), dim=0)])
        # elif sim == 'angle' :
        #     ##TODO
        #     angular_dist = torch.acos(nn.functional.cosine_similarity(centroid, pos_neg))
        #     similarity = (torch.pi - angular_dist) / temp
        #     Similarity.append(similarity)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # n,(k+1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, images_q, images_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.group_contrastive_loss(images_q, images_k, sim=self.sim)
            loss_21, q2, k1 = self.group_contrastive_loss(images_k, images_q, sim=self.sim)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.group_contrastive_loss(images_q, images_k, sim=self.sim)

        if not torch.isnan(loss).any():
            self._dequeue_and_enqueue(k)

        return loss

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    model = ModelMoCo(
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
            arch=args.arch,
            bn_splits=args.bn_splits,
            symmetric=args.symmetric,
            num_patch=args.num_patch,
            sim = args.sim,
        )
    # print(model.encoder_q)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # load model if resume
    epoch_start = 1
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))



    cudnn.benchmark = True

    # Data loading code
    
    # geometric transforms
    geo_transform = transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(180),
    # transforms.RandomAffine(180, shear=20),
    # transforms.RandomPerspective()
    ])
    color_transform = transforms.RandomChoice([ transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomGrayscale()])
    random_transform = transforms.RandomApply([transforms.RandomErasing()], p=0.5)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        color_transform,
        geo_transform,
        transforms.ToTensor(),
        # random_transform,  # 순서가 ToTensor 다음에 가야함
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    # data prepare
    train_data = CIFAR10Grpoup(root='data', train=True, transform=train_transform, download=True, num_patch=args.num_patch)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    memory_data = CIFAR10(root='data', train=True, transform=test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_data = CIFAR10(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)





    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    # training loop
    for epoch in range(epoch_start, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss = train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        # save model
    
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')




"""### Define train/test


"""

# train for one epoch
def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    # t = time.time()
    for i, (images1, images2) in enumerate(train_bar):  # (B,G/2,3,32,32) * 2
        # print(f"Data Load Time : {time.time()-t:.4f}")
        # t = time.time()

        # start = time.time()
        images1, images2 = images1.cuda(non_blocking=True), images2.cuda(non_blocking=True)
        images1, images2 = images1.view(-1,3,32,32), images2.view(-1,3,32,32)

        loss = net(images1, images2).mean()  # mean is for multi_gpu setting
        if torch.isnan(loss).any():
            print(f"NaN at iter{i+1}")
            continue

        # print(f"Forward Time : {time.time()-t:.4f}")
        # t = time.time()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        msg = 'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num)
        
        # print(f"Upadte Time : {time.time()-t:.4f}")
        # t = time.time()
        train_bar.set_description(msg)
        utils.write_log(args.results_dir + '/model_last.pth', msg)

        if total_num == 0:
            print("total_num이 0이 떠서 에러 잡기위해 멈춤.")
            breakpoint()
    return total_loss / total_num

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        msg = ''
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            msg = 'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100)
            test_bar.set_description(msg)
        utils.write_log(args.results_dir + '/log.txt', msg+'\n')

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """ (256,128), (128,50000), (50000), 10, """
    breakpoint()
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)  # (256,200), (256,200) 값이랑 위치
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices) #(256,200) sim_weight들의 라벨
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device) # 크기 맞춰서 재료만들기 (256*200, 10)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0) # sim_labels 자리에 1 넣어주기 (scatter)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1) #(256,10)
        # (256,200,10) * (256,200,1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels



if __name__ == "__main__":
    main()
