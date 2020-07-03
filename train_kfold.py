from __future__ import print_function, division, absolute_import
import argparse
import os
import time
import numpy as np
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from albumentations import (Compose, RandomCrop, Normalize, HorizontalFlip, Resize, RandomResizedCrop, OneOf,
                            ShiftScaleRotate, IAAAdditiveGaussianNoise, CenterCrop,
                            RandomBrightnessContrast, CoarseDropout)
from albumentations.pytorch import ToTensorV2
from utils.utils import save_checkpoint
from utils.utils import metric_average

from utils.metric import AverageMeter
from utils.data import MetDataset
from utils.utils import get_learning_rate

import torch.utils.data.distributed
import horovod.torch as hvd

rand_seed = 2013
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['PYTHONHASHSEED'] = str(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)


parser = argparse.ArgumentParser(description='PyTorch Product Training')

parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--base_lr', default=1e-1, type=float)
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--output', default='result')

parser.add_argument('--distribute', type=str, default='')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def train(epoch):
    losses = AverageMeter()
    # switch to train mode
    model.train()
    if args.distribute:
        train_sampler.set_epoch(epoch)
    correct = 0
    preds = []
    train_labels = []
    for i, (image, label) in enumerate(train_loader):
        rate = get_learning_rate(optimizer)
        image, label = image.cuda(), label.cuda()

        output = model(image)
        loss = criterion(output, label)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Rate:{rate}\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch, i, len(train_loader), rate=rate, loss=losses))

    return


def validate():
    losses = AverageMeter()
    val_acc1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    for i, (image, label) in enumerate(val_loader):
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            # compute output
            output = model(image)
            loss = criterion(output, label)

            # statistics
            val_acc = accuracy(output,label)
            val_acc1.update(val_acc.item(), image.size(0))
            losses.update(loss.item(), image.size(0))
            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                print('[TEST]: {0}/{1}\tLoss {loss.val:.5f} ({loss.avg:.5f})'.format(i, len(val_loader), loss=losses))

    if args.distribute:
        # Horovod: average metric values across workers.
        val_acc1.avg = metric_average(val_acc1.avg, 'val_acc')
        losses.vag = metric_average(losses.avg, 'losses.avg')

    return val_acc1.avg, losses.avg


class MetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18()

    def forward(self, x):
        # batch_size, C, H, W = x.shape
        x = self.backbone(x)
        return x


args = parser.parse_args()


# using distributed training
if args.distribute:
    # Horovod: initialize library.
    hvd.init()
    print("\n********using distributed training:%d***********"%hvd.rank())
    torch.cuda.set_device(hvd.local_rank())
    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)
if (not args.distribute) or (args.distribute and hvd.rank() == 0):
    print(args)

output_dir = args.output
os.makedirs(output_dir + '/checkpoint', exist_ok=True)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# print(model)
train_augment = Compose([
    RandomResizedCrop(224, 224), 
    HorizontalFlip(p=0.5),
    Normalize(mean, std),
    ToTensorV2(),

])
val_augment = Compose([
    Resize(256, 256),
    CenterCrop(224, 224),
    Normalize(mean, std),
    ToTensorV2(),
    ])

best_val_score = [0, 0, 0, 0, 0]
for fold in [0]:  # [0, 1, 2, 3, 4]
    if (not args.distribute) or (args.distribute and hvd.rank() == 0):
        print('\n######################## training fold {} #######################'.format(fold))
    model = MetModel()
    # optionally resume from a checkpoint
    if args.resume and (not args.distribute or (args.distribute and hvd.rank() == 0)):
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print(type(checkpoint))
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_dataset = MetDataset(train_augment, 'train.txt'.format(fold), state='train')
    val_dataset = MetDataset(val_augment, 'test.txt'.format(fold), state='val')
    train_sampler = None
    val_sampler = None

    lr_scaler = 1
    if args.distribute:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        model.cuda()
        lr_scaler = 1
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            print("If using GPU Adasum allreduce, scale learning rate by local_size")
            lr_scaler = hvd.local_size()
            # lr_scaler = 1
    else:
        model = nn.DataParallel(model)
        model.cuda()

    train_loader = DataLoader(
                        train_dataset,
                        shuffle=(train_sampler is None),
                        sampler=train_sampler,
                        batch_size  = args.batch_size if args.distribute else args.batch_size*4,
                        num_workers = args.workers,
                        pin_memory  = True)
    val_loader = DataLoader(
                        val_dataset,
                        sampler= val_sampler,
                        batch_size  = args.batch_size if args.distribute else args.batch_size*4,
                        num_workers = args.workers,
                        pin_memory  = True)

    criterion = F.cross_entropy
    # Optimizer
    finetuned_params = []
    new_params = []
    for n, p in model.named_parameters():
        if n.find('classify') >= 0:
            new_params.append(p)
        else:
            finetuned_params.append(p)
    param_groups = [{'params': finetuned_params, 'lr': args.base_lr*lr_scaler},
                    {'params': new_params, 'lr': args.base_lr*1*lr_scaler}]

    # optimizer = torch.optim.Adam(param_groups, weight_decay=1e-4)
    # optimizer = torch.optim.Adam([{'params': new_params, 'lr': args.base_lr*1*lr_scaler}], weight_decay=1e-4)
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=4, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)

    if args.distribute:
        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             compression=compression,
                                             op=hvd.Adasum if args.use_adasum else hvd.Sum) #hvd.Average备选，建议使用sum，lr换算简单
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.evaluate:
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        # train for one epoch
        train(epoch)

        # evaluate on validation set
        val_score, val_loss = validate()
        scheduler.step()

        lr_rate_1 = get_learning_rate(optimizer)
        epoch_time = time.time() - epoch_start
        if (not args.distribute) or (args.distribute and hvd.rank() == 0):
            print('Epoch[{0}] LR: {lr} Time:{time:.6f} '
                  'ValLoss {val_loss:.6f}  '
                  'Val_Score {val_score:.6f}'.format(
                   epoch, lr=lr_rate_1, time=epoch_time, val_loss=val_loss, val_score=val_score))

        is_best = val_score > best_val_score[fold]
        best_val_score[fold] = max(val_score, best_val_score[fold])
        if is_best:
            if (not args.distribute) or (args.distribute and hvd.rank() == 0):
                print("--------current best-------:%f" % best_val_score[fold])

        if (not args.distribute) or (args.distribute and hvd.rank() == 0):
            save_checkpoint({
                'fold': fold,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_score': best_val_score[fold],
                'score': val_score,
            }, is_best, output_dir)






