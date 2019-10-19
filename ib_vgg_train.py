from __future__ import print_function
import os
import time
import argparse
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from ib_vgg import *

def main():
    if args.ib_lr == -1:
        # if not specified, keep it the same as args.lr
        args.ib_lr = args.lr

    if args.ib_wd == -1:
        args.ib_wd = args.weight_decay

    if not os.path.exists(args.tb_path):
        os.makedirs(args.tb_path)
    writer = SummaryWriter(args.tb_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    n_cls = 10 if args.data_set == 'cifar10' else 100
    dset_string = 'datasets.CIFAR10' if args.data_set == 'cifar10' else 'datasets.CIFAR100'
    train_tfms = [transforms.ToTensor(), normalize]
    if not args.ban_flip:
        train_tfms = [transforms.RandomHorizontalFlip()] + train_tfms
    if not args.ban_crop:
        train_tfms = [transforms.RandomCrop(32, 4)] + train_tfms

    train_loader = torch.utils.data.DataLoader(
        eval(dset_string)(root='./data', train=True, transform=transforms.Compose(train_tfms), download=True),
        batch_size=args.batchsize, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        eval(dset_string)(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batchsize, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = VGG_IB(config=args.cfg, mag=args.mag, batch_norm=args.batch_norm, 
                    threshold=args.threshold, init_var=args.init_var, 
                    sample_in_training=args.sample_train, sample_in_testing=args.sample_test, 
                    n_cls=n_cls, no_ib=args.no_ib)
    model.cuda()

    ib_param_list, ib_name_list, cnn_param_list, cnn_name_list = [], [], [], []
    for name, param in model.named_parameters():
        if 'z_mu' in name or 'z_logD' in name:
            ib_param_list.append(param)
            ib_name_list.append(name)
        else:
            cnn_param_list.append(param)
            cnn_name_list.append(name)
    print('detected VIB params ({}): {}'.format(len(ib_name_list), ib_name_list))
    print('detected VGG params ({}): {}'.format(len(cnn_name_list), cnn_name_list))

    print('Learning rate of IB: {}, learning rate of others: {}'.format(args.ib_lr, args.lr))
    if args.opt.lower() == 'sgd':
        optimizer = torch.optim.SGD([{'params': ib_param_list, 'lr': args.ib_lr, 'weight_decay': args.ib_wd}, 
                                     {'params': cnn_param_list, 'lr': args.lr, 'weight_decay':args.weight_decay}], 
                                    momentum=args.momentum)
    elif args.opt.lower() == 'adam':
        optimizer = torch.optim.Adam([{'params': ib_param_list, 'lr': args.ib_lr, 'weight_decay': args.ib_wd}, 
                                      {'params': cnn_param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}])
    torch.backends.cudnn.benchmark = True
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    start_epoch = 0
    if args.resume != '':
        # resume from interrupted training
        state_dict = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['state_dict'])
        if 'opt_state_dict' in state_dict:
            optimizer.load_state_dict(state_dict['opt_state_dict'])
        model.print_compression_ratio(args.threshold)
        start_epoch = state_dict['epoch']
        print('loaded checkpoint {} at epoch {} with acc {}'.format(args.resume, state_dict['epoch'], state_dict['prec1'])) 
    
    if args.resume_vgg_pt:
        # VGG model trained without IB params
        state_dict = torch.load(args.resume_vgg_pt, map_location='cpu')
        try:
            print('loaded pretraind model with acc {}'.format(state_dict['best_prec1']))
        except:
            pass
        # match the state dicts
        ib_keys, vgg_keys = list(model.state_dict().keys()), list(state_dict['state_dict'].keys())
        ib_group_size = 10 if any(['num_batches_tracked' in key for key in ib_keys]) else 9
        for i in range(13):
            for j in range(6):
                ib_key = ib_keys[i*ib_group_size+j]
                vgg_key = vgg_keys[i*6+j]
                model.state_dict()[ib_key].copy_(state_dict['state_dict'][vgg_key])
        ib_offset, vgg_offset = ib_group_size*13, 6*13
        for i in range(3):
            for j in range(2):
                model.state_dict()[ib_keys[ib_offset + i*5 + j]].copy_(state_dict['state_dict'][vgg_keys[vgg_offset + i*2+j]])

    if args.resume_vgg_vib:
        # VGG model trained without IB params
        state_dict = torch.load(args.resume_vgg_vib)
        print('loaded pretraind model with acc {}'.format(state_dict['prec1']))
        # match the state dicts
        ib_keys, vgg_keys = list(model.state_dict().keys()), list(state_dict['state_dict'].keys())
        ib_group_size = 10 if any(['num_batches_tracked' in key for key in ib_keys]) else 9
        vgg_group_size = 10 if any(['num_batches_tracked' in key for key in vgg_keys]) else 9
        for i in range(13):
            for j in range(6):
                model.state_dict()[ib_keys[i*ib_group_size+j]].copy_(state_dict['state_dict'][ib_keys[i*ib_group_size+j]])
        ib_offset, vgg_offset = ib_group_size*13, 6*13
        for i in range(2):
            for j in range(2):
                model.state_dict()[ib_keys[ib_offset + i*5 + j]].copy_(state_dict['state_dict'][vgg_keys[vgg_group_size*13 + i*5 + j]])

    if args.val:
        model.eval()
        validate(val_loader, model, criterion, 0, None)
        return

    best_acc = -1

    for epoch in range(start_epoch, args.epochs):
        optimizer.param_groups[0]['lr'] = args.ib_lr * (args.lr_fac ** (epoch//args.lr_epoch))
        optimizer.param_groups[1]['lr'] = args.lr * (args.lr_fac ** (epoch//args.lr_epoch))

        train(train_loader, model, criterion, optimizer, epoch, writer)
        model.print_compression_ratio(args.threshold, writer, epoch)

        prune_acc = validate(val_loader, model, criterion, epoch, writer)
        writer.add_scalar('test_acc', prune_acc, epoch)

        if prune_acc > best_acc:
            best_acc = prune_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'prec1': best_acc,
            }, os.path.join(args.save_dir, 'best_prune_acc.pth'))
        torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'prec1': prune_acc,
            }, os.path.join(args.save_dir, 'last_epoch.pth'))
    print('Best accuracy: {}'.format(best_acc))

def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
    Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kld_meter = AverageMeter()
    top1 = AverageMeter()

    forward_time = AverageMeter()
    kl_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    start_iter = len(train_loader)*epoch
    kl_fac = args.kl_fac if not args.no_ib else 0
    print('kl fac:{}'.format((kl_fac)))
    for i, (input, target) in enumerate(train_loader):
        ite = start_iter + i

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)

        # compute output
        compute_start = time.time()
        if args.no_ib:
            output = model(input_var)
        else:
            output, kl_total = model(input_var)
            writer.add_scalar('train_kld', kl_total.data, ite)
        forward_time.update(time.time() - compute_start)

        ce_loss = criterion(output, target_var)
        
        loss = ce_loss
        if kl_fac > 0:
            loss += kl_total * kl_fac

        # compute gradient and do SGD step
        optimizer.zero_grad()
        compute_start = time.time()
        loss.backward()
        backward_time.update(time.time()-compute_start)

        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(ce_loss.item(), input.size(0))
        kld_meter.update(kl_total.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Date: {date}\t'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Forward Time {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
                  'KL Time {kl_time.val:.3f} ({kl_time.avg:.3f})\t'
                  'Backward Time {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
                  'CE {loss.val:.4f} ({loss.avg:.4f})\t'
                  'KLD {klds.val:.4f} ({klds.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), date=time.strftime("%Y-%m-%d %H:%M:%S"), batch_time=batch_time,
                      forward_time=forward_time, backward_time=backward_time, kl_time=kl_time,
                      data_time=data_time, loss=losses, klds=kld_meter, top1=top1))
    print('Date: {date}\t'
        'Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Forward Time {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
        'KL Time {kl_time.val:.3f} ({kl_time.avg:.3f})\t'
        'Backward Time {backward_time.val:.3f} ({backward_time.avg:.3f})\t'
        'CE {loss.val:.4f} ({loss.avg:.4f})\t'
        'KLD {klds.val:.4f} ({klds.avg:.4f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            epoch, i, len(train_loader), date=time.strftime("%Y-%m-%d %H:%M:%S"), batch_time=batch_time,
            forward_time=forward_time, backward_time=backward_time, kl_time=kl_time,
            data_time=data_time, loss=losses, klds=kld_meter, top1=top1))
    writer.add_scalar('train_ce_loss', losses.avg, epoch)
    

def validate(val_loader, model, criterion, epoch, writer, masks=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    

    return top1.avg
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #pdb.set_trace()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Total number of epochs.')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--save-dir', type=str, default='ib_vgg_chk',
                        help='Path to save the checkpoints')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Threshold of alpha. For pruning.')
    parser.add_argument('--kl-fac', type=float, default=1e-6,
                        help='Factor for the KL term.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Which GPU to use. Single GPU only.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--ib-lr', type=float, default=-1, 
                        help='Separate learning rate for information bottleneck params. Set to -1 to follow args.lr.')
    parser.add_argument('--ib-wd', type=float, default=-1,
                        help='Separate weight decay for information bottleneck params. Set to -1 to follow args.weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum')
    parser.add_argument('--mag', type=float, default=9,
                        help='Initial magnitude for the variances.')
    parser.add_argument('--lr-fac', type=float, default=0.5,
                        help='LR decreasing factor.')
    parser.add_argument('--lr-epoch', type=int, default=30,
                        help='Decrease learning rate every x epochs.')
    parser.add_argument('--tb-path', type=str, default='tb_ib_vgg',
                        help='Path to store tensorboard data.')
    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='Whether to use batch norm')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='Optimizer. sgd or adam.')
    parser.add_argument('--val', action='store_true', default=False,
                        help='Whether to only evaluate model.')
    parser.add_argument('--cfg', type=str, default='D0',
                        help='VGG net config.')
    parser.add_argument('--data-set', type=str, default='cifar10',
                        help='Which data set to use.')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a model to be resumes (with its optimizer states).')
    parser.add_argument('--resume-vgg-vib', type=str, default='',
                        help='Path to pretrained VGG model (with IB params), ignore IB params.')
    parser.add_argument('--resume-vgg-pt', type=str, default='',
                        help='Path to pretrained VGG model (without IB params).')
    parser.add_argument('--init-var', type=float, default=0.01, 
                        help='Variance for initializing IB parameters')
    parser.add_argument('--reg-weight', type=float, default=0)
    parser.add_argument('--ban-crop', default=False, action='store_true',
                        help='Whether to ban random cropping after padding.')
    parser.add_argument('--ban-flip', default=False, action='store_true',
                        help='Whether to ban random flipping.')
    parser.add_argument('--sample-train', default=1, type=int,
                        help='Set to non-zero to sample during training.')
    parser.add_argument('--sample-test', default=0, type=int,
                        help='Set to non-zero to sampling during test.')
    parser.add_argument('--no-ib', default=False, action='store_true',
                        help='Ignore IB operators.')
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()
    print(args)
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    main()
