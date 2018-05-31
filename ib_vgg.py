import torch
from torch import nn
import numpy as np

from ib_layers import *

# model configuration, (out_channels, kl_multiplier), 'M': Mean pooling, 'A': Average pooling
cfg = {
    'D6': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D5': [(64, 1.0/32**2), (64, 1.0/32**2), 'M', (128, 1.0/16**2), (128, 1.0/16**2), 'M', (256, 1.0/8**2), (256, 1.0/8**2), (256, 1.0/8**2), 
        'M', (512, 1.0/4**2), (512, 1.0/4**2), (512, 1.0/4**2), 'M', (512, 1.0/2**2), (512, 1.0/2**2), (512, 1.0/2**2), 'M'],
    'D4': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D3': [(64, 0.1), (64, 0.1), 'M', (128, 0.5), (128, 0.5), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D2': [(64, 0.01), (64, 0.01), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D1': [(64, 0.1), (64, 0.1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D0': [(64, 1), (64, 1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'G':[(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'], # VGG 16 with one fewer FC
    'G5': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'A']
}

class VGG_IB(nn.Module):
    def __init__(self, config=None, mag=9, batch_norm=False, threshold=0, 
                init_var=0.01, sample_in_training=True, sample_in_testing=False, n_cls=10, no_ib=False):
        super(VGG_IB, self).__init__()

        self.init_mag = mag
        self.threshold = threshold
        self.config = config
        self.init_var = init_var
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        self.no_ib = no_ib

        self.conv_layers, conv_kl_list = self.make_conv_layers(cfg[config], batch_norm)
        print('Using structure {}'.format(cfg[config]))

        fc_ib1 = InformationBottleneck(512, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing)
        fc_ib2 = InformationBottleneck(512, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing)
        self.n_cls = n_cls
        if self.config in ['G', 'D6']:
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)] if no_ib else \
                            [nn.Linear(512, 512), nn.ReLU(), fc_ib1, nn.Linear(512, self.n_cls)] 
            self.fc_layers = nn.Sequential(*fc_layer_list)
            self.kl_list = conv_kl_list + [fc_ib1]
        elif self.config == 'G5':
            self.fc_layers = nn.Sequential(nn.Linear(512, self.n_cls))
            self.kl_list = conv_kl_list
        else:
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)] if no_ib else \
                    [nn.Linear(512, 512), nn.ReLU(), fc_ib1, nn.Linear(512, 512), nn.ReLU(), fc_ib2, nn.Linear(512, self.n_cls)]
            self.fc_layers = nn.Sequential(*fc_layer_list)
            self.kl_list = conv_kl_list + [fc_ib1, fc_ib2]

    def make_conv_layers(self, config, batch_norm):
        layers, kl_list = [], []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, padding=1)
                in_channels = v[0]
                ib = InformationBottleneck(v[0], mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    kl_mult=v[1], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                if not self.no_ib:
                    layers.append(ib)
                    kl_list.append(ib)
        return nn.Sequential(*layers), kl_list

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x).view(batch_size, -1)
        x = self.fc_layers(x)

        if self.training and (not self.no_ib):
            ib_kld = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                ib_kld += ib.kld
            
            return x, ib_kld
        else:
            return x

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib_layer.get_mask_hard(threshold) for ib_layer in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy()==0) for mask in masks]
        else:
            masks = [ib_layer.get_mask_weighted(threshold) for ib_layer in self.kl_list]
            return masks

    def print_compression_ratio(self, threshold, writer=None, epoch=-1):
        # applicable for structures with global pooling before fc
        _, prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
        conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            fc_shapes = [512]
        elif self.config == 'G5':
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        net_shape = [ out_channels-prune_stat[idx] for idx, out_channels in enumerate(conv_shapes+fc_shapes)]
        #conv_shape_with_pool = [v[0] if v != 'M' else 'M' for v in cfg[self.config]]
        current_n, hdim, last_channels, flops, fmap_size = 0, 64, 3, 0, 32
        for n, pruned_channels in enumerate(prune_stat):
            if n < len(conv_shapes):
                current_channels = cfg[self.config][current_n][0] - pruned_channels
                flops += (fmap_size**2) * 9 * last_channels * current_channels
                last_channels = current_channels
                current_n += 1
                if type(cfg[self.config][current_n]) is str:
                    current_n += 1
                    fmap_size /= 2
                    hdim *= 2
            else:
                current_channels = 512 - pruned_channels
                flops += last_channels * current_channels
                last_channels = current_channels
        flops += last_channels * self.n_cls

        total_params, pruned_params, remain_params = 0, 0, 0
        # total number of conv params
        in_channels, in_pruned = 3, 0
        for n, n_out in enumerate(conv_shapes):
            n_params = in_channels * n_out * 9
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_stat[n]) * 9
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out
            in_pruned = prune_stat[n]
        # fc layers
        offset = len(prune_stat) - len(fc_shapes)
        for n, n_out in enumerate(fc_shapes):
            n_params = in_channels * n_out
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_stat[n+offset])
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out
            in_pruned = prune_stat[n+offset]
        total_params += in_channels * self.n_cls
        remain_params += (in_channels - in_pruned) * self.n_cls
        pruned_params += in_pruned * self.n_cls

        print('total parameters: {}, pruned parameters: {}, remaining params:{}, remain/total params:{}, remaining flops: {}, '
              'each layer pruned: {},  remaining structure:{}'.format(total_params, pruned_params, remain_params, 
                    float(total_params-pruned_params)/total_params, flops, prune_stat, net_shape))
        if writer is not None:
            writer.add_scalar('flops', flops, epoch)
