# VIBNet
This repository provides codes to reproduce the major experiments in the paper [Compressing Neural Networks using the Variational Information Bottleneck](https://arxiv.org/abs/1802.10399), (ICML 2018).

<div  align="center">
<img src="https://user-images.githubusercontent.com/18202259/40783710-e1b72da2-6515-11e8-9121-11d699e4d909.png" width = "60%" />
</div>

If you find this code useful in your research you could cite
```
@inproceedings{dai2018vib,
    title={Compressing Neural Networks using the Variational Information Bottleneck},
    author={Dai, Bin and Zhu, Chen and Guo, Baining and Wipf, David},
    booktitle={Proceedings of the 35th International Conference on Machine Learning (ICML 2018)},
    year={2018}
}
```

# Update on Oct 18, 2019
Fixed some compatibility issue with PyTorch 1.2.0. 

# Prerequisites
The experiments can be reproduced with PyTorch 0.3.1 and CUDA 9.0 on Ubuntu 16.04. Here is one approach to set up:

1. Install Anaconda2 (optional).
```
mkdir tmp && cd tmp
wget https://repo.anaconda.com/archive/Anaconda2-5.1.0-Linux-x86_64.sh && bash Anaconda2-5.1.0-Linux-x86_64.sh
```

2. Install PyTorch.
```
# Select the correct version according to your environment. 
pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision 
```

3. Install TensorboardX for visualization.
```
pip install tensorboardx
```

4. Download the pretrained VGG models and unzip under this directory via 
[Google Drive](https://drive.google.com/file/d/1W97IV97KtM_XBm9dGTsxdjYlCLSJbRZr/view?usp=sharing) or 
[Baidu Pan](https://pan.baidu.com/s/1YXNDcdz0mvvZhYekRPKXLA).

# Run
Tabel 3 part 1
```
python ib_vgg_train.py --gpu 0 --batch-norm --resume-vgg-pt baseline/cifar10/checkpoint_299_nocrop.tar --ban-crop --opt adam --cfg D4 --epochs 300 --lr 1.4e-3 --weight-decay 5e-5 --kl-fac 1.4e-5 --save-dir ib_vgg_chk/D4
```

Table 3 part 2
```
python ib_vgg_train.py --data-set cifar10 --gpu 0 --batch-norm --resume-vgg-vib baseline/cifar10/D6_600/last_epoch.pth --opt adam --cfg D6 --epochs 300 --lr 1e-3 --weight-decay 5e-5 --kl-fac 3e-5 --save-dir ib_vgg_chk/D6_600
```

Table 3 part 3
```
python ib_vgg_train.py --data-set cifar10 --gpu 0 --batch-norm --resume-vgg-vib baseline/cifar10/G5_400/last_epoch.pth --opt adam --cfg G5 --epochs 300 --lr 1e-3 --weight-decay 5E-5 --kl-fac 1e-5 --save-dir ib_vgg_chk/G5-400
```

Table 4 part 1
```
python ib_vgg_train.py --data-set cifar100 --gpu 0 --batch-norm --resume-vgg-vib baseline/cifar100/vgg-cifar100-pretrain.pth --opt adam --cfg G --epochs 300 --lr 1e-3 --weight-decay 1.2e-4 --kl-fac 1e-5 --ban-crop --ban-flip --save-dir ib_vgg_chk/cifar100-G
```

Table 4 part 2
```
python ib_vgg_train.py --data-set cifar100 --gpu 0 --batch-norm --resume-vgg-vib baseline/cifar100/crop_300/last_epoch.pth --opt adam --cfg G5 --epochs 300 --lr 1e-3 --weight-decay 5E-5 --kl-fac 1.5e-5 --save-dir ib_vgg_chk/cifar100-crop-300
```
