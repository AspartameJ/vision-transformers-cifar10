# vision-transformers-cifar10在Ascend910上训练
环境要求：
Ascend910b+x86环境+Ascend910b驱动固件

docker镜像：
[ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:22.0.RC3-1.8.1](https://ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:22.0.RC3-1.8.1#/detail/pytorch-modelzoo)

代码下载：
`cd /path/to/transformers-cifar10`
`git clone https://github.com/AspartameJ/vision-transformers-cifar10.git`

数据集下载：
`cd vision-transformers-cifar10`
`mkdir data && cd data`
`wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && tar xvf cifar-10-python.tar.gz && rm -rf cifar-10-python.tar.gz`

容器启动命令参考：
`docker run -it -u root --name torch-0123 --ipc=host --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /etc/ascend_install.info:/etc/ascend_install.info -v /path/to/transformers-cifar10:/root/transformers-cifar10 ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:22.0.RC3-1.8.1 /bin/bash`

安装python3依赖：
`pip install torchvision==0.9.1 einops odach`

启动训练：
单卡训练：`bash train_cifar10_npu_1p.sh`或者`python3 train_cifar10_npu.py --net swin --n_epochs 400`
2卡训练：`bash train_cifar10_npu_distribute_2p.sh`
4卡训练：`bash train_cifar10_npu_distribute_4p.sh`

查看训练日志：
`tail -f output/0/train_0.log'
查看npu资源使用情况
`watch npu-smi info`

# vision-transformers-cifar10
Let's train vision transformers for cifar 10! 

This is an unofficial and elementary implementation of `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`.

I use pytorch for implementation.

### Updates
* Added [ConvMixer]((https://openreview.net/forum?id=TVHS5Y4dNvM)) implementation. Really simple! (2021/10)

* Added wandb train log to reproduce results. (2022/3)

* Added CaiT and ViT-small. (2022/3)

* Added SwinTransformers. (2022/3)

* Added MLP mixer. (2022/6)

* Changed default training settings for ViT.

# Usage example
`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

# Results..

|             | Accuracy | Train Log |
|:-----------:|:--------:|:--------:|
| ViT patch=2 |    80%    | |
| ViT patch=4 Epoch@200 |    80%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=4 Epoch@500 |    88%   | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
| ViT patch=8 |    30%   | |
| ViT small  | 80% | |
| MLP mixer |    88%   | |
| CaiT  | 80% | |
| Swin-t  | 90% | |
| ViT small (timm transfer) | 97.5% | |
| ViT base (timm transfer) | 98.5% | |
| [ConvMixerTiny(no pretrain)](https://openreview.net/forum?id=TVHS5Y4dNvM) | 96.3% |[Log](https://wandb.ai/arutema47/cifar10-challange/reports/convmixer--VmlldzoyMjEyOTk1?accessToken=2w9nox10so11ixf7t0imdhxq1rf1ftgzyax4r9h896iekm2byfifz3b7hkv3klrt)|
|   resnet18  |  93%  | |
|   resnet18+randaug  |  95%  | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTYz?accessToken=968duvoqt6xq7ep75ob0yppkzbxd0q03gxy2apytryv04a84xvj8ysdfvdaakij2) |

# Used in..
* Vision Transformer Pruning [arxiv](https://arxiv.org/abs/2104.08500) [github](https://github.com/Cydia2018/ViT-cifar10-pruning)
