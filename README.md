# Autoslim

A pytorch toolkit for structured neural network pruning automatically


## Installation 安装

```bash
pip install -e ./Autoslim
```

## Quickstart 使用介绍

### 1 Automatic Pruning 自动化剪枝

```python
# 例子1:完全自动化剪枝

import torch_pruning as pruning
from torchvision.models import resnet18
import torch

#模型建立
model=resnet18()

#剪枝引擎建立
slim=pruning.Autoslim(model,inputs=torch.randn(1,3,224,224),compression_ratio=0.5)

#剪枝，系统默认prune_shortcut=1,prune_shortcut=0时不剪跳连层
slim.l1_norm_pruning(prune_shortcut=1)
```

### 2 Custom Pruning 自定义剪枝

```python
# 例子2:自定义剪枝

import torch_pruning as pruning
from torchvision.models import resnet18
import torch

#用户输入：模型，模型输入数据，自定义压缩率

#模型建立
model=resnet18()

#剪枝引擎建立
slim=pruning.Autoslim(model,inputs=torch.randn(1,3,224,224),compression_ratio=0.5)

#查看每个层的编号
for key,value in slim.index_of_layer().items():
  print(key,value)

#按照{层编号：层压缩率}这样的字典格式，指定自定义层的压缩率。其中1为层编号，0.6为层压缩率
layer_compression_rate={1:0.6}

#剪枝
slim.l1_norm_pruning(layer_compression_ratio=layer_compression_rate)
```
