# U-Nets

​		在深度学习的世界中流程着这么一句话**分类网络首选ResNet，检测网络选YOLO，分割网络就选U-Net**。那么今天就基于**Pytorch**来实现一下**U-Nets**系列的网络。

## U-Net的网络结构

![U-Net整体的结构](https://img-blog.csdn.net/20180826202403129?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21hbGlhbmdfMTk5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这里先套用一下U-Net的官方示意图，从图中我们可以看出U-Net主要分为两部分左侧的encode(编码)以及右侧的decode(解码)。在左侧编码部分主要是通过**降采样**操作获取图片的特征信息，然后再将这一部分的信息与右侧的部分进行整合，最终输出图片的语义信息。具体代码实现可以参考**unet_model.py**和**unet_module.py**两个模块：