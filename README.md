# U-Nets

​		在深度学习的世界中流程着这么一句话**分类网络首选ResNet，检测网络选YOLO，分割网络就选U-Net**。那么今天就基于**Pytorch**来实现一下**U-Nets**系列的网络。

## U-Net的网络结构

![U-Net整体的结构](https://img-blog.csdn.net/20180826202403129?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21hbGlhbmdfMTk5Mw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

这里先套用一下U-Net的官方示意图，从图中我们可以看出U-Net主要分为两部分左侧的encode(编码)以及右侧的decode(解码)。在左侧编码部分主要是通过**降采样**操作获取图片的特征信息，然后再将这一部分的信息与右侧的部分进行整合，最终输出图片的语义信息。具体代码实现可以参考**unet_model.py**和**unet_module.py**两个基本模块：

### 基本模块unet_module

U-Net的基本模块结构可以分为**双卷积层**、**Down层**和**UP**层。

#### 双卷积层

​	双卷积的基本结构为(conv+BN+Relu)**2，具体实现如下：

```python
class DoubleConv(nn.Module):
  	def __init__(self,in_channels,out_channels):
      super(DoubleConv,self).__init__()
      self.double_conv=nn.Sequential(
      	nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyRelu(0.1),#或者nn.Relu(inplace=True)
        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_chanels),
      	nn.LeakyRelu(0.1),#或者nn.Relu(inplace=True)
      )
    def forward(self,x):
      return self.double_conv(x)
```

#### 2.下采样Down层

下采样就是U-Net左边的网络部分，它的主要构成为**最大池化**和**双卷积层**。具体实现如下：

```python
class Down(nn.Module):
  	def __init__(self,in_channels,out_channels):
      super(Down,self).__init__()
      self.maxpool_conv=nn.Sequential(
      	nn.MaxPool2d(2),
        DoubleConv(in_channels,out_channels)
      )
    def forward(self,x):
      return self.maxpool_conv(x)
```

#### 3 上采样Up层

上采样层为U-net的网络的右边部分，它的主要构成为**反卷积(ConvTranspose2d)**、**BN**和**Relu**层。有关反卷积的知识可以参考下面的知识[反卷积知识](https://www.zhihu.com/question/48279880)，具体实现如下：

```python
class Up(nn.Module):
  def __init__(self,in_channels,out_channels):
    	super(Up,self).__init__()
      #反卷积
      self.up=nn.Sequential(
      	nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,strid=2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
     	)
      self.conv=DoubleConv(in_channels,out_channels)
  def forward(self,x1,x2):
    	x1=self.up(x1)
    	diffY=x2.size()[2]-x1.size()[2]
      diffX=x2.size()[3]-x1.size()[3]
      x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
      x=torch.cat([x2,x1],dim=1)
      return self.conv(x)
```

### 具体实现过程

U-Net的实现过程可以分为左半部分的**4个Down**模块和右侧的**4个Up**模块，具体的实现如下：

```python
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.features = 16  # 定义特征基数
        self.n_channels = 3
        self.n_classes = 1
        self.inc = DoubleConv(self.n_channels, self.features)
        self.down1 = Down(self.features, self.features * 2)
        self.down2 = Down(self.features * 2, self.features * 4)
        self.down3 = Down(self.features * 4, self.features * 8)
        self.down4 = Down(self.features * 8, self.features * 16)

        self.up1 = Up(self.features * 16, self.features * 8)
        self.up2 = Up(self.features * 8, self.features * 4)
        self.up3 = Up(self.features * 4, self.features * 2)
        self.up4 = Up(self.features * 2, self.features)
        self.outc = OutConv(self.features, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

按照上面的结构就构建成了**U-Net**的模型结构，用Pytorch实现起来非常的简单。

### 语义分割使用的损失函数

##### 1.二分类常见的损失函数

二分类常见的损失函数为**Dice Loss**,这种损失函数主要应对的是样本不均衡的数据。比如我前段时间做的基于U-Net的车牌识别模型，采用Dice Loss效果就很好。Dice loss首先定义两个轮廓区域的相似程度，然后再计算轮廓区域的点积。

##### 2. [Lovasz-Softmax loss](http://cn.arxiv.org/pdf/1705.08790v2)

这个是目前使用的最好的语义分割的损失函数，这个损失函数是开源的。可以在上述的链接中查看该损失函数的具体使用方法，该损失函数也包括了单分类和多分类的不同调用。经过本人的实测使用该损失函数的效果也较**Dice Loss**效果好很多。





