**xunfei_CV_learnin**
# 第一次参加线上的有关深度学习的课程笔记与体会。

**赛题：基于脑PET图像的疾病预测**
说明： NC指代健康    MCI指代轻度认知障碍                      
数据文件格式为.nii

## 整体流程：
      1.准备正数据集与负数据集→供以学习

      2.数据图片预处理 ①去噪

                      ②增强

                      ③对其标准空间

                      ④图像标准化

      3.预处理后提取特征①逻辑回归

                       ②CNN   （1）卷积层

                              （2）池化层

                              （3）全连接层

                       ③Vision Transformer

      4.模型训练→使用CNN监督学习①材料：使用预处理后的PET图、检查

                              ②训练法：梯度下降、反向传播

                              ③输入CNN训练：调整CNN权重、偏置

      5.模型评估与优化→用训练好的模型与测试集得出结果，最后对①准确率

                                                        ②精确率

                                                        ③召回率     

                                                        ④F1值

       6.根据最后结果再对模型的参数进行调整

## 从基线方案中得出代码整体框架：
1.导入所需的库与工具

2.读取文件（测试集与训练集）并打乱顺序

3.定义对PET图像进行特征提取的函数

      ①先加载图像，再得到第一个通道的数据

      ②随机筛选10个通道来提取特征

      ③对图片计算统计值（非零像素、零像素、平均值、标准差等…）

      ④通过看是NC还是MCI，将特征值放在样品旁边

4.对训练集与测试集分别进行30次特征提取

5.用训练集的特征作为输入，训练集的类别作为输出，对逻辑回归模型训练

6.对测试集进行预测并转置，使每个样品有30次预测结果

7.将30次预测结果中次数最多的结果作为最终预测结果，存储于test_pred_label

8.生成DataFrame，其中包括了样本ID与对应预测结果

9.排序结果并保存于.csv文件中

Summary↑：第1到3步是前期的准备工作，第4到7步为预测结果，第8到9为保存文件

## 我出现的问题：
（下面是对基线代码保存到本地时，我出现的问题）

```
# 生成提交结果的DataFrame，其中包括样本ID和预测类别。
submit = pd.DataFrame(
    {
        'uuid': [int(x.split('/')[-1][:-4]) for x in test_path],  # 提取测试集文件名中的ID
        'label': test_pred_label                                  # 预测的类别
    }
)
```

上面是基线方案中截取的代码，我的本地环境运行时uuid这里出现了问题

![img](https://img-blog.csdnimg.cn/a8bde8517f694d2fbaac9d401261baa2.png)

查阅资料知道是出现了字符串中有非数字内容，导致出错

通过调整代码发现原数据输出的非数字字符为下面图像↓

![img](https://img-blog.csdnimg.cn/23565a5a38254391a2233af9c4f92774.png)

显然是“Test\”这个字符串导致问题，这个时候我们查阅split函数的用法 

 具体可以参考[Python中超好用的split()函数，详解_python split_捣鼓Python的博客-CSDN博客](https://blog.csdn.net/weixin_44793743/article/details/126572303)

或[Python学习：split()方法以及关于str.split等形式内容的详细讲解_景墨轩的博客-CSDN博客](https://blog.csdn.net/qq_41780295/article/details/88555183)

调整过后以下是我的修改方案：

```
# 生成提交结果的DataFrame，其中包括样本ID和预测类别。
submit = pd.DataFrame(
    {
        'uuid': [int(x.split('/')[-1][:-4].split("\\")[-1]) for x in test_path],  # 提取测试集文件名中的ID
        'label': test_pred_label  # 预测的类别
    }
)
```

主要是对于Python中split的学习与应用

就可以输出一个标准的格式

![img](https://img-blog.csdnimg.cn/08c6b4f391af439db93164cf2f9b85bf.png)
最后就完成本地环境的结果输出 ！

​
# 第二次课程笔记与体会

笔记：
先把结果搬上来↓

![img](https://img-blog.csdnimg.cn/4af5a180e7e642f5806c0d244a2e33f2.png)

大致分析一下CNN代码（叠一下甲，我python和深度学习的代码都不太熟悉，有错误欢迎指出）

## 个人对代码的分析：
（1）先导入要用到的库和包个人对代码的分析：
（1）先导入要用到的库和包

```
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import albumentations as A
import cv2
import torch
import nibabel as nib
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from nibabel.viewers import OrthoSlicer3D
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
```
（2）说明读取本地文件的地址，复制过来的代码一定要改成本地文件的地址哈，不然连数据集都读不进去。还有就是打乱顺序函数具体参考这位哥们解说↓
[numpy.random.shuffle打乱顺序函数](https://blog.csdn.net/qq_35091353/article/details/112797653)

```
train_path = glob.glob('E:/XUNFEIproject/脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('E:/XUNFEIproject/脑PET图像分析和疾病预测挑战赛公开数据/Test/*')
np.random.shuffle(train_path)
np.random.shuffle(test_path)
```

（3）明显这个就是处理自定义数据集，下面是我看的大佬解说参考↓
[Pytorch Dataset和Dataloader构建自定义数据集，代码模板](https://zhuanlan.zhihu.com/p/611825420)

```
class XunFeiDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index]) 
            img = img.dataobj[:,:,:, 0]
            DATA_CACHE[self.img_path[index]] = img
        
        # 随机选择一些通道            
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image = img)['image']
        
        img = img.transpose([2,0,1])
        return img,torch.from_numpy(np.array(int('NC' in self.img_path[index])))
    
    def __len__(self):
        return len(self.img_path)
```

（4）就是选择神经网络了。CNN_baseline中选用了resnet18这个模型，这样子的话你当然可以选择resnet50甚至自己来设定各个层来自定义网络。这里推荐一些资料阅读↓

[ResNet 详解](https://zhuanlan.zhihu.com/p/550360817)
[pytorch实践改造属于自己的resnet网络结构并训练二分类网络](https://zhuanlan.zhihu.com/p/62525824)

当时听水哥说用resnet18就足够了，所以我也继续用了resnet18

```
class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()

        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out
```

（5）定义训练函数train

先设置训练模式；初始化训练损失；再使用for循环遍历训练集的每一批次。

部署input和target后，设置好损失函数loss，优化器optimizer

```
def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()  # (non_blocking=True)
        target = target.cuda()  # (non_blocking=True)

        output = model(input)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:  # 设置log_interval=20，所以每隔20个batch会输出，而batch_size=2,所以每隔40个数据输出一次。
            print(loss.item())

        train_loss += loss.item()

    return train_loss / len(train_loader)
```

同理设置好验证函数↓

```
def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target.long())

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)
```

（6）对训练集，验证集，测试集的设置（数据增强等）

我用到了后面的一些内容，交叉验证，更多数据增强以及更多循环训练次数设置了69次，所以代码与原始CNN会有些许不同。

```
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

skf = KFold(n_splits=10, random_state=233, shuffle=True)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_path, train_path)):

    train_loader = torch.utils.data.DataLoader(
        XunFeiDataset(train_path[:-10],
                      A.Compose([
                          A.RandomCrop(120, 120),
                          A.RandomRotate90(p=0.5),
                          A.HorizontalFlip(p=0.5),
                          A.RandomContrast(p=0.5),
                          A.RandomBrightnessContrast(p=0.5),
                          A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1,
                                             border_mode=4, value=None, mask_value=None, always_apply=False, p=0.4),
                          A.GridDistortion(num_steps=10, distort_limit=0.3, border_mode=4, always_apply=False, p=0.4),

                      ])
                      ), batch_size=8, shuffle=True#, num_workers=0, pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        XunFeiDataset(train_path[-10:],
                      A.Compose([
                          A.RandomCrop(120, 120),
                      ])
                      ), batch_size=8, shuffle=False#, num_workers=0, pin_memory=False
    )

    model = XunFeiNet()
    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 0.001)

    for _ in range(69):
        train_loss = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)
        train_acc = validate(train_loader, model, criterion)

        print(train_loss, train_acc, val_acc)
        torch.save(model.state_dict(), 'E:/XUNFEIproject/resnet18_fold{0}.pt'.format(fold_idx))

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
                  A.Compose([
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=8, shuffle=False#, num_workers=0, pin_memory=False
)
```

（7）最后就是综合预测与打印结果了
```
def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


pred = None
model_path = glob.glob(r'E:\XUNFEIproject')

for model_path in ['resnet18_fold0.pt', 'resnet18_fold1.pt', 'resnet18_fold2.pt',
                   'resnet18_fold3.pt', 'resnet18_fold4.pt', 'resnet18_fold5.pt',
                   'resnet18_fold6.pt', 'resnet18_fold7.pt', 'resnet18_fold8.pt',
                   'resnet18_fold9.pt']:

    model = XunFeiNet()
    model = model.to('cuda')
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 0.001)

    for _ in range(10):
        if pred is None:
            pred = predict(test_loader, model, criterion)
        else:
            pred += predict(test_loader, model, criterion)

submit = pd.DataFrame(
    {
        'uuid': [int(x.split('/')[-1][:-4].split("\\")[-1]) for x in test_path],
        'label': pred.argmax(1)
    })
submit['label'] = submit['label'].map({1: 'NC', 0: 'MCI'})
submit = submit.sort_values(by='uuid')
submit.to_csv('submit_cnn_Kflod5.csv', index=None)
```

**OVER！**

## 下面我将讲解我本地环境部署时遇到的问题与解决方法。

问题一：
![img](https://img-blog.csdnimg.cn/6329e9cdd8104d70adcde48ddaca3796.png)
找不到cudnn，查阅了大量博主的回答得知有可能是pytorch，torchvison，cuda，cudnn，python里面有版本的匹配错误，必须要互相适配版本！（但是警告是黄色的，不理会也没问题狗头护体

问题二：
![img](https://img-blog.csdnimg.cn/f82d139f27a9411c9c1da0e07b883daf.png)
这个意思是以后版本的公式可能会改动，所以这条可以忽视，不影响运行

问题三：
![img](https://img-blog.csdnimg.cn/be2c82bd1e3c42bdab17800cf70a34f4.png)
查阅大量资料可参考这篇文章↓
[“nll_loss_forward_reduce_cuda_kernel_2d_index“ not implemented for ‘Int‘ 问题解决](https://blog.csdn.net/weixin_45218778/article/details/129046690)

即在代码中所有的↓
```
loss = criterion(output, target)
```
改为↓
```
loss = criterion(output, target.long())
```
即转换格式，问题就解决了

问题四：
![img](https://img-blog.csdnimg.cn/b3324a99b863403f9b24c98ea9b05583.png)
出现了这个问题,在查阅了大量文章，终于找到了解决这个问题的蛛丝马迹，具体可参考↓
[Pytorch中Dataloader踩坑：RuntimeError: DataLoader worker](https://blog.csdn.net/qq_38662733/article/details/108549461)

即将DataLoader中的num_workers=2,改成num_workers=0,仅执行主进程。
在我的CNN中就是将下面代码↓

```
train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
                  A.Compose([
                      A.RandomRotate90(),
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=True, num_workers=2, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-10:],
                  A.Compose([
                      A.RandomCrop(120, 120),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=2, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
                  A.Compose([
                      A.RandomCrop(128, 128),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=2, pin_memory=False
)
```
改为↓
```
train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
                  A.Compose([
                      A.RandomRotate90(),
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=True, num_workers=0, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-10:],
                  A.Compose([
                      A.RandomCrop(120, 120),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
                  A.Compose([
                      A.RandomCrop(128, 128),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=0, pin_memory=False
)
```
**至此问题全部解决！！！！！！！！！！**
**运行！输出.csv文件！！！！！！！！！！！！**

![img](https://img-blog.csdnimg.cn/cb04c99f039643af80c2fc9150deb431.png)

最后附上我目前的CNN训练代码，还有待改进!

```
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import albumentations as A
import cv2
import torch
import nibabel as nib
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from nibabel.viewers import OrthoSlicer3D
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

train_path = glob.glob('E:/XUNFEIproject/脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('E:/XUNFEIproject/脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

np.random.shuffle(train_path)
np.random.shuffle(test_path)
DATA_CACHE = {}
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


class XunFeiDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        if self.img_path[index] in DATA_CACHE:
            img = DATA_CACHE[self.img_path[index]]
        else:
            img = nib.load(self.img_path[index])
            img = img.dataobj[:, :, :, 0]
            DATA_CACHE[self.img_path[index]] = img

        # 随机选择一些通道
        idx = np.random.choice(range(img.shape[-1]), 50)
        img = img[:, :, idx]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(int('NC' in self.img_path[index])))

    def __len__(self):
        return len(self.img_path)


class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()

        model = models.resnet18(True)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 2)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()  # (non_blocking=True)
        target = target.cuda()  # (non_blocking=True)

        output = model(input)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:  # 设置log_interval=20，所以每隔20个batch会输出，而batch_size=2,所以每隔40个数据输出一次。
            print(loss.item())

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(val_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target.long())

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)


from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

skf = KFold(n_splits=10, random_state=233, shuffle=True)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_path, train_path)):

    train_loader = torch.utils.data.DataLoader(
        XunFeiDataset(train_path[:-10],
                      A.Compose([
                          A.RandomCrop(120, 120),
                          A.RandomRotate90(p=0.5),
                          A.HorizontalFlip(p=0.5),
                          A.RandomContrast(p=0.5),
                          A.RandomBrightnessContrast(p=0.5),
                          A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1,
                                             border_mode=4, value=None, mask_value=None, always_apply=False, p=0.4),
                          A.GridDistortion(num_steps=10, distort_limit=0.3, border_mode=4, always_apply=False, p=0.4),

                      ])
                      ), batch_size=8, shuffle=True#, num_workers=0, pin_memory=False
    )

    val_loader = torch.utils.data.DataLoader(
        XunFeiDataset(train_path[-10:],
                      A.Compose([
                          A.RandomCrop(120, 120),
                      ])
                      ), batch_size=8, shuffle=False#, num_workers=0, pin_memory=False
    )

    model = XunFeiNet()
    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 0.001)

    for _ in range(69):
        train_loss = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)
        train_acc = validate(train_loader, model, criterion)

        print(train_loss, train_acc, val_acc)
        torch.save(model.state_dict(), 'E:/XUNFEIproject/resnet18_fold{0}.pt'.format(fold_idx))

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
                  A.Compose([
                      A.RandomCrop(120, 120),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=8, shuffle=False#, num_workers=0, pin_memory=False
)


def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


pred = None
model_path = glob.glob(r'E:\XUNFEIproject')

for model_path in ['resnet18_fold0.pt', 'resnet18_fold1.pt', 'resnet18_fold2.pt',
                   'resnet18_fold3.pt', 'resnet18_fold4.pt', 'resnet18_fold5.pt',
                   'resnet18_fold6.pt', 'resnet18_fold7.pt', 'resnet18_fold8.pt',
                   'resnet18_fold9.pt']:

    model = XunFeiNet()
    model = model.to('cuda')
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 0.001)

    for _ in range(10):
        if pred is None:
            pred = predict(test_loader, model, criterion)
        else:
            pred += predict(test_loader, model, criterion)

submit = pd.DataFrame(
    {
        'uuid': [int(x.split('/')[-1][:-4].split("\\")[-1]) for x in test_path],
        'label': pred.argmax(1)
    })
submit['label'] = submit['label'].map({1: 'NC', 0: 'MCI'})
submit = submit.sort_values(by='uuid')
submit.to_csv('submit_cnn_Kflod5.csv', index=None)

```












