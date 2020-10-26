## 百度飞桨图像分割7日打卡营总结心得

经过几天的学习认识了什么是图像分割，图像分割的种类，已经几种图像分割深度学习的方法。

但学到的不仅如此。

首先是对于python的学习。我是python小白，一点一点学习的。这次课程也让我对OOP编程有了一定的了解。继承，以及一些__len__方法，__call__方法。

其次就是对opencv以及pillow的一些学习。在做visualize的时候，图像显现出来真的很有成就感。

然后学习了一个使用paddlepaddle动态图搭建一个项目的架构。
从dataloader（包含transfrom），然后是train，val，infer。朱老师带着一点一点的学习，感觉让我进步很大。

最后就是课程的主体、几种算法。
下面简单说一下我的理解。FCN是利用分类网络VGG的结构，将后面的全连接层改为卷积层，做得一个全卷积网络。为了保证输出，模型的输入需要加上padding的操作。

然后是UNET，这个模型就真的是U型的网络。比较简单，encoder的feature map 直接concat到decoder对应的feature map上。

接着是PSP。主要的部分是pyramid pool的结构。多个使用adaptive pool的分支。然后通过interpolate达到相同尺寸。这是一种增大感受野的方法。paddlepaddle提供了adaptive pool的API。

最后是deeplab系列。老师主要讲了两个方面。一个是空洞卷积，一个是ASPP。
空洞卷积很好理解。就是在filter中加上一些0的参数。扩大了filter的尺寸，能看到更多的地方，ASPP和pyramid pool有一点像，不过它是在不同的分支上使用不同的空洞卷积。

后面的图网络太难了，而且没有作业，没有听得太懂。看来有时候写代码真的能帮助理解。
