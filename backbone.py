"""
输入（batchisize，3，512，512），变换在dataset中实现
输出 m （B，256*4，512，512） 256*4是因为输入mce的是fpn生成的四个特征图再次组合后，四个特征图上采样拼接起来的
tf，tc，xoffset，yoffset是（B，1，512，512），groundtruth中对应为原图大小，在dataset时变换
"""


from config import *
import torchvision.models as models


from config import *
import torchvision.models as models


# LFE模块
class LFE(nn.Module):
    def __init__(self):
        super(LFE, self).__init__()

        self.conv1 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding=(1,0))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0,1))
        self.bn2 = nn.BatchNorm2d(256)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self,k,q):
        k = self.conv1(k)
        k = self.bn1(k)
        k = self.relu(k)
        k = self.conv2(k)
        k = self.bn2(k)
        k = self.relu(k)
        k = self.maxpool(k)
        k = k+q
        k = self.conv3(k)
        k = self.bn3(k)
        k = self.relu(k)
        k = self.conv4(k)
        k = self.bn4(k)
        k = self.relu(k)

        return k


# MCE模块
class MCE(nn.Module):
    def __init__(self):
        super(MCE, self).__init__()

        # 生成F1
        self.maxpool = nn.AdaptiveMaxPool2d((1, 160))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 160))
        self.getf1 = nn.Sequential(
            nn.Conv2d(1024 * 2, 1, kernel_size=(1, 1)),
            nn.Conv2d(1, 1, kernel_size=(1, 160)),
            nn.Conv2d(1, 1, kernel_size=(160, 1)),
            nn.Sigmoid()
        )

        # 生成F2
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(1024, 1024, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(1024)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(1024)
        )

        # 生成F3
        self.conv3x3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)

        self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x1 = self.maxpool(x).expand(-1, -1, 160, 160)  # 展开维度以匹配其他特征图
        x2 = self.avgpool(x).expand(-1, -1, 160, 160)
        f0 = torch.cat((x1, x2), dim=1)
        f1 = self.getf1(f0)
        f1 = self.relu(f1)

        x3 = self.conv7x7(x)
        x4 = self.conv5x5(x)
        f2 = x3 + x4
        f2 = self.relu(f2)

        f3 = self.conv3x3(x)
        f3 = self.bn1(f3)
        f3 = self.relu(f3)
        f4 = f2 + f3
        f5 = f4 * f1
        f6 = f5 + f4
        f6 = self.bn2(f6)
        f6 = self.relu(f6)

        return f6


# 全卷积网络
class FCNBranch(nn.Module):
    def __init__(self):
        super(FCNBranch, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,2, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.conv3(x)
        return x


class BACKBONE(nn.Module):  # 输入尺寸为(batch_size,3,640,640)
    def __init__(self):
        super(BACKBONE, self).__init__()

        # 加载预训练的ResNet50模型
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V2')
        # for param in self.resnet50.parameters():
        #     param.requires_grad = False

        # 获取ResNet50的前五个子模块，包括初始卷积层、池化层和第一个残差块
        self.layer1 = nn.Sequential(*list(self.resnet50.children())[:5])
        # 获取ResNet50的第二个残差块
        self.layer2 = self.resnet50.layer2
        # 获取ResNet50的第三个残差块
        self.layer3 = self.resnet50.layer3
        # 获取ResNet50的第四个残差块
        self.layer4 = self.resnet50.layer4

        # 定义1x1卷积层，用于降低特征图的通道数
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )   # c2->p2
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )  # c3->p3
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )  # c4->p4
        self.conv5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )  # c5->p5

        # 定义上采样层，用于将特征图的尺寸放大
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 放大2倍
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 放大4倍
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 放大8倍

        # 定义的3x3卷积层，用于进一步处理特征图
        self.conv_q2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU()  # 通常在BatchNorm后添加激活函数
        )
        self.conv_q3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU()  # 通常在BatchNorm后添加激活函数
        )
        self.conv_q4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU()  # 通常在BatchNorm后添加激活函数
        )
        self.conv_q5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU()  # 通常在BatchNorm后添加激活函数
        )

        # 定义生成k2-k5的LFE
        self.conv_k2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.lfe_k3 = LFE()
        self.lfe_k4 = LFE()
        self.lfe_k5 = LFE()

        # MCE
        self.mce = MCE()

        # Branch for predicting text regions (Tf and Tc)
        self.text_branch = FCNBranch()
        # Branch for predicting offsets (Xoffset and Yoffset)
        self.offset_branch = FCNBranch()

    def forward(self,x):
        # 通过ResNet50的各个层提取特征c1-c4
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 通过1x1卷积层降低特征图的通道数
        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)

        # 特征融合并卷积得到q2-q5
        q2 = self.conv_q2(p2+self.upsample2(p3))
        q3 = self.conv_q3(p3+self.upsample2(p4))
        q4 = self.conv_q4(p4+self.upsample2(p5))
        q5 = self.conv_q5(p5)

        # 生成k2-k5
        k2 = self.conv_k2(q2)  # (bsize,256,160,160)
        k3 = self.lfe_k3(k2,q3)  # (bsize,256,80,80)
        k4 = self.lfe_k4(k3,q4)  # (bsize,256,40,40)
        k5 = self.lfe_k5(k4,q5)  # (bsize,256,20,20)

        # 将k2-k5上采样至原图1/4（即160x160），并组合为c
        k_3 = self.upsample2(k3)
        k_4 = self.upsample4(k4)
        k_5 = self.upsample8(k5)
        c = torch.cat((k2,k_3,k_4,k_5),dim=1)  # b,1024,128,128

        # 将c传入MCE，得到m
        m = self.mce(c)  # B,1024,128,128

        # 得到Tf,Tc,Xoffset,Yoffset
        tf_tc = self.text_branch(m)
        xoffset_yoffset = self.offset_branch(m)

        # 拆分Tf, Tc, Xoffset, Yoffset
        Tf, Tc = tf_tc.split(1, dim=1)  # 按通道维度拆分
        Xoffset, Yoffset = xoffset_yoffset.split(1, dim=1)  # 按通道维度拆分

        return m, Tf, Tc, Xoffset, Yoffset

