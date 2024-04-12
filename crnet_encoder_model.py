from config import *
from backbone import BACKBONE
from transformer_detr import *
from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# 全卷积
class FCNBranch(nn.Module):
    def __init__(self):
        super(FCNBranch, self).__init__()
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
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


class CRNETENCODER(nn.Module):
    def __init__(self):
        super(CRNETENCODER,self).__init__()
        self.crnet = BACKBONE()
        self.conv = nn.Conv2d(1024,512,kernel_size=1)
        self.maxpool = nn.MaxPool2d(4)
        self.positionembedding = PositionEmbeddingSine()
        self.encoder = Transformer_DETR(d_model=512,nhead=8,num_encoder_layers=6,
                                        dim_feedforward=2048,dropout=0.1,
                                        activation='relu',normalize_before=False)
        self.tftc = FCNBranch()
        self.xyoffset = FCNBranch()
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 放大4倍

    def forward(self,img):

        m, tf, tc, xoffset, yoffset = self.crnet(img)
        m = self.maxpool(self.conv(m))  # B,512,32,32

        mask = torch.zeros((1, 32, 32), dtype=torch.bool, device=m.device)

        s = NestedTensor(m, mask)
        pos = self.positionembedding(s)  # B,256,32,32

        memory = self.encoder(m,mask=None,pos_embed=pos)  # B,512,32*32

        memory = memory.view(-1,512,32,32)  # B,512,32,32

        memory = self.upsample4(memory)  # B,512,128,128

        # 得到Tf,Tc,Xoffset,Yoffset
        tf_tc = self.tftc(memory)
        xoffset_yoffset = self.xyoffset(memory)

        # 拆分Tf, Tc, Xoffset, Yoffset
        Tf, Tc = tf_tc.split(1, dim=1)  # 按通道维度拆分
        Xoffset, Yoffset = xoffset_yoffset.split(1, dim=1)  # 按通道维度拆分

        out = {'tf_all': Tf, 'tc_all': Tc, 'xf_all': Xoffset, 'yf_all': Yoffset,
               'tf_crnet': tf, 'tc_crnet': tc,'xf_crnet': xoffset, 'yf_crnet': yoffset}

        return out


