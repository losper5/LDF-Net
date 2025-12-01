import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

# ——— MFM 融合模块 ——— #
class MFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        x = torch.cat(in_feats, dim=1)               # [B, height*C, H, W]
        x = x.view(B, self.height, C, H, W)          # [B, h, C, H, W]
        feats_sum = x.sum(dim=1)                     # [B, C, H, W]
        attn = self.mlp(self.avg_pool(feats_sum))    # [B, h*C, 1, 1]
        attn = attn.view(B, self.height, C, 1, 1)    # [B, h, C,1,1]
        attn = self.softmax(attn)                    # along height
        out = (x * attn).sum(dim=1)                  # [B, C, H, W]
        return out

# ——— 基础组件 ——— #
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.eps   = eps
    def forward(self, x):
        mu  = x.mean([1,2,3], keepdim=True)
        var = x.var([1,2,3], keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gamma + self.beta

# ——— DBlock ——— #
class Branch(nn.Module):
    def __init__(self, c, DW_Expand=1, dilation=1):
        super().__init__()
        dw = DW_Expand * c
        self.branch = nn.Conv2d(dw, dw, 3,
                                padding=dilation,
                                groups=dw,
                                dilation=dilation,
                                bias=True)
    def forward(self, x):
        return self.branch(x)

class DBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2,
                 dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.conv1      = nn.Conv2d(c, self.dw_channel, 1, bias=True)
        self.extra_conv = ( nn.Conv2d(self.dw_channel, self.dw_channel,
                                      3, padding=1, groups=c, bias=True)
                           if extra_depth_wise else nn.Identity() )
        self.branches   = nn.ModuleList([
            Branch(self.dw_channel, DW_Expand=1, dilation=d)
            for d in dilations
        ])
        self.sca   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel//2,
                      self.dw_channel//2, 1, bias=True)
        )
        self.sg1   = SimpleGate()
        self.sg2   = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel//2, c, 1, bias=True)
        ffn_ch     = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_ch, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch//2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, c, 1, 1))
    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = sum(b(x) for b in self.branches)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + self.gamma * x

# ——— CBAM ——— #
class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=True)
        )
    def forward(self, x):
        B, C, _, _ = x.size()
        avg = F.adaptive_avg_pool2d(x,1).view(B,C)
        mx  = F.adaptive_max_pool2d(x,1).view(B,C)
        w   = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(B,C,1,1)
        return x * w

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self, x):
        avg = x.mean(dim=1,keepdim=True)
        mx,_= x.max(dim=1,keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg,mx],dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cg = ChannelGate(channels, reduction)
        self.sg = SpatialGate()
    def forward(self, x):
        return self.sg(self.cg(x))

class UWnet(nn.Module):
    def __init__(self, num_layers=3, base_ch=64):
        super().__init__()
        # 双路输入
        self.input_l   = nn.Conv2d(1,  base_ch, 3, padding=1, bias=False)
        self.input_ab  = nn.Conv2d(2,  base_ch, 3, padding=1, bias=False)
        self.relu      = nn.ReLU(inplace=True)
        self.pool      = nn.MaxPool2d(2)
        self.up        = nn.Upsample(scale_factor=2,
                                     mode='bilinear', align_corners=False)
        self.reduce    = nn.Conv2d(base_ch*2, base_ch, 1, bias=False)

        # 中层：各自 DBlock + CBAM
        self.blocks_l   = nn.Sequential(*[
            DBlock(base_ch, DW_Expand=2, FFN_Expand=2, dilations=[1])
            for _ in range(num_layers)
        ])
        self.blocks_ab  = nn.Sequential(*[
            DBlock(base_ch, DW_Expand=2, FFN_Expand=2, dilations=[1])
            for _ in range(num_layers)
        ])
        self.cbam_l    = CBAM(base_ch, reduction=16)
        self.cbam_ab   = CBAM(base_ch, reduction=16)

        # 浅层 MFM 
        self.shallow_mfm = MFM(base_ch)


        # 输出
        self.output    = nn.Conv2d(base_ch, 3, 3, padding=1, bias=False)

    def forward(self, x_lab):
        L  = x_lab[:, :1, ...]   
        AB = x_lab[:, 1:, ...]  

        # ——— 浅层特征提取 ———
        f_l   = self.relu(self.input_l(L))
        p_l   = self.pool(f_l); u_l = self.up(p_l)
        cat_l = torch.cat([u_l, f_l], dim=1)
        sl    = self.relu(self.reduce(cat_l))       

        f_ab   = self.relu(self.input_ab(AB))
        p_ab   = self.pool(f_ab); u_ab = self.up(p_ab)
        cat_ab = torch.cat([u_ab, f_ab], dim=1)
        sa     = self.relu(self.reduce(cat_ab))        

        # —— 浅层融合 —— MFM ——— 
        shallow_feat = self.shallow_mfm([sl, sa])    

        # ——— 中层 L 分支 ———
        b_l    = self.blocks_l(shallow_feat)
        attn_l = self.cbam_l(b_l)

        # ——— 中层 AB 分支 ———
        b_ab    = self.blocks_ab(shallow_feat)
        attn_ab = self.cbam_ab(b_ab)

        # ——— 融合 ———
        fused = attn_l + attn_ab                       

        # ——— 输出 ——— 
        return self.output(fused)