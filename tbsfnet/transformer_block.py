# -----------------------------------------------------------------------------------
# 网络使用的Transformer模块
# 在ViT代码的基础上修改
# -----------------------------------------------------------------------------------
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """将图像转化为带有位置编码（可选）的图像块嵌入，卷积核大小和步长为patch_size，实现下采样"""
    def __init__(self, in_chans=256, embed_dim=256, patch_size=1, pos_embed_flag=True, img_sz=16):
        """
        Args:
            in_chans: 输入通道数
            embed_dim: 输出通道数
            patch_size: 图像块大小
            pos_embed_flag: 是否嵌入位置信息
            img_sz: 输入图像大小，用于嵌入位置信息  注意：这样网络只能处理固定大小的输入
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed_flag = pos_embed_flag
        if self.pos_embed_flag:
            self.pos_embed = nn.Parameter(torch.zeros(1, (img_sz//patch_size)**2, embed_dim))

    def forward(self, x):
        """
        Args:
            x: b * c * h * w

        output: b * n * embed_dim
        """
        # b*c*h*w -> b*d*h/p*w/p -> b*d*(h/p*w/p) -> b*(h/p*w/p)*d
        # p=1, b*c*h*w -> b*d*h*w -> b*d*(h*w) -> b*(h*w)*d
        assert x.shape[-1] == x.shape[-2]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.pos_embed_flag:
            x = x + self.pos_embed
        return x


class PatchUnEmbed(nn.Module):
    """通过卷积将块嵌入转化为图像"""
    def __init__(self, out_chans=256, embed_dim=256, patch_size=1, img_sz=16):
        """
        Args:
            out_chans: 输出通道数
            embed_dim: 输入通道数
            patch_size: 图像块大小
            img_sz: 输出图像大小
        """
        super().__init__()
        self.patch_size = patch_size
        self.img_sz = img_sz
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: b * n * embed_dim

        out: b * c * h * w
        """
        b, n, d = x.shape
        # b*(h/p*w/p)*d -> b*d*(h/p*w/p) -> b*d*h/p*w/p
        x = x.transpose(1, 2).view(b, d, self.img_sz//self.patch_size, self.img_sz//self.patch_size)
        # b*d*h/p*w/p -> b*c*h*w
        x = self.proj(x)
        return x


class MHA(nn.Module):
    """多头注意力模块"""
    def __init__(self, embed_dim=256, num_heads=8):
        """
        Args:
            embed_dim: 块嵌入长度
            num_heads: 注意力头数
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """多层感知机模块"""
    def __init__(self, in_features=256, hidden_features=None, act_layer=nn.GELU):
        """
        Args:
            in_features: 输入特征数
            hidden_features: 中间特征数
            act_layer: 激活函数
        """
        super().__init__()
        hidden_features = hidden_features if hidden_features else in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class M2Block(nn.Module):
    """多头注意力模块和多层感知机模块"""
    def __init__(self, embed_dim=256, num_heads=8, hidden_features=1024, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            embed_dim: 块嵌入长度
            num_heads: 注意力头数
            hidden_features: 中间特征数
            act_layer: MLP激活函数
            norm_layer: 层激活函数
        """
        super().__init__()
        self.mha_norm = norm_layer(embed_dim)
        self.mha = MHA(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp_norm = norm_layer(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=hidden_features, act_layer=act_layer)

    def forward(self, x):
        x = x + self.mha(self.mha_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerBlock(nn.Module):
    """网络使用的Transformer模块"""
    def __init__(self, in_chans=256, embed_dim=256, patch_size=1, pos_embed_flag=True, img_sz=16, out_chans=256,
                 m2block_n=6, num_heads=8, hidden_features=1024, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            in_chans: 输入通道数
            embed_dim: 块嵌入长度
            patch_size: 图像块大小
            pos_embed_flag: 是否嵌入位置信息
            img_sz: 输入图像大小
            out_chans: 输出通道数
            m2block_n: MHA和MLP模块数目
            num_heads: 注意力头数
            hidden_features: 中间特征数
            act_layer: MLP激活函数
            norm_layer: 层激活函数
        """
        super().__init__()
        block = []
        patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size,
                                 pos_embed_flag=pos_embed_flag, img_sz=img_sz)
        block.append(patch_embed)
        for i in range(m2block_n):
            m2block = M2Block(embed_dim=embed_dim, num_heads=num_heads, hidden_features=hidden_features,
                              act_layer=act_layer, norm_layer=norm_layer)
            block.append(m2block)
        patch_un_embed = PatchUnEmbed(out_chans=out_chans, embed_dim=embed_dim,
                                      patch_size=patch_size, img_sz=img_sz)
        block.append(patch_un_embed)
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
