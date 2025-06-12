"""
大多数网络只需要继承BaseNet，并定义set_blocks和forward
还可以修改set_optimizer, set_scheduler, init_weight
__init__需要修改name和block，并调用init_net
根据模型的损失函数定义Loss
"""
import torch
from torch import nn
from torch.nn import utils
from torch import optim
from torch.optim import lr_scheduler
from utilizations import function_for_weight_decay, get_block_list_generator
from transformer_block import TransformerBlock


class WeightedMaskL1Loss(nn.Module):
    """带权重的Mask-L1损失"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, x, y, wt, seg=None):
        if seg is None:
            seg = wt
        loss = torch.abs(x-y)*wt
        loss = loss.sum()/(seg.sum()+self.eps)
        return loss


class WeightedMaskMSELoss(nn.Module):
    """带权重的Mask-MSE损失"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, x, y, wt, seg=None):
        if seg is None:
            seg = wt
        loss = (x-y)**2*wt
        loss = loss.sum()/(seg.sum()+self.eps)
        return loss


class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, x, y):
        loss = 1-(2*x*y).sum()/((x+y).sum()+self.eps)
        return loss


MinutiaeLoss = WeightedMaskMSELoss                      # 细节点损失
ReconstructionLoss = WeightedMaskL1Loss                 # 重建损失
OrientationLoss = WeightedMaskMSELoss                   # 方向损失
FrequencyLoss = WeightedMaskMSELoss                     # 频率损失
SegmentationLoss = DiceLoss                             # 分割损失


class PlainResidualBlock(nn.Module):
    """ResNet的残差模块（数据大小不发生变化）"""
    def __init__(self, ch_num):
        super().__init__()
        self.ReLU = nn.ReLU()
        block = [nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1),
                 nn.BatchNorm2d(ch_num),
                 nn.ReLU(),
                 nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1),
                 nn.BatchNorm2d(ch_num)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = x+self.block(x)
        y = self.ReLU(y)
        return y


class DBSABlock(nn.Module):
    """DBSA（双分支空间注意力）模块，源于AG"""
    def __init__(self, ch_num):
        super().__init__()
        self.ch_num = ch_num
        self.conv1 = nn.Conv2d(ch_num, ch_num//8, 1, 1, 0)
        self.conv2 = nn.Conv2d(ch_num, ch_num//8, 1, 1, 0)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(ch_num//8, 2, 1, 1, 0)
        self.softmax = nn.Softmax(1)
        self.eps = 1e-7

    def forward(self, x1, x2, show_attention=False):
        f = self.relu(self.conv1(x1) + self.conv2(x2))
        f = self.softmax(self.conv3(f))
        s1 = f[:, [0], :, :]
        s2 = f[:, [1], :, :]
        if show_attention:
            return s1*x1+s2*x2, [s1, s2]
        else:
            return s1*x1+s2*x2


class MSFEBlock(nn.Module):
    """MSFE（多尺度特征提取）模块"""
    def __init__(self, ch_num, scale=(1, 3, 5)):
        super().__init__()
        block = [nn.Conv2d(ch_num, ch_num, 3, 1, scale[0], scale[0]),]
        self.scale1 = nn.Sequential(*block)
        block = [nn.Conv2d(ch_num, ch_num, 3, 1, scale[0], scale[0]),
                 nn.Conv2d(ch_num, ch_num, 3, 1, scale[1], scale[1]),]
        self.scale2 = nn.Sequential(*block)
        block = [nn.Conv2d(ch_num, ch_num, 3, 1, scale[0], scale[0]),
                 nn.Conv2d(ch_num, ch_num, 3, 1, scale[1], scale[1]),
                 nn.Conv2d(ch_num, ch_num, 3, 1, scale[2], scale[2]), ]
        self.scale3 = nn.Sequential(*block)

    def forward(self, x):
        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        return [x, f1, f2, f3]


class SFFBlock(nn.Module):
    """SFF（尺度特征融合）模块"""
    def __init__(self, ch_num):
        super().__init__()
        self.ch_num = ch_num
        self.conv1 = nn.Conv2d(ch_num*2, ch_num, 1, 1, 0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(ch_num, ch_num//8, 1, 1, 0)
        self.relu = nn.ReLU()
        self.conv3_1 = nn.Conv2d(ch_num//8, ch_num, 1, 1, 0)
        self.conv3_2 = nn.Conv2d(ch_num//8, ch_num, 1, 1, 0)
        self.conv3_3 = nn.Conv2d(ch_num//8, ch_num, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, f1, f2):
        f3 = self.conv1(torch.cat((f1, f2), 1))
        f4 = self.relu(self.conv2(self.gap(f3)))
        f4_1 = self.conv3_1(f4)
        f4_2 = self.conv3_2(f4)
        f4_3 = self.conv3_3(f4)
        f4_12 = torch.cat((f4_1, f4_2), dim=1)
        f4_12 = f4_12.view(-1, 2, self.ch_num, 1, 1)
        f4_12 = self.softmax(f4_12)
        s1 = f4_12[:, 0, :, :, :]
        s2 = f4_12[:, 1, :, :, :]
        s3 = self.sigmoid(f4_3)
        return s1*f1+s2*f2+s3*f3


class MSFFBlock(nn.Module):
    """MSFF（多尺度特征融合）模块"""
    def __init__(self, ch_num, ch_out):
        super().__init__()
        self.sff1 = SFFBlock(ch_num)
        self.sff2 = SFFBlock(ch_num)
        block = [nn.Conv2d(ch_num, (ch_num+ch_out)//2, 1, 1),
                 nn.BatchNorm2d((ch_num+ch_out)//2),
                 nn.ReLU(),
                 nn.Conv2d((ch_num+ch_out)//2, ch_out, 1, 1)]
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        y = self.sff1(x[1], x[2])
        y = self.sff2(y, x[3])
        y = x[0] + y
        return self.conv(y)


class GuidanceBlock(nn.Module):
    """引导模块"""
    def __init__(self, ch_num, branch_num, ch_out, act_fn):
        super().__init__()
        assert len(ch_out) == len(act_fn) == branch_num
        self.blocks = nn.ModuleList()
        for i in range(branch_num):
            blocks = [MSFFBlock(ch_num, ch_out[i]), act_fn[i]]
            self.blocks.append(nn.Sequential(*blocks))

    def forward(self, x):
        y = []
        for i in range(len(self.blocks)):
            y.append(self.blocks[i](x))
        return y


# self.opt.device self.opt.learning_rate
# self.opt.epoch_start self.opt.epoch_end self.opt.epoch_fixed_lr
# self.opt.weight_path
class BaseNet(nn.Module):
    def __init__(self, opt):
        """基本网络结构"""
        super().__init__()
        self.name = 'base_net'  # 网络名称
        self.opt = opt
        self.optimizer = None
        self.scheduler = None
        self.parameter_number = 0

    def set_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.Adam(self.parameters(), lr=self.opt.learning_rate)

    def set_scheduler(self):
        """
        设置规划器
        学习率在固定一定轮数后以线性衰减
        """
        fn = function_for_weight_decay(self.opt.epoch_start, self.opt.epoch_end, self.opt.epoch_fixed_lr)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fn)

    def calculate_parameter_number(self):
        """计算网络参数"""
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        self.parameter_number = num_params

    def set_clip_grad_norm(self, max_norm=0.1):
        """梯度裁剪，缓解梯度爆炸"""
        utils.clip_grad_norm_(self.parameters(), max_norm)

    def set_requires_grad(self, requires_grad):
        """设置参数是否需要计算梯度"""
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

    def save_network(self, save_flag):
        """保存网络参数"""
        save_filename = f'{save_flag}_{self.name}.pth'
        save_path = self.opt.weight_path / save_filename
        torch.save(self.cpu().state_dict(), save_path)
        self.to(self.opt.device)

    def init_weight(self):
        """网络权重初始化"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight.data)
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight.data, 1)
                nn.init.constant_(module.bias.data, 0)

    def init_net(self):
        """初始化网络"""
        # 设置网络子结构
        self.set_blocks()
        # 设置优化器
        self.set_optimizer()
        # 设置规划器
        self.set_scheduler()
        # 计算网络参数量
        self.calculate_parameter_number()
        # 存放到指定设备
        self.to(self.opt.device)


# self.opt.ch_num_in, self.opt.ch_num_n, self.opt.res_num_n
class FENet(BaseNet):
    def __init__(self, opt):
        """特征提取网络"""
        super().__init__(opt)
        self.name = 'fe_net'  # 网络名称
        self.blocks = nn.ModuleList()
        self.init_net()

    def set_blocks(self):
        ch_in, ch_n = self.opt.ch_num_in, self.opt.ch_num_n
        assert len(self.opt.res_num_n) == 5
        res_n_h, res_n_l, res_n_b, res_n_f, res_n_f2 = self.opt.res_num_n
        tf_ps_flag = self.opt.position_embedding
        tf_img_sz = self.opt.image_pad_size[0] // (2**5)
        # Conv-BatchNorm-PReLU
        conv_generator = get_block_list_generator([nn.Conv2d, nn.BatchNorm2d, nn.PReLU])
        # ConvTranspose-BatchNorm-PReLU-Conv-BatchNorm结构
        conv_t_generator = get_block_list_generator([nn.ConvTranspose2d, nn.BatchNorm2d, nn.PReLU,
                                                     nn.Conv2d, nn.BatchNorm2d])
        # region 0_FeatureExtraction
        # Conv1
        args = [[ch_in, ch_n, 3, 1, 1], [ch_n], []]
        kwargs = [{}, {}, {}]
        block = conv_generator(args, kwargs)
        args = [[ch_n, ch_n, 3, 1, 1], [ch_n], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n, ch_n, 3, 2, 1], [ch_n], []]
        block += conv_generator(args, kwargs)
        # Conv2
        args = [[ch_n, ch_n*2, 3, 1, 1], [ch_n*2], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n*2, ch_n*2, 3, 1, 1], [ch_n*2], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n*2, ch_n*2, 3, 2, 1], [ch_n*2], []]
        block += conv_generator(args, kwargs)
        # Conv3
        args = [[ch_n*2, ch_n*4, 3, 1, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n*4, ch_n*4, 3, 1, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        block += conv_generator(args, kwargs)
        args = [[ch_n*4, ch_n*4, 3, 2, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        self.blocks.append(nn.Sequential(*block))               # blocks_0  特征提取
        # endregion
        # region 1_HighQualityBranch
        # ResBlocks1
        block = [PlainResidualBlock(ch_n*4) for _ in range(res_n_h)]
        self.blocks.append(nn.Sequential(*block))               # blocks_1  高质量分支
        # endregion
        # region 2_LowQualityBranch
        # Conv4
        args = [[ch_n*4, ch_n*4, 3, 1, 1], [ch_n*4], []]
        block = conv_generator(args, kwargs)
        args = [[ch_n*4, ch_n*4, 3, 2, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        # Conv5
        args = [[ch_n*4, ch_n*4, 3, 1, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n*4, ch_n*4, 3, 2, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        # Transformer
        tf_block = TransformerBlock(in_chans=ch_n*4, embed_dim=ch_n*4, patch_size=1, pos_embed_flag=tf_ps_flag,
                                    img_sz=tf_img_sz, out_chans=ch_n*4, m2block_n=res_n_l, num_heads=8,
                                    hidden_features=ch_n*16, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        block.append(tf_block)
        # DeConv5
        args = [[ch_n*4, ch_n*4, 3, 2, 1, 1], [ch_n*4], [], [ch_n*4, ch_n*4, 3, 1, 1], [ch_n*4]]
        kwargs = [{}, {}, {}, {}, {}]
        block += conv_t_generator(args, kwargs)
        # DeConv4
        args = [[ch_n*4, ch_n*4, 3, 2, 1, 1], [ch_n*4], [], [ch_n*4, ch_n*4, 3, 1, 1], [ch_n*4]]
        block += conv_t_generator(args, kwargs)
        self.blocks.append(nn.Sequential(*block))               # blocks_2  低质量分支
        # endregion
        # region 3_LowQualityGuidance
        # MSFEBlock2_GuidanceBlock2
        block = [MSFEBlock(ch_n*4), GuidanceBlock(ch_n*4, 2, [2, 1], [nn.Tanh(), nn.Sigmoid()])]
        self.blocks.append(nn.Sequential(*block))               # blocks_3  低质量分支引导模块
        # endregion
        # region 4_BackgroundBranch
        # Conv4
        args = [[ch_n * 4, ch_n * 4, 3, 1, 1], [ch_n * 4], []]
        block = conv_generator(args, kwargs)
        args = [[ch_n * 4, ch_n * 4, 3, 2, 1], [ch_n * 4], []]
        block += conv_generator(args, kwargs)
        # Conv5
        args = [[ch_n * 4, ch_n * 4, 3, 1, 1], [ch_n * 4], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n * 4, ch_n * 4, 3, 2, 1], [ch_n * 4], []]
        block += conv_generator(args, kwargs)
        # ResBlocks3
        if res_n_b > 0:
            block += [PlainResidualBlock(ch_n * 4) for _ in range(res_n_b)]
        # DeConv5
        args = [[ch_n * 4, ch_n * 4, 3, 2, 1, 1], [ch_n * 4], [], [ch_n * 4, ch_n * 4, 3, 1, 1], [ch_n * 4]]
        kwargs = [{}, {}, {}, {}, {}]
        block += conv_t_generator(args, kwargs)
        # DeConv4
        args = [[ch_n * 4, ch_n * 4, 3, 2, 1, 1], [ch_n * 4], [], [ch_n * 4, ch_n * 4, 3, 1, 1], [ch_n * 4]]
        block += conv_t_generator(args, kwargs)
        self.blocks.append(nn.Sequential(*block))               # blocks_4  背景分支
        # endregion
        # region 5_ForegroundAttention
        # DBSABlock1
        self.blocks.append(DBSABlock(ch_n * 4))                 # blocks_5  前景注意力模块
        # endregion
        # region 6_ForegroundFusion
        # ResBlocks4
        if res_n_f > 0:
            block = [PlainResidualBlock(ch_n*4) for _ in range(res_n_f)]
        else:
            block = [nn.Identity(), ]
        self.blocks.append(nn.Sequential(*block))               # blocks_6  前景融合模块
        # endregion
        # region 7_FinalAttention
        # DBSABlock2
        self.blocks.append(DBSABlock(ch_n * 4))                 # blocks_7  最终注意力模块
        # endregion
        # region 8_FinalFeatureExtraction
        # ResBlocks5_MSFEBlock5
        if res_n_f2 > 0:
            block = [PlainResidualBlock(ch_n*4) for _ in range(res_n_f2)]
        else:
            block = [nn.Identity(), ]
        block += [MSFEBlock(ch_n*4), ]
        self.blocks.append(nn.Sequential(*block))               # blocks_8  最终多尺度特征提取
        # endregion
        # region 9_FinalGuidance
        # blocks_9 最终引导模块
        block = [GuidanceBlock(ch_n*4, 2, [2, 1], [nn.Tanh(), nn.Sigmoid()]), ]
        self.blocks.append(nn.Sequential(*block))
        # endregion

    def forward(self, x, verbose=False):
        if verbose:
            y = self.blocks[0](x)
            y_h = self.blocks[1](y)
            y_l = self.blocks[2](y)
            o_l, f_l = self.blocks[3](y_l)
            y_b = self.blocks[4](y)
            y_f, at1 = self.blocks[5](y_h, y_l, show_attention=True)
            y_f = self.blocks[6](y_f)
            y_f2, at2 = self.blocks[7](y_f, y_b, show_attention=True)
            y_f2 = self.blocks[8](y_f2)
            o_f2, f_f2 = self.blocks[9](y_f2)
            a_h, a_l = at1
            a_f, a_b = at2
            return y_f2, o_f2, f_f2, y_f, y_h, y_l, o_l, f_l, y_b, a_h, a_l, a_f, a_b
        else:
            y = self.blocks[0](x)
            y_h = self.blocks[1](y)
            y_l = self.blocks[2](y)
            o_l, f_l = self.blocks[3](y_l)
            y_b = self.blocks[4](y)
            y_f = self.blocks[5](y_h, y_l)
            y_f = self.blocks[6](y_f)
            y_f2, at2 = self.blocks[7](y_f, y_b, show_attention=True)
            y_f2 = self.blocks[8](y_f2)
            o_f2, f_f2 = self.blocks[9](y_f2)
            a_f, a_b = at2
            return y_f2, o_f2, f_f2, o_l, f_l, a_f


# self.opt.ch_num_n
class ReNet(BaseNet):
    def __init__(self, opt):
        """重建网络"""
        super().__init__(opt)
        self.name = 're_net'  # 网络名称
        self.block = None
        self.init_net()

    def set_blocks(self):
        ch_n = self.opt.ch_num_n
        # Multi-Scale-Feature-Fusion Block
        block = [MSFFBlock(ch_n*4, ch_n*4),]
        # ConvTranspose-BatchNorm-PReLU-Conv-BatchNorm结构
        block_generator = get_block_list_generator([nn.ConvTranspose2d, nn.BatchNorm2d, nn.PReLU,
                                                    nn.Conv2d, nn.BatchNorm2d])
        # DeConv1
        args = [[ch_n*4, ch_n*4, 3, 2, 1, 1], [ch_n*4], [], [ch_n*4, ch_n*4, 3, 1, 1], [ch_n*4]]
        kwargs = [{}, {}, {}, {}, {}]
        block += block_generator(args, kwargs)
        # DeConv2
        args = [[ch_n*4, ch_n*2, 3, 2, 1, 1], [ch_n*2], [], [ch_n*2, ch_n*2, 3, 1, 1], [ch_n*2]]
        block += block_generator(args, kwargs)
        # DeConv3
        args = [[ch_n*2, ch_n, 3, 2, 1, 1], [ch_n], [], [ch_n, ch_n, 3, 1, 1], [ch_n]]
        block += block_generator(args, kwargs)
        # Conv4
        block += [nn.Conv2d(ch_n, 1, 7, 1, 3), nn.Tanh()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        y = self.block(x)
        y = (y + torch.ones_like(y)) * 0.5
        return y


# self.opt.ch_num_n
class MNet(BaseNet):
    def __init__(self, opt):
        '''细节点提取器'''
        super().__init__(opt)
        self.name = 'mnet'   # 网络名称
        self.block = None
        self.init_net()

    def set_blocks(self):
        ch_n = self.opt.ch_num_n
        # Conv-BatchNorm-LeakyReLU结构
        conv_generator = get_block_list_generator([nn.Conv2d, nn.BatchNorm2d, nn.LeakyReLU])
        # Conv1
        args = [[1, ch_n, 3, 1, 1], [ch_n], []]
        kwargs = [{}, {}, {}]
        block = conv_generator(args, kwargs)
        args = [[ch_n, ch_n, 3, 2, 1], [ch_n], []]
        block += conv_generator(args, kwargs)
        # Conv2
        args = [[ch_n, ch_n*2, 3, 1, 1], [ch_n*2], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n*2, ch_n*2, 3, 2, 1], [ch_n*2], []]
        block += conv_generator(args, kwargs)
        # Conv3
        args = [[ch_n*2, ch_n*4, 3, 1, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        args = [[ch_n*4, ch_n*4, 3, 2, 1], [ch_n*4], []]
        block += conv_generator(args, kwargs)
        # Conv4
        block += [PlainResidualBlock(ch_n*4) for _ in range(3)]
        block += [nn.Conv2d(ch_n*4, 1, 3, 1, 1), nn.Sigmoid()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


"""
# self.opt.ch_num_n
class MNet(BaseNet):
    def __init__(self, opt):
        '''细节点提取器'''
        super().__init__(opt)
        self.name = 'mnet'   # 网络名称
        self.block = None
        self.init_net()

    def set_blocks(self):
        ch_n = self.opt.ch_num_n
        # Conv-BatchNorm-LeakyReLU结构
        block_generator = get_block_list_generator([nn.Conv2d, nn.BatchNorm2d, nn.LeakyReLU])
        # Conv1
        block = [nn.Conv2d(1, ch_n, 4, 2, 1), nn.LeakyReLU()]
        # Conv2
        args = [[ch_n, ch_n*2, 4, 2, 1], [ch_n*2], []]
        kwargs = [{}, {}, {}]
        block += block_generator(args, kwargs)
        # Conv3
        args = [[ch_n*2, ch_n*4, 4, 2, 1], [ch_n*4], []]
        block += block_generator(args, kwargs)
        # Conv4
        block += [nn.Conv2d(ch_n*4, 1, 3, 1, 1), nn.Sigmoid()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
"""