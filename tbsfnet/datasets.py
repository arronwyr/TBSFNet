"""实验使用的数据
修改preprocess_data和MyDataset的__getitem__"""
import torch
import torch.nn.functional as tnf
from torch.utils.data import Dataset
from fplab.tools.tensor import imt_read, imt_zscore
from fplab.intrinsic.orientation.gradient import imt_rao1990
from pathlib import Path
from gc import collect
from utilizations import average_orientation_nd


def preprocess_data(im, lb=None, mk=None, fq=None, mt=None, need_zscore=True, test_mode=False):
    """对数据进行预处理"""
    if need_zscore:
        im_in = imt_zscore(im, 0., 1.)
    else:
        im_in = im
    if test_mode:
        return im_in
    else:
        # 标签
        lb_out = lb
        # 图像mask
        lb_mk = mk
        # 分割标签
        lb_sg = tnf.avg_pool2d(mk, 8)
        lb_sg_b = torch.where(lb_sg >= 0.99, torch.ones_like(lb_sg), torch.zeros_like(lb_sg))
        # 频率标签
        lb_fq = tnf.avg_pool2d(fq, 8)
        lb_fq = torch.clip(lb_fq, 0., 1.)
        lb_fq = lb_fq*lb_sg_b
        # 细节点标签
        lb_mt = tnf.avg_pool2d(mt, 8)
        lb_mt = torch.clip(lb_mt, 0., 1.)
        # 方向场标签
        lb_ot = imt_rao1990(lb, blk_sz=(16, 16), need_quality=False)
        lb_ot = average_orientation_nd(lb_ot, blk_sz=(16, 16), s=(8, 8))
        lb_ot = torch.cat((torch.cos(2*lb_ot), torch.sin(2*lb_ot)), -3)
        lb_ot = lb_ot*lb_sg_b
        return im_in, lb_out, lb_mk, lb_sg, lb_sg_b, lb_fq, lb_mt, lb_ot


class MyDataset(Dataset):
    """
    网络使用的数据集
    从一个指定的文件读取保存所有数据地址的文件地址
    """
    def __init__(self, d_path, device, test_mode=False, need_inverse=True):
        super().__init__()
        self.dirs_path = d_path     # 指定的文件地址，保存地址文件的路径
        self.device = device
        self.test_mode = test_mode
        self.need_inverse = need_inverse
        self.dirs_file = Path("")   # 地址文件路径
        self.update_dirs_file()     # 更新地址文件
        self.dirs_number = 0    # 数据数目
        self.update_dirs_number()   # 更新数据数目

    def update_dirs_file(self):
        """更新地址文件"""
        with open(self.dirs_path, 'r') as dpf:
            self.dirs_file = Path(dpf.readline())
        del dpf
        collect()

    def update_dirs_number(self):
        """更新数据数目"""
        with open(self.dirs_file, 'r') as df:
            self.dirs_number = sum(1 for _ in df)
        del df
        collect()

    def __len__(self):
        return self.dirs_number

    def __getitem__(self, idx):
        self.update_dirs_file()     # 更新地址文件
        with open(self.dirs_file, 'r') as df:
            for i in range(idx+1):
                text = df.readline()    # 获取文件中第idx个路径
                text = text.strip()
        if self.test_mode:
            return text
        else:
            im_dir, lb_dir, mk_dir, fq_dir, mt_dir = text.split("\t")
            # im为图像，lb为标签，mk为mask（值为0或1），fq为频率图，mt为细节点图
            im = imt_read(Path(im_dir), self.device)
            if self.need_inverse and torch.randn(1).item() >= 0:
                im = 1 - im
            lb = imt_read(Path(lb_dir), self.device)
            mk = imt_read(Path(mk_dir), self.device)
            fq = imt_read(Path(fq_dir), self.device)
            mt = imt_read(Path(mt_dir), self.device)
            return {'image': im, 'label': lb, 'mask': mk, 'frequency': fq, 'minutiae': mt}
