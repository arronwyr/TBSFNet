"""根据模型添加或修改参数"""
import math
from argparse import ArgumentParser
from pathlib import Path
from time import strftime, localtime
from gc import collect


class MyOptions:
    """训练网络需要的参数"""
    def __init__(self):
        self.parser = ArgumentParser()  # 初始化分析器
        self.set_options()    # 添加参数
        self.opt = self.parser.parse_args()     # 开始分析
        self.induce_options()   # 推断选项
        self.save_options()     # 保存选项

    def set_options(self):
        """给分析器设置基本的参数，即训练大多数模型都需要的参数"""
        # 输入输出地址
        self.parser.add_argument('--input_files_dir', type=str, default=None, help='信息文件所在路径')
        self.parser.add_argument('--output_dir', type=str, default=None, help='输出的位置')
        # 数据选项
        self.parser.add_argument('--batch_size', type=int, default=32, help='一次处理多少数据')
        self.parser.add_argument('--num_workers', type=int, default=0, help='处理数据的进程数')
        self.parser.add_argument('--pin_memory', action='store_true', default=False, help='是否pin_memory')
        self.parser.add_argument('--no_shuffle', action='store_false',
                                 dest='shuffle', default=True, help='禁用乱序加载数据')
        self.parser.add_argument('--data_per_epoch', type=int, default=-1, help='每轮训练处理数据量')
        self.parser.add_argument('--validate_per_epoch', type=int, default=-1, help='每轮验证处理数据量')
        # 训练选项
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='-1(CPU)、n(单GPU)')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001, help='初始学习率')
        self.parser.add_argument('--epoch_start', type=int, default=0, help='开始迭代轮数')
        self.parser.add_argument('--epoch_end', type=int, default=30, help='结束迭代轮数')
        self.parser.add_argument('--epoch_fixed_lr', type=int, default=60, help='学习率不变轮数')
        self.parser.add_argument('--prepare_data_freq', type=int, default=60, help='准备数据的频率')
        self.parser.add_argument('--test_freq', type=int, default=60, help='测试的频率')
        self.parser.add_argument('--save_freq', type=int, default=60, help='保存网络权重的频率')
        self.parser.add_argument('--fixed_seed', type=int, default=2024, help='程序中使用的随机数种子')
        # 断点训练
        self.parser.add_argument('--load_dir', type=str, default=None, help='权重所在位置')
        # 模型选项
        self.parser.add_argument('--name', type=str, default='Proposed', help='模型的名字')
        self.parser.add_argument('--ch_num_in', type=int, default=1, help='输入通道数')
        self.parser.add_argument('--ch_num_n', type=int, default=64, help='基本通道数')
        # 特有选项
        self.parser.add_argument('--epoch_stage1', type=int, default=0, help='热身阶段轮数')
        self.parser.add_argument('--epoch_stage2', type=int, default=0, help='生成器训练阶段轮数')
        self.parser.add_argument('--load_weight_num', type=int, default=0, help='读取权重数目')
        self.parser.add_argument('--m_lr', type=float, default=1., help='细节点损失的权重')
        self.parser.add_argument('--m_r', type=float, default=1., help='细节点区域的方向频率权重系数')
        self.parser.add_argument('--m8_r', type=float, default=1., help='细节点区域的重建权重系数')
        self.parser.add_argument('--r_lr', type=float, default=1., help='重建损失的权重')
        self.parser.add_argument('--s_lr', type=float, default=1., help='分割损失的权重')
        self.parser.add_argument('--o_l_lr', type=float, default=1., help='低质量方向损失的权重')
        self.parser.add_argument('--f_l_lr', type=float, default=0., help='低质量频率损失的权重')
        self.parser.add_argument('--o_f2_lr', type=float, default=1., help='最终方向损失的权重')
        self.parser.add_argument('--f_f2_lr', type=float, default=0., help='最终频率损失的权重')
        self.parser.add_argument('--image_size_max', type=int, default=448, help='输入图像边长最大值')
        self.parser.add_argument('--image_pad_size', type=int,
                                 nargs='+', default=[512, 512], help='测试图像填充大小')
        self.parser.add_argument('--res_num_n', type=int,
                                 nargs='+', default=[3, 6, 3, 3, 6], help='各网络残差块数')
        self.parser.add_argument('--no_inverse', action='store_false',
                                 dest='need_inverse', default=True, help='禁用输入随机取反')
        self.parser.add_argument('--no_zscore', action='store_false',
                                 dest='need_zscore', default=True, help='禁用输入标准化')
        self.parser.add_argument('--no_position_embedding', action='store_false',
                                 dest='position_embedding', default=True, help='禁用位置信息嵌入')

    def induce_options(self):
        """推断特定参数"""
        # 输入文件夹下的各种信息文件地址
        self.opt.flag_path = Path(self.opt.input_files_dir) / "flag.txt"
        self.opt.train_path = Path(self.opt.input_files_dir) / "train_dirs.txt"
        self.opt.validate_path = Path(self.opt.input_files_dir) / "validate_dirs.txt"
        self.opt.test_path = Path(self.opt.input_files_dir) / "test_dirs.txt"
        # 模型开始时间
        self.opt.time = strftime("%Y%m%d%H%M%S", localtime())
        # 在输出文件夹下新建子文件夹，并建立权重文件夹和结果文件夹
        self.opt.output_path = Path(self.opt.output_dir) / (self.opt.name + "_" + self.opt.time)
        self.opt.weight_path = self.opt.output_path / "weights"
        self.opt.result_path = self.opt.output_path / "results"
        self.opt.output_path.mkdir(parents=True, exist_ok=True)
        self.opt.weight_path.mkdir(parents=True, exist_ok=True)
        self.opt.result_path.mkdir(parents=True, exist_ok=True)
        self.opt.script_path = self.opt.output_path / "scripts.txt"
        self.opt.train_loss_path = self.opt.output_path / "train_loss.txt"
        self.opt.validate_loss_path = self.opt.output_path / "validate_loss.txt"
        # 根据gpu_ids确定模型使用的device
        if self.opt.gpu_ids == '-1':
            self.opt.device = 'cpu'
        else:
            self.opt.device = f'cuda:{int(self.opt.gpu_ids)}'
        # 处理随机数种子
        if self.opt.fixed_seed <= 0:
            self.opt.fixed_seed = int(strftime("%H%M%S", localtime()))
        # 计算每轮训练迭代次数
        if self.opt.data_per_epoch <= 0:
            self.opt.max_iter = 0       # 最大迭代次数设为0，则不判断是否达到最大迭代次数
        else:
            self.opt.max_iter = math.ceil(self.opt.data_per_epoch/self.opt.batch_size)
        # 计算每轮验证迭代次数
        if self.opt.validate_per_epoch <= 0:
            self.opt.max_iter_validate = 0  # 最大迭代次数设为0，则不判断是否达到最大迭代次数
        else:
            self.opt.max_iter_validate = math.ceil(self.opt.validate_per_epoch / self.opt.batch_size)

    def save_options(self):
        """保存参数"""
        msg = ''
        for k, v in sorted(vars(self.opt).items()):
            msg += f'{str(k)}: {str(v)}\n'
        path = self.opt.output_path
        file_name = path / f'options.txt'
        with open(file_name, 'w') as opt_file:
            opt_file.write(msg)
        del msg, k, v, path, file_name
        collect()
