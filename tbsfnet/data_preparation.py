# 这个程序将数据准备过程从网络训练中分离出来
# 必须在网络训练的同时运行该程序
# 该程序会在指定目录下生成两个子目录、一个指示文件、一个信息文件、三个路径文件，并在子目录下生成三个路径文件
# 三个路径文件分别代表训练集、验证集、测试集
# 初始时会从数据源处读取并处理数据，结果存放在其中的一个子目录中
# 指定目录下的路径文件将指示该子目录下的路径文件地址，信息文件存储命令行提示信息，指示文件传递对程序的指示
# 随后程序进入休眠状态
# 一段时间后，程序唤醒，读取指示文件判断是否需要再次准备数据
# 如果需要，读取并处理数据，并将结果存放在另一个目录，路径文件也随之改变
# 如果不需要，继续休眠
# 这个过程重复进行
# 这样，训练网络时可以通过指示文件指示程序准备数据，通过程序给出的地址读取数据
# 该程序的主程序具有普适性，一般不需要修改
# 该程序的prepare_data函数承担了从in_path到out_path的数据准备任务，使用的参数直接通过命令行传递
# 修改该程序时建议增加parse_argument的参数并修改prepare_data函数
import os
import random
import numpy as np
import albumentations as ag
import tqdm
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from time import strftime, localtime
from scipy.ndimage import distance_transform_edt
from fplab.tools.image import get_data_dirs
from fplab.tools.array import IMA


def parse_argument():
    """读取命令行参数"""
    parser = ArgumentParser()
    # 基本参数
    parser.add_argument('--source_dir', type=str, default='./data', help='数据源的位置')
    parser.add_argument('--output_dir', type=str, default='./input', help='数据存放位置')
    parser.add_argument('--sleep_time', type=int, default=1800, help='休眠时间')
    parser.add_argument('--fixed_seed', type=int, default=0, help='程序中使用的随机数种子')
    # prepare_data参数，根据需要添加
    # 常用参数
    parser.add_argument('--output_size', type=int, nargs='+', default=[512, 512], help='输出数据大小')
    # 特殊变换
    parser.add_argument('--minutiae_sigma', type=float, default=8.0, help='细节点权重图的sigma')
    # 获取参数并添加信息
    opt = parser.parse_args()
    opt.message = "----------------------------------------------------------------\n"
    opt.message += "该程序将数据准备的过程从模型训练的过程中分离出来，以减少训练时间。\n"
    opt.message += "该程序将符合网络要求的训练数据、验证数据、测试数据的地址存放到文本文档中，网络可以据此直接读取数据。\n"
    opt.message += "数据源下的数据文件将根据路径中是否有 'train', 'validate' 和 'test' 被分为训练集、验证集和测试集，"
    opt.message += "并被执行不同操作，因此应该注意数据文件的路径（至少是相对路径）。\n"
    opt.message += "----------------------------------------------------------------\n"
    return opt


def prepare_data(in_path, out_path, opt):
    assert opt
    """准备数据并存放到指定目录下。"""
    train_path = out_path / "train.txt"  # 训练数据路径文件
    validate_path = out_path / "validate.txt"  # 验证数据路径文件
    test_path = out_path / "test.txt"  # 测试数据路径文件
    train_path.unlink(missing_ok=True)
    validate_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)
    # 参数设置
    # 拷贝变换：无变换
    transforms = [
        ag.NoOp(),
    ]
    copy_transform = ag.Compose(transforms)
    # 增强变换：高斯噪声
    transforms = [
        ag.GaussNoise(),
    ]
    augment_transform = ag.Compose(transforms)
    # 获取所有数据路径
    data_dirs0 = []
    data_dirs0 = get_data_dirs(in_path, data_dirs0)
    data_dirs = []
    for d in data_dirs0:
        rf_d = d.replace(str(in_path.absolute()), "")
        if ("images" in rf_d) or ("test" in rf_d):
            data_dirs.append(d)
    with tqdm.tqdm(total=len(data_dirs)) as pbar:
        pbar.set_description("准备数据中")
        for d in data_dirs:
            rf_d = d.replace(str(in_path.absolute()), "")
            if 'test' in rf_d:
                res_d = str(out_path.absolute()) + rf_d
                IMA.read(d).save(res_d)
                with open(test_path, "a") as tpf:
                    tpf.write(str(res_d) + "\n")
            else:
                # 准备图像、标签、mask、频率图和细节点图
                lb0_dir = str(in_path.absolute()) + rf_d.replace("images", "labels")
                mk0_dir = str(in_path.absolute()) + rf_d.replace("images", "masks")
                fq0_dir = str(in_path.absolute()) + rf_d.replace("images", "frequencies")
                mt0_dir = str(in_path.absolute()) + rf_d.replace("images", "minutiae").replace(".png", "_point.png")
                im0, lb0, mk0 = IMA.read(d).rgb2l(), IMA.read(lb0_dir).rgb2l(), IMA.read(mk0_dir).rgb2l()
                fq0, mt0 = IMA.read(fq0_dir).rgb2l(), IMA.read(mt0_dir).rgb2l()
                # 保存图像
                # 图像添加高斯噪声
                im0 = im0.float2uint()
                im1_dir = str(out_path.absolute()) + rf_d
                if "train" in rf_d:
                    im1 = augment_transform(image=im0.ima)["image"]
                    IMA(im1).uint2float().save(im1_dir)
                elif "validate" in rf_d:
                    im1 = copy_transform(image=im0.ima)["image"]
                    IMA(im1).uint2float().save(im1_dir)
                # 保存mask
                # mask转化为二值图
                mk1_dir = str(out_path.absolute()) + rf_d.replace("images", "masks")
                mk1 = mk0.ima
                mk1 = np.where(mk1 > 0, np.ones_like(mk1), np.zeros_like(mk1))
                IMA(mk1).save(mk1_dir)
                # 保存标签
                # 选取目标区域的标签
                lb1_dir = str(out_path.absolute()) + rf_d.replace("images", "labels")
                inv_flag = lb0.ima[:10, :10].mean() > 0.5
                if inv_flag:
                    temp = 1 - lb0.ima
                    temp = temp*mk1
                    temp = 1 - temp
                    lb1 = IMA(temp)
                else:
                    lb1 = IMA(lb0.ima*mk1)
                lb1.save(lb1_dir)
                # 保存频率图
                fq1_dir = str(out_path.absolute()) + rf_d.replace("images", "frequencies")
                fq1 = IMA(fq0.ima*mk1)
                fq1.save(fq1_dir)
                # 保存细节点
                # 根据任务需要处理细节点，sigma小于0则使用距离倒数
                mt1_dir = str(out_path.absolute()) + rf_d.replace("images", "minutiae")
                mt1 = np.where(mt0.ima > 0, np.zeros_like(mt0.ima), np.ones_like(mt0.ima))
                mt1 = distance_transform_edt(mt1)
                if opt.minutiae_sigma <= 0:
                    mt1 = 1/(mt1+1)
                else:
                    mt1 = np.exp(-mt1**2/(2*opt.minutiae_sigma**2))
                IMA(mt1).save(mt1_dir)
                if "train" in rf_d:
                    with open(train_path, "a") as tpf:
                        cmd = str(im1_dir) + "\t" + str(lb1_dir) + "\t" + str(mk1_dir) + "\t"
                        cmd = cmd + str(fq1_dir) + "\t" + str(mt1_dir) + "\n"
                        tpf.write(cmd)
                elif "validate" in rf_d:
                    with open(validate_path, "a") as vpf:
                        cmd = str(im1_dir) + "\t" + str(lb1_dir) + "\t" + str(mk1_dir) + "\t"
                        cmd = cmd + str(fq1_dir) + "\t" + str(mt1_dir) + "\n"
                        vpf.write(cmd)
            pbar.update(1)


if __name__ == '__main__':
    # 读取参数
    options = parse_argument()
    source_path = Path(options.source_dir)  # 源数据路径
    output_path = Path(options.output_dir)  # 数据保存路径
    output_path.mkdir(parents=True, exist_ok=True)
    # 初始化程序参数
    prepare_iteration = 0   # 准备数据次数
    suffix_number = 1   # 保存路径后缀
    flag_path = output_path / "flag.txt"    # 指示文件路径
    script_path = output_path / "scripts.txt"   # 提示信息文件路径
    train_dirs_path = output_path / "train_dirs.txt"  # 训练集文件路径
    validate_dirs_path = output_path / "validate_dirs.txt"  # 验证集文件路径
    test_dirs_path = output_path / "test_dirs.txt"  # 测试集文件路径
    # 设置随机数
    if options.fixed_seed <= 0:
        options.fixed_seed = int(strftime("%H%M%S", localtime()))
    np.random.seed(options.fixed_seed)
    random.seed(options.fixed_seed)
    os.environ['PYTHONHASHSEED'] = str(options.fixed_seed)
    # 命令行输出提示信息
    msg = "数据准备程序开始运行！\n"
    msg += f"本程序将读取{str(source_path.absolute())}目录下的文件，\n"
    msg += f"并将结果保存到{str(output_path.absolute())}目录下。\n"
    msg += f"程序休眠时间为{str(options.sleep_time)}秒。\n"
    msg += f"程序随机数种子设置为{str(options.fixed_seed)}。\n"
    msg += options.message
    print(msg)
    # 将提示信息保存到指定文件
    with open(script_path, 'a') as sf:
        sf.write(msg)
    # 检查并初始化指示文件
    if flag_path.exists():
        pass
    else:
        with open(flag_path, 'w') as ff:
            ff.write("YES")  # 写入是否需要更新

    # 程序开始工作
    while True:
        with open(flag_path, 'r') as ff:
            flag = ff.readline().strip()
        if flag == 'STOP':
            # 如果收到STOP命令，那么程序停止工作
            msg = f"程序停止运行。\n"
            print(msg)
            with open(script_path, 'a') as sf:
                sf.write(msg)
            break
        elif flag == 'NO':
            # 如果收到NO命令，那么程序继续休眠
            msg = f"程序休眠中...\n"
            print(msg)
            with open(script_path, 'a') as sf:
                sf.write(msg)
            sleep(options.sleep_time)
            continue
        elif flag == 'YES':
            # 如果收到YES命令，那么程序将准备数据
            # 更改保存路径后缀
            msg = f"程序开始第{prepare_iteration+1}次数据准备：\n"
            print(msg)
            with open(script_path, 'a') as sf:
                sf.write(msg)
            suffix_number = 1 if suffix_number == 0 else 0
            specified_path = output_path / f"data{suffix_number}"
            # 准备数据
            start_time = time()
            prepare_data(source_path, specified_path, options)
            end_time = time()
            prepare_iteration += 1  # 更新增强次数
            # 更新指示文件
            with open(flag_path, 'w') as ff:
                ff.write("NO")  # 写入是否需要更新
            with open(train_dirs_path, 'w') as tdf:
                train_dirs = specified_path / "train.txt"
                tdf.write(f"{str(train_dirs.absolute())}")  # 写入训练数据所在路径
            with open(validate_dirs_path, 'w') as vdf:
                validate_dirs = specified_path / "validate.txt"
                vdf.write(f"{str(validate_dirs.absolute())}")  # 写入验证数据所在路径
            with open(test_dirs_path, 'w') as tdf:
                test_dirs = specified_path / "test.txt"
                tdf.write(f"{str(test_dirs.absolute())}")  # 写入测试数据所在路径
            msg = f"第{prepare_iteration}次数据准备花费时间{int(end_time - start_time)}秒。\n"
            print(msg)
            with open(script_path, 'a') as sf:
                sf.write(msg)
        else:
            raise (Exception("未识别的命令！"))
