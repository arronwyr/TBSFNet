"""
模型训练代码
大多数情况下，初始化模型之后都需要修改
"""
import torch
import time
import os
import random
import cv2
import numpy as np
import gc
from pathlib import Path
from torch.utils import data
from fplab.tools.image import change_image
from fplab.tools.array import IMA
from fplab.tools.tensor import IMT
from options import MyOptions
from datasets import MyDataset, preprocess_data
from models import FENet, ReNet, MNet
from models import MinutiaeLoss, ReconstructionLoss, OrientationLoss, FrequencyLoss, SegmentationLoss


if __name__ == '__main__':
    # 获取选项
    options = MyOptions().opt
    # 设置随机数种子
    np.random.seed(options.fixed_seed)
    random.seed(options.fixed_seed)
    os.environ['PYTHONHASHSEED'] = str(options.fixed_seed)
    torch.manual_seed(options.fixed_seed)
    torch.cuda.manual_seed(options.fixed_seed)
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # 检查数据是否准备就绪
    while True:
        with open(options.flag_path, 'r') as ff:
            flag = ff.readline().strip()
        if flag == 'NO':
            break
        else:
            time.sleep(10)
    msg = "数据集准备就绪！\n"
    # 训练数据集
    train_dataset = MyDataset(options.train_path, options.device, need_inverse=options.need_inverse)
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=options.batch_size,
                                       shuffle=options.shuffle,
                                       num_workers=options.num_workers,
                                       pin_memory=options.pin_memory)
    # 验证数据集
    validate_dataset = MyDataset(options.validate_path, options.device, need_inverse=options.need_inverse)
    validate_dataloader = data.DataLoader(validate_dataset,
                                          batch_size=options.batch_size,
                                          shuffle=False,
                                          num_workers=options.num_workers,
                                          pin_memory=options.pin_memory)
    # 测试数据集
    test_dataset = MyDataset(options.test_path, options.device, test_mode=True)
    msg += f"共有训练数据{train_dataset.__len__()}个，验证数据{validate_dataset.__len__()}个，"
    msg += f"测试数据{test_dataset.__len__()}个。\n"
    print(msg)
    with open(options.script_path, "a") as sf:
        sf.write(msg)
    # 初始化模型
    fe_net = FENet(options)
    re_net = ReNet(options)
    mnet = MNet(options)
    nets = [fe_net, re_net, mnet]
    msg = ""
    for i in range(options.load_weight_num):
        nets[i].load_state_dict(torch.load(options.load_dir.replace(nets[0].name, nets[i].name), map_location=options.device))
        msg += f"网络{nets[i].name}加载完毕，参数总数为: {nets[i].parameter_number/1e6:.2f}M。\n"
    for i in range(options.load_weight_num, len(nets)):
        nets[i].init_weight()
        msg += f"网络{nets[i].name}初始化完毕，参数总数为: {nets[i].parameter_number/1e6:.2f}M。\n"
    print(msg)
    with open(options.script_path, "a") as sf:
        sf.write(msg)
    loss_m = MinutiaeLoss()
    loss_r = ReconstructionLoss()
    loss_o = OrientationLoss()
    loss_f = FrequencyLoss()
    loss_s = SegmentationLoss()
    # 初始化参数
    max_validate_loss = 1000
    epoch_best = 0
    # 程序正式开始
    for epoch in range(options.epoch_start, options.epoch_end):
        # 命令辅助程序准备数据
        if (epoch + 1) % options.prepare_data_freq == 0:
            with open(options.flag_path, 'w') as ff:
                ff.write("YES")
        # 开始训练
        start_time = time.time()
        # 该轮的平均损失
        train_loss_mean = {"loss": 0, "m_loss": 0, "m_loss_lb": 0, "r_loss": 0, "s_loss": 0,
                           "o_l_loss": 0, "f_l_loss": 0, "o_f2_loss": 0, "f_f2_loss": 0}
        for net in nets:
            net.train()
        for i, data in enumerate(train_dataloader):
            # 图像和标签
            im, lb, mk, fq, mt = data['image'], data['label'], data['mask'], data['frequency'], data['minutiae']
            im_in, lb_out, lb_mk, lb_sg, lb_sg_b, lb_fq, lb_mt, lb_ot = preprocess_data(im, lb, mk, fq, mt, options.need_zscore)
            # 训练mnet
            mnet.set_requires_grad(True)
            mnet.optimizer.zero_grad()
            mt_out_lb = mnet(lb_out)
            m_loss_lb = loss_m(mt_out_lb, lb_mt, torch.ones_like(lb_mt), torch.ones_like(lb_mt))
            m_loss_lb.backward()
            mnet.optimizer.step()
            train_loss_mean["m_loss_lb"] += m_loss_lb.item()
            if epoch < options.epoch_stage1:
                # 使用重建损失训练fe_net和re_net
                for net in [fe_net, re_net, mnet]:
                    net.set_requires_grad(True)
                    net.optimizer.zero_grad()
                im_out = re_net(fe_net(im_in)[0])
                r_loss = loss_r(im_out, lb_out, lb_mk)
                r_loss.backward()
                for net in [fe_net, re_net]:
                    net.optimizer.step()
                train_loss_mean["r_loss"] += r_loss.item()
                train_loss_mean["loss"] += r_loss.item()
            elif epoch < options.epoch_stage2:
                # 使用除细节点损失外的损失训练fe_net和re_net
                for net in [fe_net, re_net]:
                    net.set_requires_grad(True)
                    net.optimizer.zero_grad()
                im_f, ot_f2, fq_f2, ot_l, fq_l, at_f = fe_net(im_in)
                im_out = re_net(im_f)
                r_loss = loss_r(im_out, lb_out, torch.ones_like(lb_out))
                train_loss_mean["r_loss"] += r_loss.item()
                s_loss = loss_s(at_f, lb_sg)
                train_loss_mean["s_loss"] += s_loss.item()
                o_l_loss = loss_o(ot_l, lb_ot, lb_sg_b, lb_sg_b)
                train_loss_mean["o_l_loss"] += o_l_loss.item()
                f_l_loss = loss_f(fq_l, lb_fq, lb_sg_b, lb_sg_b)
                train_loss_mean["f_l_loss"] += f_l_loss.item()
                o_f2_loss = loss_o(ot_f2, lb_ot, torch.ones_like(lb_ot), torch.ones_like(lb_ot))
                train_loss_mean["o_f2_loss"] += o_f2_loss.item()
                f_f2_loss = loss_f(fq_f2, lb_fq, torch.ones_like(lb_fq), torch.ones_like(lb_fq))
                train_loss_mean["f_f2_loss"] += f_f2_loss.item()
                loss = r_loss*options.r_lr+s_loss*options.s_lr
                loss += o_l_loss*options.o_l_lr+f_l_loss*options.f_l_lr
                loss += o_f2_loss*options.o_f2_lr+f_f2_loss*options.f_f2_lr
                loss.backward()
                train_loss_mean["loss"] += loss.item()
                for net in [fe_net, re_net]:
                    net.optimizer.step()
            else:
                # 加入细节点损失
                for net in [fe_net, re_net]:
                    net.set_requires_grad(True)
                    net.optimizer.zero_grad()
                mt_out_lb = mnet(lb_out).detach()
                im_f, ot_f2, fq_f2, ot_l, fq_l, at_f = fe_net(im_in)
                im_out = re_net(im_f)
                mt_out = mnet(im_out)
                m_loss = loss_m(mt_out, mt_out_lb, torch.ones_like(mt_out_lb), torch.ones_like(mt_out_lb))
                train_loss_mean["m_loss"] += m_loss.item()
                mt_out_8 = mt_out.repeat_interleave(8, -1).repeat_interleave(8, -2)
                r_loss = loss_r(im_out, lb_out, torch.ones_like(lb_out)+mt_out_8.detach()*options.m8_r)
                train_loss_mean["r_loss"] += r_loss.item()
                s_loss = loss_s(at_f, lb_sg)
                train_loss_mean["s_loss"] += s_loss.item()
                o_l_loss = loss_o(ot_l, lb_ot, lb_sg_b+mt_out.detach()*options.m_r*lb_sg_b)
                train_loss_mean["o_l_loss"] += o_l_loss.item()
                f_l_loss = loss_f(fq_l, lb_fq, lb_sg_b+mt_out.detach()*options.m_r*lb_sg_b)
                train_loss_mean["f_l_loss"] += f_l_loss.item()
                o_f2_loss = loss_o(ot_f2, lb_ot, torch.ones_like(lb_ot)+mt_out.detach()*options.m_r)
                train_loss_mean["o_f2_loss"] += o_f2_loss.item()
                f_f2_loss = loss_f(fq_f2, lb_fq, torch.ones_like(lb_fq)+mt_out.detach()*options.m_r)
                train_loss_mean["f_f2_loss"] += f_f2_loss.item()
                loss = r_loss*options.r_lr+s_loss*options.s_lr
                loss += o_l_loss*options.o_l_lr+f_l_loss*options.f_l_lr
                loss += o_f2_loss*options.o_f2_lr+f_f2_loss*options.f_f2_lr
                loss += m_loss*options.m_lr
                loss.backward()
                train_loss_mean["loss"] += loss.item()
                for net in [fe_net, re_net]:
                    net.optimizer.step()
            if options.max_iter:
                if i == options.max_iter-1:
                    break
        for net in [fe_net, re_net, mnet]:
            net.scheduler.step()
        train_dataset.update_dirs_number()
        temp = options.max_iter if options.max_iter else np.ceil(train_dataset.__len__() / options.batch_size)
        temp = min(temp, np.ceil(train_dataset.__len__() / options.batch_size))
        for k, w in train_loss_mean.items():
            train_loss_mean[k] = w/temp
        with open(options.train_loss_path, 'a') as tlf:
            for k, w in train_loss_mean.items():
                tlf.write(f"{k}\t{w}\t")
            tlf.write(f"\n")
        end_time = time.time()
        train_time = end_time - start_time
        # 开始验证
        start_time = time.time()
        validate_loss_mean = {"loss": 0, "m_loss": 0, "m_loss_lb": 0, "r_loss": 0, "s_loss": 0,
                              "o_l_loss": 0, "f_l_loss": 0, "o_f2_loss": 0, "f_f2_loss": 0}
        for net in nets:
            net.eval()
        for i, data in enumerate(validate_dataloader):
            im, lb, mk, fq, mt = data['image'], data['label'], data['mask'], data['frequency'], data['minutiae']
            im_in, lb_out, lb_mk, lb_sg, lb_sg_b, lb_fq, lb_mt, lb_ot = preprocess_data(im, lb, mk, fq, mt, options.need_zscore)
            with torch.no_grad():
                mt_out_lb = mnet(lb_out)
                m_loss_lb = loss_m(mt_out_lb, lb_mt, torch.ones_like(lb_mt), torch.ones_like(lb_mt))
                validate_loss_mean["m_loss_lb"] += m_loss_lb.item()
                im_f, ot_f2, fq_f2, ot_l, fq_l, at_f = fe_net(im_in)
                im_out = re_net(im_f)
                mt_out = mnet(im_out)
                m_loss = loss_m(mt_out, mt_out_lb, torch.ones_like(mt_out_lb), torch.ones_like(mt_out_lb))
                validate_loss_mean["m_loss"] += m_loss.item()
                mt_out_8 = mt_out.repeat_interleave(8, -1).repeat_interleave(8, -2)
                r_loss = loss_r(im_out, lb_out, torch.ones_like(lb_out)+mt_out_8.detach()*options.m8_r)
                validate_loss_mean["r_loss"] += r_loss.item()
                s_loss = loss_s(at_f, lb_sg)
                validate_loss_mean["s_loss"] += s_loss.item()
                o_l_loss = loss_o(ot_l, lb_ot, lb_sg_b+mt_out.detach()*options.m_r*lb_sg_b)
                validate_loss_mean["o_l_loss"] += o_l_loss.item()
                f_l_loss = loss_f(fq_l, lb_fq, lb_sg_b+mt_out.detach()*options.m_r*lb_sg_b)
                validate_loss_mean["f_l_loss"] += f_l_loss.item()
                o_f2_loss = loss_o(ot_f2, lb_ot, torch.ones_like(lb_ot)+mt_out.detach()*options.m_r)
                validate_loss_mean["o_f2_loss"] += o_f2_loss.item()
                f_f2_loss = loss_f(fq_f2, lb_fq, torch.ones_like(lb_fq)+mt_out.detach()*options.m_r)
                validate_loss_mean["f_f2_loss"] += f_f2_loss.item()
                loss = r_loss*options.r_lr+s_loss*options.s_lr
                loss += o_l_loss*options.o_l_lr+f_l_loss*options.f_l_lr
                loss += o_f2_loss*options.o_f2_lr+f_f2_loss*options.f_f2_lr
                loss += m_loss*options.m_lr
                validate_loss_mean["loss"] += loss.item()
            if options.max_iter_validate:
                if i == options.max_iter_validate-1:
                    break
        validate_dataset.update_dirs_number()
        temp = options.max_iter_validate if options.max_iter_validate else np.ceil(validate_dataset.__len__() / options.batch_size)
        temp = min(temp, np.ceil(train_dataset.__len__() / options.batch_size))
        for k, w in validate_loss_mean.items():
            validate_loss_mean[k] = w/temp
        with open(options.validate_loss_path, 'a') as vlf:
            for k, w in validate_loss_mean.items():
                vlf.write(f"{k}\t{w}\t")
            vlf.write(f"\n")
        end_time = time.time()
        validate_time = end_time - start_time
        # 开始测试
        start_time = time.time()
        if (epoch+1) % options.test_freq == 0:
            for net in nets:
                net.eval()
            for i in range(test_dataset.__len__()):
                im_d = test_dataset[i]
                im_d = str(Path(im_d).absolute())
                im_path_parts = Path(im_d).parts[Path(im_d).parts.index('test'):]
                sv_d = str(options.result_path.joinpath(*im_path_parts).absolute())
                shape = IMA.read(im_d).shape[:2]
                Path(sv_d).parent.mkdir(parents=True, exist_ok=True)
                # 若图像原始大小小于给定的image_size，则不改变图像大小
                # 否则，按比例将图像的最长边缩小到给定大小
                length_max = shape[0] if shape[0] >= shape[1] else shape[1]
                if length_max > options.image_size_max:
                    r = options.image_size_max / length_max
                    resize_shape = [int(shape[0] * r), int(shape[1] * r)]
                else:
                    resize_shape = shape
                sv_d = str((change_image(im_d, sv_pt=Path(sv_d).parent, md='L', re_sz=[resize_shape[1], resize_shape[0]])).absolute())
                im = IMT.read(sv_d, options.device).rgb2l().imt
                im_in = preprocess_data(im, need_zscore=options.need_zscore, test_mode=True)
                im_in = IMT(im_in).pad_crop(options.image_pad_size, pad_md='replicate').imt.unsqueeze(0)
                with torch.no_grad():
                    im_out = re_net(fe_net(im_in)[0])
                im_out = IMT(im_out.squeeze().unsqueeze(0)).pad_crop(resize_shape).imt
                im_out = IMA.imt2ima(im_out)
                im_out = cv2.resize(im_out.ima, (shape[1], shape[0]))
                IMA(im_out).save(sv_d.replace(".", f"_enhance_{epoch}."))
        end_time = time.time()
        test_time = end_time - start_time
        # 保存网络参数
        if (epoch+1) % options.save_freq == 0:
            for net in nets:
                net.save_network(f'iter_{epoch}')
        if validate_loss_mean['loss'] < max_validate_loss:
            epoch_best = epoch
            max_validate_loss = validate_loss_mean['loss']
            for net in nets:
                net.save_network(f'best')
        for net in nets:
            net.save_network(f'latest')
        # 显示并保存文本信息
        msg = f"第{epoch}轮训练数据集大小为{train_dataset.__len__()}，"
        msg += f"验证数据集大小为{validate_dataset.__len__()}，"
        msg += f"测试数据集大小为{test_dataset.__len__()}。\n"
        msg += f"训练耗时{train_time}秒，验证耗时{validate_time}秒，测试耗时{test_time}秒。\n"
        msg += f"训练时生成器损失为{train_loss_mean['loss']}，细节点提取器损失为{train_loss_mean['m_loss_lb']}\n"
        msg += f"生成器重建损失为{train_loss_mean['r_loss']}，细节点损失为{train_loss_mean['m_loss']}\n"
        msg += f"验证时生成器损失为{validate_loss_mean['loss']}，细节点提取器损失为{validate_loss_mean['m_loss_lb']}\n"
        msg += f"生成器重建损失为{validate_loss_mean['r_loss']}，细节点损失为{validate_loss_mean['m_loss']}\n"
        msg += f"当前最佳轮数是{epoch_best}\n"
        print(msg)
        with open(options.script_path, "a") as sf:
            sf.write(msg)
        with open(options.script_path.parent/"flag.txt", "w") as ff:
            ff.write(str(epoch))
    print("当前gpu名称：", torch.cuda.get_device_name(options.device))
    print("当前gpu容量：", torch.cuda.get_device_capability(options.device))
    print("已分配的GPU内存：", torch.cuda.memory_allocated())
    print("已缓存的GPU内存：", torch.cuda.memory_reserved())
    print("删除变量...")
    del train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset
    del data, im, lb, mk, fq, mt, im_in, lb_out, lb_mk, lb_sg, lb_sg_b, lb_fq, lb_mt, lb_ot
    del im_out, im_f, ot_f2, fq_f2, ot_l, fq_l, at_f
    del loss_m, loss_r, loss_o, loss_f, loss_s, loss, r_loss, s_loss, m_loss, m_loss_lb
    del o_l_loss, f_l_loss, o_f2_loss, f_f2_loss
    del mt_out_lb, mt_out, mt_out_8
    gc.collect()
    print("已分配的GPU内存：", torch.cuda.memory_allocated())
    print("已缓存的GPU内存：", torch.cuda.memory_reserved())
    print("释放GPU内存...")
    torch.cuda.empty_cache()
    print("已分配的GPU内存：", torch.cuda.memory_allocated())
    print("已缓存的GPU内存：", torch.cuda.memory_reserved())
    print("删除模型...")
    for net in nets:
        del net
    gc.collect()
    print("已分配的GPU内存：", torch.cuda.memory_allocated())
    print("已缓存的GPU内存：", torch.cuda.memory_reserved())
    print("释放GPU内存...")
    torch.cuda.empty_cache()
    print("已分配的GPU内存：", torch.cuda.memory_allocated())
    print("已缓存的GPU内存：", torch.cuda.memory_reserved())
    input("是否结束程序y/n：")
