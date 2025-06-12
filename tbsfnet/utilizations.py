"""
常用函数
按照需求添加
"""
import torch


def function_for_weight_decay(start, end, fix):
    """该函数返回一个分段函数，在start和fix之间值为1，在fix和end之间由1线性衰减到0"""
    def weight_decay(x):
        y = 1.0 - max(0, x + start - fix) / float(end - fix)
        return y
    return weight_decay


def get_block_list_generator(funcs_list):
    """该函数返回funcs_list中的各个函数组成的序列"""
    def get_block_list(args_list, kwargs_list):
        """args_list和kwargs_list中的一项对应funcs_list中的一个函数的参数"""
        block = []
        for i in range(len(funcs_list)):
            block.append(funcs_list[i](*args_list[i], **kwargs_list[i]))
        return block
    return get_block_list


def average_orientation_nd(ort, blk_sz, s=(1, 1), keep_shape=False):
    """计算方向场邻域平均值。nd是no direction的缩写，表示相差pi的方向视为同一方向。
    :param ort:                 待处理方向场，n*1*h*w，弧度
    :param blk_sz:              平均方向场邻域大小，2维整数向量
    :param s:                   步长，2维整数向量
    :param keep_shape:          输出是否与输入大小相同"""
    assert len(ort.shape) == 4 and ort.shape[-3] == 1
    assert len(blk_sz) == 2 and len(s) == 2
    assert ort.shape[-1] % s[-1] == 0 and ort.shape[-2] % s[-2] == 0    # 图像高宽必须被步长整除
    conv_k = torch.ones((1, 1, blk_sz[0], blk_sz[1]), dtype=ort.dtype, device=ort.device)
    conv_k = conv_k / conv_k.numel()
    pd_sp = (blk_sz[1]-s[1]-(blk_sz[1]-s[1])//2, (blk_sz[1]-s[1])//2,
             blk_sz[0]-s[0]-(blk_sz[0]-s[0])//2, (blk_sz[0]-s[0])//2)
    cos2t = torch.cos(2*ort)
    cos2t = torch.nn.functional.pad(cos2t, pd_sp, mode='reflect')
    cos2t = torch.nn.functional.conv2d(cos2t, conv_k, stride=s)
    sin2t = torch.sin(2*ort)
    sin2t = torch.nn.functional.pad(sin2t, pd_sp, mode='reflect')
    sin2t = torch.nn.functional.conv2d(sin2t, conv_k, stride=s)
    out = 0.5*torch.atan2(sin2t, cos2t)
    if keep_shape:
        out = out.repeat_interleave(s[-1], -1).repeat_interleave(s[-2], -2)
    return out
