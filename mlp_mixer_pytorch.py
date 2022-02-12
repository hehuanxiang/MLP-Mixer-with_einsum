from turtle import forward
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

import functools
from collections import namedtuple
import tvm
from tvm import te
from tvm.te.operation import placeholder

import numpy as np
import torch

from torchsummary import summary

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class einsum_linear(nn.Module):
    def __init__(self, batch_size, num_patches, patch_size, channels, dim):
        super().__init__()
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.channels = channels
        self.dim = dim
    
    def forward(self, x):
        Arg = namedtuple('Arg', ['name', 'shape'])      # Arg, 实例名， 后面为属性
        z, phl = einsum('ijk,kl->ijl', 
                        Arg('x', (self.batch_size, (self.num_patches ** 2), (self.patch_size ** 2) * self.channels)),
                        Arg('w', (((self.patch_size ** 2) * self.channels), self.dim)), 
                        output_name='Z')
        
        s = te.create_schedule(z.op)
        tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

        bx, tx = s[z].split(z.op.axis[0], factor=64)
        s[z].bind(bx, te.thread_axis("blockIdx.x"))
        s[z].bind(tx, te.thread_axis("threadIdx.x"))
        fadd = tvm.build(s, [phl[0], phl[1], z], target=tgt_gpu, name="myadd")

        dev = tvm.device(tgt_gpu.kind.name, 0)
        w = np.random.random((self.num_patches, (self.patch_size ** 2) * self.channels))
        w = tvm.nd.array(w, dev)
        x = tvm.nd.array(x.cpu(), dev)
        z = tvm.nd.array(np.zeros((self.num_patches, self.dim)))
        fadd(w, x, z)
        z = np.array(z)
        z = torch.from_numpy(z)

        return z


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, batch_size, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2       # 图像块数量，图片大小为 image_size ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    Arg = namedtuple('Arg', ['name', 'shape'])      # Arg, 实例名， 后面为属性
    return nn.Sequential(
        # 1. 将图片拆成多个patches
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        
        # 2. 用一个全连接网络对所有patch进行处理，提取tokens
        # nn.Linear((patch_size ** 2) * channels, dim),
        einsum_linear(batch_size, num_patches, patch_size, channels, dim),

        # 3. 经过N个Mixer层，混合提炼特征信息
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),

        # 4. 最后一个全连接层进行类别预测
        nn.Linear(dim, num_classes)
    )


def einsum(pattern, *args, output_name=None):
    # pattern: ij,jk->ik
    input_patterns, output_pattern = pattern.split('->')
    input_patterns = input_patterns.split(",")  # ['ij', 'jk'], 'ik'

    # shapes: {'i': 128, 'j': 128, 'k': 128}
    shapes = {}
    for p, a in zip(input_patterns, args):
        for axis, size in zip(p, a.shape):
            shapes[axis] = size

    placeholders = []       # 参与计算的占位符的集合
    for a in args:
        placeholders.append(te.placeholder(a.shape, name=a.name)) #, dtype=a.dtype))
    reduce_axes = set()
    for axis in shapes:
        if axis not in output_pattern:
            reduce_axes.add(axis)
    # reduce_axes = {'j'}

    reduce_ops = {}
    for axis in reduce_axes:
        reduce_ops[axis] = te.reduce_axis((0, shapes[axis]), name=axis)

    output_shape = [shapes[axis] for axis in output_pattern]  # [128, 128]

    def func(*output_axes):
        axis_map = {}
        for op, oa in zip(output_pattern, output_axes):
            axis_map[op] = oa
        for ax in reduce_axes:
            axis_map[ax] = reduce_ops[ax]
        # axis_map: {'i': <?>, 'k': <>, 'j': <reduce_op>}
        multiplicands = []
        for i, p in zip(input_patterns, placeholders):
            p_axes = tuple(axis_map[k] for k in i)
            multiplicands.append(p[p_axes])
        m = functools.reduce(lambda x, y: x * y, multiplicands)
        for ax in reduce_axes:
            m = te.sum(m, axis=axis_map[ax])
        return m

    o = te.compute(output_shape, func, name=output_name)
    return o, placeholders

model = MLPMixer(
    batch_size = 52,
    image_size = 28,
    channels = 1,
    patch_size = 7,
    dim = 14,
    depth = 3,
    num_classes = 10
)
print(model)
summary(model, (1, 28, 28), batch_size=52, device="cpu")
