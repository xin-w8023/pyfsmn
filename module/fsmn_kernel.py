import enum

import torch
import torch.nn as nn


class Pad(enum.Enum):
    ZERO = 'zero padding'
    EDGE = 'edge padding'


class FSMNKernel(nn.Module):
    def __init__(self, dims, l_order, r_order, l_stride=1, r_stride=1):
        super().__init__()
        self.filter = nn.Parameter(torch.randn(l_order + r_order + 1, dims))
        self.l_order = l_order
        self.r_order = r_order
        self.l_stride = l_stride
        self.r_stride = r_stride
        self.dims = dims

    def extra_repr(self) -> str:
        return f'l_order={self.l_order}, r_order={self.r_order}, ' \
               f'l_stride={self.l_stride}, r_stride={self.r_stride}, ' \
               f'dims={self.dims}'

    def forward(self, x):
        """Apply FSMN-kernel to x.
        :param x: BxTxD
        :return: BxTxD
        """
        batch, frames, _ = x.size()
        # pad zeros, BxTxD -> Bx(l_order+T+r_order)xD
        x = torch.cat(
            (
                torch.zeros((batch, self.l_order * self.l_stride, self.dims), device=x.device),
                x,
                torch.zeros((batch, self.r_order * self.r_stride, self.dims), device=x.device)
            ),
            dim=1)

        kernel_out = []
        for frame in range(self.l_order * self.l_stride, frames + self.l_order * self.l_stride):
            l_frame = torch.sum(
                x[:, frame - self.l_order * self.l_stride:frame:self.l_stride] * self.filter[:self.l_order],
                dim=1,
                keepdim=True
            )
            c_frame = x[:, frame:frame + 1] * self.filter[self.l_order:self.l_order + 1]
            cur_frame = l_frame + c_frame
            if self.r_order > 0:
                r_frame = torch.sum(
                    x[:, frame + 1:frame + 1 + self.r_order * self.r_stride:self.r_stride] * self.filter[
                                                                                             self.l_order + 1:],
                    dim=1,
                    keepdim=True
                )
                cur_frame = cur_frame + r_frame
            kernel_out.append(cur_frame)
        kernel_out = torch.cat(kernel_out, dim=1)
        return kernel_out


class FSMNKernelParallel(nn.Module):

    def __init__(self, dims, l_order, r_order, l_stride=1, r_stride=1, padding_mode=Pad.ZERO):
        super().__init__()
        assert l_stride == r_stride == 1, f'Parallel version expected l_stride == r_stride == 1, ' \
                                          f'now get ({l_stride}, {r_stride})'
        self.filter = nn.Conv1d(in_channels=dims, out_channels=dims, kernel_size=l_order+r_order+1, stride=l_stride,
                                groups=dims, padding=0, bias=False)
        self.l_order = l_order
        self.r_order = r_order
        self.l_stride = l_stride
        self.r_stride = r_stride
        self.dims = dims
        self.pm = padding_mode

    def extra_repr(self) -> str:
        return f'l_order={self.l_order}, r_order={self.r_order}, ' \
               f'l_stride={self.l_stride}, r_stride={self.r_stride}, ' \
               f'dims={self.dims}, padding_mode={self.pm}'

    def forward(self, x):
        batch, time, dim = x.size()
        if self.pm is Pad.ZERO:
            x = torch.cat(
                (
                    torch.zeros((batch, self.l_order * self.l_stride, self.dims), device=x.device),
                    x,
                    torch.zeros((batch, self.r_order * self.r_stride, self.dims), device=x.device)
                ),
                dim=1
            )
        elif self.pm is Pad.EDGE:
            x = torch.cat(
                (
                    torch.ones((batch, self.l_order * self.l_stride, self.dims), device=x.device) * x.data[:, 0],
                    x,
                    torch.ones((batch, self.r_order * self.r_stride, self.dims), device=x.device) * x.data[:, -1]
                ),
                dim=1
            )
        else:
            raise ValueError(f'padding mode {self.pm} is not supported for now.')
        x = x.transpose(1, 2).contiguous()
        y = self.filter(x)
        return y
