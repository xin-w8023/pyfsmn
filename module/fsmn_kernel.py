import torch
import torch.nn as nn


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
            if self.r_order > 0:
                cur_frame = torch.sum(
                    x[:, frame - self.l_order * self.l_stride:frame:self.l_stride] * self.filter[:self.l_order] +
                    x[:, frame:frame + 1] * self.filter[self.l_order:self.l_order + 1] +
                    x[:, frame + 1:frame + 1 + self.r_order * self.r_stride:self.r_stride] * self.filter[self.l_order + 1:],
                    dim=1,
                    keepdim=True
                )
            else:
                cur_frame = torch.sum(
                    x[:, frame - self.l_order * self.l_stride:frame:self.l_stride] * self.filter[:self.l_order] +
                    x[:, frame:frame + 1] * self.filter[self.l_order:self.l_order + 1],
                    dim=1,
                    keepdim=True
                )
            kernel_out.append(cur_frame)
        kernel_out = torch.cat(kernel_out, dim=1)
        return kernel_out
