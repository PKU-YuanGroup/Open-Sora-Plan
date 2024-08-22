import torch
import torch.nn.functional as F


def causal_pad(x):
    x = torch.concatenate((x[:, :, :1, :, :], x), dim=2)
    return x

def haar_wavelet_transform_3d(x):
    if len(x.shape) != 5:
        raise ValueError(
            "Input tensor must be 5D (batch_size, channels, depth, height, width)"
        )
    dtype = x.dtype
    device = x.device
    h = (
        torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=dtype, device=x.device)
        * 0.3536
    )
    g = (
        torch.tensor(
            [[[1, -1], [1, -1]], [[1, -1], [1, -1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    hh = (
        torch.tensor(
            [[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    gh = (
        torch.tensor(
            [[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    h_v = (
        torch.tensor(
            [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    g_v = (
        torch.tensor(
            [[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    hh_v = (
        torch.tensor(
            [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    gh_v = (
        torch.tensor(
            [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=dtype, device=x.device
        )
        * 0.3536
    )
    h = h.view(1, 1, 2, 2, 2)
    g = g.view(1, 1, 2, 2, 2)
    hh = hh.view(1, 1, 2, 2, 2)
    gh = gh.view(1, 1, 2, 2, 2)
    h_v = h_v.view(1, 1, 2, 2, 2)
    g_v = g_v.view(1, 1, 2, 2, 2)
    hh_v = hh_v.view(1, 1, 2, 2, 2)
    gh_v = gh_v.view(1, 1, 2, 2, 2)
    b, c, t, height, width = x.shape
    x = causal_pad(x)
    x = x.reshape(b * c, 1, t + 1, height, width)
    low_low_low = F.conv3d(x, h, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    low_low_high = F.conv3d(x, g, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    low_high_low = F.conv3d(x, hh, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    low_high_high = F.conv3d(x, gh, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    high_low_low = F.conv3d(x, h_v, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    high_low_high = F.conv3d(x, g_v, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    high_high_low = F.conv3d(x, hh_v, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )
    high_high_high = F.conv3d(x, gh_v, stride=2).reshape(
        b, c, (t - 1) // 2 + 1, height // 2, width // 2
    )

    return torch.concat(
        [
            low_low_low,
            low_low_high,
            low_high_low,
            low_high_high,
            high_low_low,
            high_low_high,
            high_high_low,
            high_high_high,
        ],
        dim=1,
    )


def inverse_haar_wavelet_transform_3d(coeffs):
    (
        low_low_low,
        low_low_high,
        low_high_low,
        low_high_high,
        high_low_low,
        high_low_high,
        high_high_low,
        high_high_high,
    ) = coeffs.chunk(8, dim=1)
    dtype = low_low_low.dtype
    device = low_low_low.device
    b, c, t_half, height_half, width_half = low_low_low.shape
    t, height, width = (t_half - 1) * 2 + 1, height_half * 2, width_half * 2
    h = (
        torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=dtype, device=device)
        * 0.3536
    )
    g = (
        torch.tensor(
            [[[1, -1], [1, -1]], [[1, -1], [1, -1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    hh = (
        torch.tensor(
            [[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    gh = (
        torch.tensor(
            [[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    h_v = (
        torch.tensor(
            [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    g_v = (
        torch.tensor(
            [[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    hh_v = (
        torch.tensor(
            [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    gh_v = (
        torch.tensor(
            [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=dtype, device=device
        )
        * 0.3536
    )
    h = h.view(1, 1, 2, 2, 2)
    g = g.view(1, 1, 2, 2, 2)
    hh = hh.view(1, 1, 2, 2, 2)
    gh = gh.view(1, 1, 2, 2, 2)
    h_v = h_v.view(1, 1, 2, 2, 2)
    g_v = g_v.view(1, 1, 2, 2, 2)
    hh_v = hh_v.view(1, 1, 2, 2, 2)
    gh_v = gh_v.view(1, 1, 2, 2, 2)
    low_low_low = F.conv_transpose3d(
        low_low_low.reshape(b * c, 1, t_half, height_half, width_half), h, stride=2
    )
    low_low_high = F.conv_transpose3d(
        low_low_high.reshape(b * c, 1, t_half, height_half, width_half), g, stride=2
    )
    low_high_low = F.conv_transpose3d(
        low_high_low.reshape(b * c, 1, t_half, height_half, width_half), hh, stride=2
    )
    low_high_high = F.conv_transpose3d(
        low_high_high.reshape(b * c, 1, t_half, height_half, width_half), gh, stride=2
    )
    high_low_low = F.conv_transpose3d(
        high_low_low.reshape(b * c, 1, t_half, height_half, width_half), h_v, stride=2
    )
    high_low_high = F.conv_transpose3d(
        high_low_high.reshape(b * c, 1, t_half, height_half, width_half), g_v, stride=2
    )
    high_high_low = F.conv_transpose3d(
        high_high_low.reshape(b * c, 1, t_half, height_half, width_half), hh_v, stride=2
    )
    high_high_high = F.conv_transpose3d(
        high_high_high.reshape(b * c, 1, t_half, height_half, width_half),
        gh_v,
        stride=2,
    )
    reconstructed = (
        low_low_low[:, :, 1:]
        + low_low_high[:, :, 1:]
        + low_high_low[:, :, 1:]
        + low_high_high[:, :, 1:]
        + high_low_low[:, :, 1:]
        + high_low_high[:, :, 1:]
        + high_high_low[:, :, 1:]
        + high_high_high[:, :, 1:]
    )
    reconstructed = reconstructed.view(b, c, t, height, width)
    return reconstructed

def haar_wavelet_transform_2d(x):
    assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, "Height and width must be even"
    x_even_rows = x[:, :, 0::2, :]
    x_odd_rows = x[:, :, 1::2, :]
    L = (x_even_rows + x_odd_rows) / 2
    H = (x_even_rows - x_odd_rows) / 2
    LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) / 2
    LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) / 2
    HL = (H[:, :, :, 0::2] + H[:, :, :, 1::2]) / 2
    HH = (H[:, :, :, 0::2] - H[:, :, :, 1::2]) / 2
    return torch.concat([LL, LH, HL, HH], dim=1)

def haar_wavelet_transform_2d_new(x):
    dtype = x.dtype
    device = x.device
    
    aa = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device) / 2
    ad = torch.tensor([[1, 1], [-1, -1]], dtype=dtype, device=device) / 2
    da = torch.tensor([[1, -1], [1, -1]], dtype=dtype, device=device) / 2
    dd = torch.tensor([[1, -1], [-1, 1]], dtype=dtype, device=device) / 2
    aa = aa.view(1, 1, 2, 2)
    ad = ad.view(1, 1, 2, 2)
    da = da.view(1, 1, 2, 2)
    dd = dd.view(1, 1, 2, 2)
    b, c, h, w = x.shape
    x = x.reshape(b*c, 1, h, w)
    low_low = F.conv2d(x, aa, stride=2).reshape(b, c, h // 2, w // 2)
    low_high = F.conv2d(x, ad, stride=2).reshape(b, c, h // 2, w // 2)
    high_low = F.conv2d(x, da, stride=2).reshape(b, c, h // 2, w // 2)
    high_high = F.conv2d(x, dd, stride=2).reshape(b, c, h // 2, w // 2)
    coeffs = torch.cat([low_low, low_high, high_low, high_high], dim=1)
    
    return coeffs


def inverse_haar_wavelet_transform_2d_new(coeffs):
    dtype = coeffs.dtype
    device = coeffs.device
    
    low_low, low_high, high_low, high_high = coeffs.chunk(4, dim=1)
    b, c, height_half, width_half = low_low.shape
    
    aa = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device) / 2
    ad = torch.tensor([[1, 1], [-1, -1]], dtype=dtype, device=device) / 2
    da = torch.tensor([[1, -1], [1, -1]], dtype=dtype, device=device) / 2
    dd = torch.tensor([[1, -1], [-1, 1]], dtype=dtype, device=device) / 2
    aa = aa.view(1, 1, 2, 2)
    ad = ad.view(1, 1, 2, 2)
    da = da.view(1, 1, 2, 2)
    dd = dd.view(1, 1, 2, 2)
    
    height = height_half * 2
    width = width_half * 2

    low_low = F.conv_transpose2d(low_low.reshape(b*c, 1, height_half, width_half), aa, stride=2)
    low_high = F.conv_transpose2d(low_high.reshape(b*c, 1, height_half, width_half), ad, stride=2)
    high_low = F.conv_transpose2d(high_low.reshape(b*c, 1, height_half, width_half), da, stride=2)
    high_high = F.conv_transpose2d(high_high.reshape(b*c, 1, height_half, width_half), dd, stride=2)
    
    return (
		low_low + low_high + high_low + high_high
	).reshape(b, c, height, width)
    