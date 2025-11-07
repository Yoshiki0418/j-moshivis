import math
import typing as tp
import torch
from torch import nn


def _delay_sequence(delays: tp.List[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []

    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: tp.List[int], tensor: torch.Tensor,
                      fill_value: tp.Union[int, float] = float('NaN')) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all([delay == 0 for delay in delays]):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


def _get_init_fn(input_dim: int) -> tp.Callable[[torch.Tensor], None]:
    def _init(x: torch.Tensor) -> None:
        std = 1 / math.sqrt(input_dim)
        x_orig = x
        if x.device.type == 'cpu' and x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        torch.nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if x_orig is not x:
            x_orig.data[:] = x.to(x_orig)
    return _init


def _init_layer(m: nn.Module,
                zero_bias_init: bool = True):
    if isinstance(m, nn.Linear):
        init_fn = _get_init_fn(m.in_features)
        init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = _get_init_fn(m.embedding_dim)
        init_fn(m.weight)
