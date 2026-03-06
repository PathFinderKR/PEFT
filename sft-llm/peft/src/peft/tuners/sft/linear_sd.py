import torch
import torch.nn.functional as F


def _merged_weight(weight: torch.Tensor, dv: torch.Tensor, di: torch.Tensor) -> torch.Tensor:
    flat = weight.reshape(-1).clone()
    flat.scatter_reduce_(0, di.long(), dv.to(weight.dtype), reduce="sum", include_self=True)
    return flat.reshape_as(weight)


def forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    dv: torch.Tensor,
    di: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    merged = _merged_weight(weight, dv, di)
    return F.linear(input, merged, bias)


def backward(
    output_grad: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    dv: torch.Tensor,
    di: torch.Tensor,
    need_input_grad: bool,
    need_weight_grad: bool,
    need_dv_grad: bool,
    need_bias_grad: bool,
    bias: torch.Tensor = None,
):
    # Rebuild the forward graph with grad-enabled copies and use autograd.
    input_t = input.detach().requires_grad_(need_input_grad)
    weight_t = weight.detach().requires_grad_(need_weight_grad)
    dv_t = dv.detach().requires_grad_(need_dv_grad)
    bias_t = None
    if bias is not None:
        bias_t = bias.detach().requires_grad_(need_bias_grad)

    out = forward(input_t, weight_t, dv_t, di, bias_t)
    grad_input, grad_weight, grad_dv, grad_bias = torch.autograd.grad(
        outputs=out,
        inputs=(input_t, weight_t, dv_t, bias_t),
        grad_outputs=output_grad,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return [grad_input, grad_weight, grad_dv, None, grad_bias]
