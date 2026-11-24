"""Isolate the sampling and projection operations in the failing fixture."""

import platform

import torch

from kornia.geometry.transform import flow as warps


def show(label, value):
    print(label, value.shape, value.stride(), value.detach().cpu().tolist(), flush=True)


def probe(device):
    print("device", device, "torch", torch.__version__, "macOS", platform.mac_ver()[0], flush=True)
    flow = torch.tensor([100.0, -25.0], device=device).reshape(1, 2, 1, 1)
    matrix = flow.new_tensor([[[2, 0, 1], [0, 2, 1], [0, 0, 1]]])
    expected = 2 * flow
    original, valid = warps.warp_optical_flow(flow, matrix, (1, 1))
    show("warp baseline", original)
    show("valid", valid)
    known = torch.isfinite(flow).all(dim=1, keepdim=True)
    source = flow.new_tensor([[[-0.5, -0.5]]])
    vectors = torch.where(known, flow, 0)
    sample, _ = warps._sample_flow(vectors, known, source)
    show("sample", sample)
    flat = vectors.flatten(2)
    idx = torch.zeros((1, 1, 4), device=device, dtype=torch.long).expand(1, 2, 4)
    neighbors = flat.gather(2, idx)
    show("gather expanded indices", neighbors)
    show("gather contiguous indices", flat.gather(2, idx.contiguous()))
    show("gather channels packed", vectors.flatten(1).gather(1, idx.new_tensor([[0, 0, 0, 0, 1, 1, 1, 1]])))
    weights = source[..., 0].clamp(0, 0).unsqueeze(1)
    a, b, c, d = neighbors.unbind(-1)
    upper = a.lerp(b, weights)
    lower = c.lerp(d, weights)
    show("lerp upper", upper)
    show("lerp lower", lower)
    show("lerp result", upper.lerp(lower, weights))
    show("project actual samples", sample @ matrix[:, :, :2].mT)
    show("project known samples", flow[:, :, 0, 0].unsqueeze(1) @ matrix[:, :, :2].mT)
    show("project contiguous matrices", flow[:, :, 0, 0].unsqueeze(1) @ matrix[:, :, :2].mT.contiguous())

    active_sampler = warps._sample_flow

    def singleton_sample(vectors, validity, positions):
        return vectors[:, :, 0, 0].unsqueeze(1).expand(-1, positions.shape[1], -1), (
            validity[:, 0, 0, 0][:, None, None].expand(-1, positions.shape[1], -1)
        )

    warps._sample_flow = singleton_sample
    try:
        actual, mask = warps.warp_optical_flow(flow, matrix, (1, 1))
        show("warp direct singleton sampling", actual)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert bool(mask.all())
        flow = flow.requires_grad_()
        result, _ = warps.warp_optical_flow(flow, matrix, (1, 1))
        gradient = torch.autograd.grad(result.sum(), flow)[0]
        torch.testing.assert_close(gradient, torch.full_like(flow, 2), atol=0, rtol=0)
    finally:
        warps._sample_flow = active_sampler


probe("cpu")
assert torch.backends.mps.is_available(), "fork runner does not have an MPS device"
probe("mps")
