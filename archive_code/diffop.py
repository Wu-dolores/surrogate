import torch

def diffop(Fnet: torch.Tensor, coord: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute dFnet/dcoord with centered differences on non-uniform grids.

    Args:
        Fnet: (..., N) tensor
        coord: (..., N) tensor, strictly monotone along last dim (e.g., logp)
        mask: (..., N) boolean tensor, True for valid points. If None, assumes all valid.
              Masked mode assumes right-padding (invalid points are contiguous at the end).

    Returns:
        dF/dcoord with same shape as Fnet.
    """
    if mask is None:
        d = torch.zeros_like(Fnet)
        # interior: centered difference
        d[..., 1:-1] = (Fnet[..., 2:] - Fnet[..., :-2]) / (coord[..., 2:] - coord[..., :-2])
        # boundaries: one-sided difference
        d[..., 0] = (Fnet[..., 1] - Fnet[..., 0]) / (coord[..., 1] - coord[..., 0])
        d[..., -1] = (Fnet[..., -1] - Fnet[..., -2]) / (coord[..., -1] - coord[..., -2])
        return d

    # masked (ragged) case: handle per-sample valid length (assumes right-padding)
    d = torch.zeros_like(Fnet)
    valid_len = mask.long().sum(dim=-1)  # (...,)

    flat_F = Fnet.reshape(-1, Fnet.shape[-1])
    flat_c = coord.reshape(-1, coord.shape[-1])
    flat_m = mask.reshape(-1, mask.shape[-1])
    flat_d = d.reshape(-1, d.shape[-1])

    for i in range(flat_F.shape[0]):
        n = int(valid_len.reshape(-1)[i].item())
        if n < 2:
            continue

        Fi = flat_F[i, :n]
        ci = flat_c[i, :n]
        di = flat_d[i, :n]

        if n > 2:
            di[1:-1] = (Fi[2:] - Fi[:-2]) / (ci[2:] - ci[:-2])

        di[0] = (Fi[1] - Fi[0]) / (ci[1] - ci[0])
        di[-1] = (Fi[-1] - Fi[-2]) / (ci[-1] - ci[-2])

        flat_d[i, :n] = di
        flat_d[i, n:] = 0.0

    return d
