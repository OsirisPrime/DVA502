"""Check that a gsplat build computes correct gradients on this GPU.

Compares gsplat's backward pass against central finite differences through gsplat's own forward pass.
On the RX 9070 XT (gfx1201) an unpatched wave64 build renders correctly but returns 30-90% of each
gradient, so a forward-only test is not enough.

Usage (inside the gsplat venv):  python src/gaussian_splat/rocm/verify_gsplat_rocm.py
Exit code 0 = gradients OK.
"""
import sys

import torch
from gsplat import rasterization

torch.manual_seed(0)
dev = "cuda"
N, W, H = 3000, 160, 120
P = {
    "means": torch.randn(N, 3, device=dev) * 0.6 + torch.tensor([0, 0, 4.0], device=dev),
    "scales": torch.log(torch.rand(N, 3, device=dev) * 0.06 + 0.01),
    "quats": torch.randn(N, 4, device=dev),
    "opacities": torch.randn(N, device=dev),
    "sh0": torch.randn(N, 1, 3, device=dev) * 0.5,
    "shN": torch.randn(N, 15, 3, device=dev) * 0.1,
}
P = {k: v.requires_grad_() for k, v in P.items()}
K = torch.tensor([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]], device=dev)
viewmat = torch.eye(4, device=dev)
target = torch.rand(H, W, 3, device=dev)


def render():
    img, _, _ = rasterization(
        means=P["means"], quats=P["quats"], scales=torch.exp(P["scales"]),
        opacities=torch.sigmoid(P["opacities"]), colors=torch.cat([P["sh0"], P["shN"]], 1),
        viewmats=viewmat[None], Ks=K[None], width=W, height=H, sh_degree=3, packed=False)
    return img[0]


def loss():
    return ((render() - target) ** 2).double().mean()


loss().backward()
grads = {k: v.grad.clone() for k, v in P.items()}

ok = True
# sh0: loss is quadratic in colour -> central difference is exact (up to float rounding)
# opacities: smooth -> central difference with a moderate step
for name, eps, tol in (("sh0", 1e-2, 0.02), ("opacities", 2e-2, 0.05)):
    g = grads[name].flatten()
    idx = torch.argsort(g.abs(), descending=True)[:6].tolist()
    print(f"{name}:  index |  finite diff |  gsplat grad | rel. error")
    for j in idx:
        flat = P[name].data.view(-1)
        old = flat[j].item()
        with torch.no_grad():
            flat[j] = old + eps
            lp = loss().item()
            flat[j] = old - eps
            lm = loss().item()
            flat[j] = old
        fd = (lp - lm) / (2 * eps)
        rel = abs(g[j].item() - fd) / max(abs(fd), 1e-12)
        ok &= rel < tol
        print(f"  {j:12d} | {fd:+.4e} | {g[j].item():+.4e} | {rel:.4f}{'' if rel < tol else '  <-- MISMATCH'}")

print("\nGRADIENTS OK" if ok else "\nGRADIENTS WRONG: this gsplat build must not be used for training")
sys.exit(0 if ok else 1)
