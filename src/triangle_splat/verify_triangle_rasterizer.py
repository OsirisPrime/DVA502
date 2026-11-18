"""Check that the compiled triangle rasterizer is correct on this GPU.

The gsplat ROCm port rendered correctly but returned wrong gradients before it was patched, so both the image
and the gradients are checked here, against a slow but independent pure-PyTorch re-implementation of the
official rasterizer (`reference` below):

1. Forward: the rendered image must match the reference.
2. Backward: gradients of all parameters must match autograd through the reference.
3. Colour, opacity and sigma gradients are also compared with finite differences.

Note on vertex positions: the official backward pass is not the exact derivative. It leaves out the
pixel-scale factor (W/2, H/2) of the NDC->pixel conversion and treats the incircle normalisation as a
constant. That is how the method is defined and trained (Adam is insensitive to the scale), so the reference
reproduces these two choices (`official_grad=True`) instead of comparing vertex gradients with finite
differences. The last line reports how far the official vertex gradient is from the exact one.

Usage (inside the triangle-splat venv):  python src/triangle_splat/verify_triangle_rasterizer.py
Exit code 0 = build OK.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("TRIANGLE_SPLAT_SRC", Path.home() / "triangle-splat-rocm" / "src"))
sys.path[:0] = [str(HERE / "shims"), str(SRC)]

from scene.cameras import MiniCam  # noqa: E402
from scene.triangle_model import TriangleModel  # noqa: E402
from triangle_renderer import render  # noqa: E402
from utils.graphics_utils import BasicPointCloud, getProjectionMatrix  # noqa: E402

C0 = 0.28209479177387814
TILE = 16


def reference(model, cam, official_grad):
    """Pure-PyTorch version of the official forward pass (colour image only, SH degree 0, black background)."""
    W, H = cam.image_width, cam.image_height
    pts = model._triangles_points                                   # (N,3,3) world-space vertices
    opac = torch.sigmoid(model._opacity).squeeze(-1)
    sigma = (0.01 + torch.exp(model._sigma)).squeeze(-1)
    col = (model._features_dc[:, 0, :] * C0 + 0.5).clamp(min=0)
    V, M = cam.world_view_transform, cam.full_proj_transform       # row-vector convention: p_hom = p @ M
    dev = pts.device
    size_wh = torch.tensor([W, H], device=dev, dtype=pts.dtype)
    one = torch.ones_like(pts[..., :1])

    # projection of the vertices to pixel coordinates
    hom = torch.cat([pts, one], -1) @ M
    ndc = hom[..., :2] / (hom[..., 3:4] + 1e-7)
    pix = ((ndc + 1) * size_wh - 1) * 0.5
    if official_grad:  # official backward: d(pixel)/d(ndc) = 1 instead of W/2, H/2
        pix = ndc + (pix - ndc).detach()

    # centroid: depth for sorting, culling
    cen = torch.cat([pts.mean(1), one[:, 0]], -1)
    p_view = (cen @ V)[:, :3]
    depth = p_view[:, 2]
    chom = cen @ M
    cpix = ((chom[:, :2] / (chom[:, 3:4] + 1e-7) + 1) * size_wh - 1) * 0.5
    dist_pts = (pix - cpix[:, None]).norm(dim=-1).max(1).values
    cross = torch.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0], dim=-1) @ V[:3, :3]
    cross = cross / cross.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    cos = (cross * p_view / p_view.norm(dim=-1, keepdim=True).clamp(min=1e-4)).sum(-1)

    # edge lines n.p + offset = 0 with n.p + offset < 0 inside; incircle distance normalises the window
    A, B, Cv = pix[:, 0], pix[:, 1], pix[:, 2]
    a, b, c = (B - Cv).norm(dim=-1), (A - Cv).norm(dim=-1), (A - B).norm(dim=-1)
    incenter = (a[:, None] * A + b[:, None] * B + c[:, None] * Cv) / (a + b + c)[:, None]
    p1, p2 = pix, pix.roll(-1, dims=1)
    n = torch.stack([p2[..., 1] - p1[..., 1], -(p2[..., 0] - p1[..., 0])], -1)
    n = n / n.norm(dim=-1, keepdim=True)
    off = -(n * p1).sum(-1)
    d_in = (n * incenter[:, None]).sum(-1) + off
    flip = torch.where(d_in > 0, -1.0, 1.0)
    n, off, d_in = n * flip[..., None], off * flip, d_in * flip
    phi_c = 1 / d_in[:, 2]
    if official_grad:  # official backward: incircle normalisation is a constant
        phi_c = phi_c.detach()

    valid = ((opac >= 0.01) & (depth > 0.2) & (cos.abs() >= 0.001)
             & (dist_pts >= 1) & (dist_pts <= 1600) & (d_in[:, 2] <= -1))

    # tile rectangle: bounding box of the inner triangle where alpha >= 0.01
    with torch.no_grad():
        gx, gy = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
        o2 = off - (d_in[:, 0] * (0.01 / opac) ** (1 / sigma))[:, None]
        rmin = torch.tensor([gx, gy], device=dev).repeat(len(pts), 1).float()
        rmax = torch.zeros_like(rmin)
        lim = torch.tensor([gx, gy], device=dev).float()
        for i, j in ((1, 0), (2, 1), (0, 2)):
            det = n[:, i, 0] * n[:, j, 1] - n[:, i, 1] * n[:, j, 0]
            ok = (det.abs() >= 1e-3)[:, None]
            xy = torch.stack([-(o2[:, i] * n[:, j, 1] - o2[:, j] * n[:, i, 1]) / det,
                              -(o2[:, j] * n[:, i, 0] - o2[:, i] * n[:, j, 0]) / det], -1)
            lo = torch.minimum((xy / TILE).clamp(min=0).trunc(), lim)
            hi = torch.minimum(((xy + TILE - 1) / TILE).clamp(min=0).trunc(), lim)
            rmin = torch.where(ok, torch.minimum(rmin, lo), rmin)
            rmax = torch.where(ok, torch.maximum(rmax, hi), rmax)

    keep = valid.nonzero().squeeze(1)
    keep = keep[torch.argsort(depth[keep], stable=True)]            # front to back
    n, off, phi_c, opac, sigma, col = n[keep], off[keep], phi_c[keep], opac[keep], sigma[keep], col[keep]
    rmin, rmax = rmin[keep], rmax[keep]

    ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    px, py = xs.reshape(-1).float(), ys.reshape(-1).float()        # pixel centres are integer coordinates
    tx, ty = (px / TILE).floor(), (py / TILE).floor()
    covered = ((tx >= rmin[:, :1]) & (tx < rmax[:, :1]) & (ty >= rmin[:, 1:]) & (ty < rmax[:, 1:]))

    d = n[:, :, 0, None] * px + n[:, :, 1, None] * py + off[:, :, None]   # (N,3,P) signed edge distances
    inside = (d <= 0).all(1)
    phi = d.max(1).values * phi_c[:, None]                          # 1 at the incenter, 0 on the edges
    pos = inside & (phi > 0)
    Cx = torch.where(pos, torch.pow(torch.where(pos, phi, 1.0), sigma[:, None]), 0.0)
    a_raw = opac[:, None] * Cx
    alpha = a_raw + (a_raw.clamp(max=0.99) - a_raw).detach()        # official backward ignores the 0.99 clamp
    alpha = torch.where(inside & covered & (alpha >= 1.0 / 255), alpha, 0.0)

    T_after = torch.cumprod(1 - alpha, 0)
    T_before = torch.cat([torch.ones_like(T_after[:1]), T_after[:-1]], 0)
    weights = alpha * T_before * (T_after >= 1e-4)                  # stop once the pixel is saturated
    return (weights.T @ col).reshape(H, W, 3).permute(2, 0, 1)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, W, H, focal = 1500, 240, 180, 225.0
    fovx, fovy = 2 * np.arctan(W / (2 * focal)), 2 * np.arctan(H / (2 * focal))
    view = torch.eye(4, device="cuda")
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    cam = MiniCam(W, H, fovy, fovx, 0.01, 100.0, view, view @ proj)

    pts = rng.normal(scale=0.6, size=(N, 3)) + np.array([0, 0, 4.0])
    pcd = BasicPointCloud(points=pts, colors=rng.uniform(0.1, 0.9, (N, 3)), normals=np.zeros((N, 3)))
    model = TriangleModel(sh_degree=3)
    model.create_from_pcd(pcd, 1.0, opacity=0.28, init_size=2.23, nb_points=3, set_sigma=1.16, no_dome=True)
    with torch.no_grad():  # vary opacity and sharpness so every code path is exercised
        model._opacity += torch.from_numpy(rng.normal(scale=1.0, size=(N, 1))).float().cuda()
        model._sigma += torch.from_numpy(rng.normal(scale=0.3, size=(N, 1))).float().cuda()
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=1.0, debug=False)
    bg = torch.zeros(3, device="cuda")
    target = torch.from_numpy(rng.uniform(size=(3, H, W))).float().cuda()
    params = {"features_dc": model._features_dc, "opacity": model._opacity, "sigma": model._sigma,
              "triangles_points": model._triangles_points}

    def build_image():
        return render(cam, model, pipe, bg)["render"]

    def loss_of(image):
        return ((image - target) ** 2).double().mean()

    ok = True

    # ---- 1. forward ----
    with torch.no_grad():
        img, ref = build_image(), reference(model, cam, official_grad=True)
    mse = ((img - ref) ** 2).mean().item()
    psnr = -10 * np.log10(max(mse, 1e-20))
    fwd_ok = psnr > 60
    ok &= fwd_ok
    print(f"{N} triangles, {W}x{H}, image mean {img.mean().item():.3f}, {(img.sum(0) > 0).float().mean().item():.0%} "
          f"of pixels covered")
    print(f"forward : rasterizer vs reference  PSNR {psnr:.1f} dB, max |diff| {(img - ref).abs().max().item():.2e}"
          f"{'' if fwd_ok else '  <-- MISMATCH'}")

    # ---- 2. backward vs autograd through the reference ----
    loss_of(build_image()).backward()
    grads = {k: p.grad.clone() for k, p in params.items()}
    ref_grads = dict(zip(params, torch.autograd.grad(loss_of(reference(model, cam, True)), list(params.values()))))
    for k in params:
        g, r = grads[k].flatten().double(), ref_grads[k].flatten().double()
        cos = F.cosine_similarity(g, r, dim=0).item()
        ratio = (g.norm() / r.norm()).item()
        good = cos > 0.999 and abs(ratio - 1) < 0.01
        ok &= good
        print(f"backward: {k:17s} cosine {cos:.5f}  norm ratio {ratio:.4f}{'' if good else '  <-- MISMATCH'}")

    # ---- 3. finite differences where the official gradient is the exact derivative ----
    for name, eps, tol in (("features_dc", 1e-2, 0.02), ("opacity", 2e-2, 0.05), ("sigma", 2e-2, 0.05)):
        g = grads[name].flatten()
        worst = 0.0
        for j in torch.argsort(g.abs(), descending=True)[:6].tolist():
            flat = params[name].data.view(-1)
            old = flat[j].item()
            with torch.no_grad():
                flat[j] = old + eps
                lp = loss_of(build_image()).item()
                flat[j] = old - eps
                lm = loss_of(build_image()).item()
                flat[j] = old
            fd = (lp - lm) / (2 * eps)
            worst = max(worst, abs(g[j].item() - fd) / max(abs(fd), 1e-12))
        ok &= worst < tol
        print(f"finite differences: {name:12s} worst rel. error of 6 largest {worst:.4f}"
              f"{'' if worst < tol else '  <-- MISMATCH'}")

    exact = torch.autograd.grad(loss_of(reference(model, cam, False)), model._triangles_points)[0]
    cos = F.cosine_similarity(grads["triangles_points"].flatten().double(), exact.flatten().double(), dim=0).item()
    print(f"(info) official vertex gradient vs exact derivative: cosine {cos:.3f}, "
          f"norm ratio {(grads['triangles_points'].norm() / exact.norm()).item():.4f}")

    print("\nRASTERIZER OK" if ok else "\nRASTERIZER WRONG: this build must not be used for training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
