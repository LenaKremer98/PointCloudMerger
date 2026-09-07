#!/usr/bin/env python3
"""Kompakter 3D-Gaussian-Splatting-Trainer direkt auf der gsplat-Bibliotheks-API.
Liest die COLMAP-Rekonstruktion (offizielles pycolmap), initialisiert Gaussians aus
der Punktwolke und trainiert mit MCMC-Strategie (begrenzte Anzahl -> 8 GB-tauglich).
Export als Standard-.ply (SuperSplat / antimatter15 kompatibel)."""
import os, sys, math, random, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pycolmap
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat import export_splats
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---------------- Config ----------------
ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.environ.get("DATA_DIR", ROOT)          # erlaubt eigenes Daten-/Arbeitsverzeichnis
SPARSE    = os.path.join(DATA, "sparse", "0")
IMG_DIR   = os.path.join(DATA, "images")
OUT_DIR   = os.environ.get("OUT_DIR", os.path.join(ROOT, "output"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 7000))
MAX_GS    = int(os.environ.get("MAX_GS", 850_000))    # harter Gaussian-Cap (8GB-Schutz)
SH_DEGREE = 3
DEVICE    = "cuda"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ---------------- COLMAP laden ----------------
print("Lade COLMAP-Rekonstruktion ...")
rec = pycolmap.Reconstruction(SPARSE)
views = []
for img in rec.images.values():
    cam = rec.cameras[img.camera_id]
    p = list(cam.params)            # SIMPLE_RADIAL: f, cx, cy, k
    f, cx, cy = p[0], p[1], p[2]
    K = torch.tensor([[f,0,cx],[0,f,cy],[0,0,1]], dtype=torch.float32)
    M = np.asarray(img.cam_from_world().matrix(), dtype=np.float32)  # (3,4) world->cam
    viewmat = torch.eye(4, dtype=torch.float32); viewmat[:3,:4] = torch.from_numpy(M)
    views.append(dict(name=img.name, K=K, viewmat=viewmat,
                      W=int(cam.width), H=int(cam.height)))
print(f"  {len(views)} registrierte Kameras")

# Bilder vorab auf CPU laden
imgs = []
for v in views:
    im = Image.open(os.path.join(IMG_DIR, v["name"])).convert("RGB")
    imgs.append(torch.from_numpy(np.asarray(im, np.float32)/255.0))   # (H,W,3)
W, H = views[0]["W"], views[0]["H"]

# Kamerazentren -> scene_scale
centers = []
for v in views:
    Rt = v["viewmat"][:3,:4]; Rm = Rt[:,:3]; t = Rt[:,3]
    centers.append((-Rm.t() @ t).numpy())
centers = np.stack(centers); ctr = np.median(centers, 0)
d = np.linalg.norm(centers-ctr, axis=1)
# fehlregistrierte Ausreisser-Kameras verwerfen (robust gg. einzelne Fehllokalisierungen)
keep = d < np.percentile(d, 95) * 4
views = [v for v,k in zip(views,keep) if k]
imgs  = [im for im,k in zip(imgs,keep) if k]
centers = centers[keep]
scene_scale = float(np.percentile(np.linalg.norm(centers-np.median(centers,0),axis=1), 90)) * 1.1
print(f"  {int(keep.sum())}/{len(keep)} Kameras behalten (Ausreisser entfernt), scene_scale = {scene_scale:.2f}")

# ---------------- Gaussians aus Punktwolke initialisieren ----------------
pts = np.stack([np.asarray(p.xyz, np.float32) for p in rec.points3D.values()])
cols = np.stack([np.asarray(p.color, np.float32)/255.0 for p in rec.points3D.values()])
# Ausreisser-Punkte verwerfen: robuste Perzentil-Grenzen (szenenunabhaengig, auch fuer Nadir)
lo = np.percentile(pts, 0.5, axis=0); hi = np.percentile(pts, 99.5, axis=0)
keep_p = np.all((pts>=lo)&(pts<=hi), axis=1)
pts, cols = pts[keep_p], cols[keep_p]
print(f"  Init aus {len(pts)} COLMAP-Punkten ({(~keep_p).sum()} Ausreisser verworfen)")
means = torch.from_numpy(pts).float()
# Skala aus mittlerem Nachbarabstand (kNN, k=4)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=4).fit(pts)
dist,_ = nn.kneighbors(pts)
mean_d = np.clip(dist[:,1:].mean(1), 1e-4, None)
scales = torch.log(torch.from_numpy(mean_d).float())[:,None].repeat(1,3)
quats  = torch.zeros(len(pts),4); quats[:,0]=1.0
opacities = torch.logit(torch.full((len(pts),), 0.1))
C0 = 0.28209479177387814
sh0 = ((torch.from_numpy(cols).float()-0.5)/C0)[:,None,:]            # (N,1,3)
shN = torch.zeros(len(pts), (SH_DEGREE+1)**2 - 1, 3)

params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(means),
    "scales": torch.nn.Parameter(scales),
    "quats": torch.nn.Parameter(quats),
    "opacities": torch.nn.Parameter(opacities),
    "sh0": torch.nn.Parameter(sh0),
    "shN": torch.nn.Parameter(shN),
}).to(DEVICE)

LRS = {"means":1.6e-4*scene_scale, "scales":5e-3, "quats":1e-3,
       "opacities":5e-2, "sh0":2.5e-3, "shN":2.5e-3/20}
optimizers = {k: torch.optim.Adam([{"params":[params[k]], "lr":lr, "name":k}],
                                  eps=1e-15, betas=(0.9,0.999)) for k,lr in LRS.items()}

# absgrad-Densification (verdichtet gezielter in Detailregionen), laenger wachsen lassen
strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(MAX_STEPS*0.4),
                           reset_every=3000, refine_every=100,
                           grow_grad2d=0.0006, prune_opa=0.005,
                           absgrad=True, verbose=True)
strat_state = strategy.initialize_state(scene_scale=scene_scale)
strategy.check_sanity(params, optimizers)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

# ---------------- Training ----------------
print(f"Training: {MAX_STEPS} Schritte, max {MAX_GS} Gaussians, absgrad")
order = list(range(len(views))); t0=time.time()
for step in range(MAX_STEPS):
    i = order[step % len(order)]
    if step % len(order) == 0: random.shuffle(order)
    v = views[i]
    gt = imgs[i].to(DEVICE)
    viewmat = v["viewmat"].to(DEVICE); K = v["K"].to(DEVICE)
    Wv, Hv = v["W"], v["H"]                     # pro Bild (Avata 1600x1600, M4T 1600x1200)
    sh_deg = min(step // 1000, SH_DEGREE)
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    renders, alphas, info = rasterization(
        means=params["means"], quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors, viewmats=viewmat[None], Ks=K[None],
        width=Wv, height=Hv, sh_degree=sh_deg, packed=False, absgrad=True,
        near_plane=0.01, far_plane=1e10, render_mode="RGB")
    pred = renders[0].clamp(0,1)                       # (H,W,3)

    strategy.step_pre_backward(params, optimizers, strat_state, step, info)
    l1 = F.l1_loss(pred, gt)
    ssim_v = ssim(pred.permute(2,0,1)[None], gt.permute(2,0,1)[None])
    loss = 0.8*l1 + 0.2*(1.0-ssim_v)
    loss.backward()
    for opt in optimizers.values(): opt.step()
    for opt in optimizers.values(): opt.zero_grad(set_to_none=True)
    strategy.step_post_backward(params, optimizers, strat_state, step, info, packed=False)
    # harter Cap: Wachstum einfrieren sobald MAX_GS erreicht (OOM-Schutz auf 8GB)
    if params["means"].shape[0] >= MAX_GS and strategy.refine_stop_iter > step:
        strategy.refine_stop_iter = step
        print(f"  [Cap] {MAX_GS} Gaussians erreicht @ step {step} -> Densification gestoppt", flush=True)

    if step % 200 == 0 or step == MAX_STEPS-1:
        n = params["means"].shape[0]
        vram = torch.cuda.max_memory_allocated()/1e9
        print(f"  step {step:5d}/{MAX_STEPS}  loss {loss.item():.4f}  "
              f"l1 {l1.item():.4f}  ssim {ssim_v.item():.3f}  "
              f"#gauss {n}  VRAM {vram:.1f}GB  {time.time()-t0:.0f}s", flush=True)

# ---------------- Export ----------------
ply = os.path.join(OUT_DIR, "avata360_splat.ply")
export_splats(
    means=params["means"].detach(), scales=params["scales"].detach(),
    quats=params["quats"].detach(), opacities=params["opacities"].detach(),
    sh0=params["sh0"].detach(), shN=params["shN"].detach(),
    format="ply", save_to=ply)
print(f"FERTIG: {params['means'].shape[0]} Gaussians -> {ply}")
print(f"Dateigroesse: {os.path.getsize(ply)/1e6:.1f} MB")
