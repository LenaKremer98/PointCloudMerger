#!/usr/bin/env python3
"""Interaktives Ausricht-Tool: Punktwolke (FG, halbtransparent) ueber Splat (BG) legen.
- Maus ziehen  : Position verschieben
- Mausrad      : skalieren (fein mit Shift)
- Slider       : Skalierung / FG-Deckkraft
- Pfeiltasten  : Position 1px-weise
- 's' / Button : Parameter nach align_params.json speichern
- 'r'          : zuruecksetzen

Transform-Konvention (identisch zu ffmpeg scale+overlay):
  FG-Pixel (u,v) -> Canvas (scale*u + dx, scale*v + dy), Skalierung um obere linke Ecke.
"""
import argparse, json, os
import tkinter as tk
from PIL import Image, ImageTk

ap = argparse.ArgumentParser()
ap.add_argument("--bg", default="renders/align/bg_splat.png")
ap.add_argument("--fg", default="renders/align/fg_pcd.png")
ap.add_argument("--out", default="align_params.json")
ap.add_argument("--ds", type=float, default=0.5, help="Display-Skalierung")
A = ap.parse_args()

bg = Image.open(A.bg).convert("RGB")
fg = Image.open(A.fg).convert("RGB")
W, H = bg.size
DS = A.ds
DW, DH = int(W*DS), int(H*DS)
bg_d = bg.resize((DW, DH))
fg_full = fg

# Zustand (in VOLL-Aufloesung gespeichert)
st = {"scale": 1.0, "dx": 0.0, "dy": 0.0, "alpha": 0.55}
# vorhandene Parameter laden, falls vorhanden
if os.path.exists(A.out):
    try: st.update({k: float(v) for k, v in json.load(open(A.out)).items() if k in st})
    except Exception: pass

root = tk.Tk()
root.title("Modell-Ausrichtung  (Splat = BG, Punktwolke = FG)")

canvas = tk.Label(root); canvas.grid(row=0, column=0, rowspan=12)
info = tk.Label(root, font=("monospace", 11), justify="left"); info.grid(row=0, column=1, sticky="w", padx=8)

_imgref = [None]
_drag = {"x": 0, "y": 0, "dx0": 0.0, "dy0": 0.0}

def compose():
    s = st["scale"]
    # FG in Display-Aufloesung skalieren
    fw, fh = max(1, int(W*s*DS)), max(1, int(H*s*DS))
    fgr = fg_full.resize((fw, fh))
    canvas_img = Image.new("RGB", (DW, DH), (0, 0, 0))
    ox, oy = int(st["dx"]*DS), int(st["dy"]*DS)
    canvas_img.paste(fgr, (ox, oy))
    blended = Image.blend(bg_d, canvas_img, st["alpha"])
    tkimg = ImageTk.PhotoImage(blended)
    _imgref[0] = tkimg
    canvas.config(image=tkimg)
    info.config(text=(f"scale = {s:.3f}\n dx   = {st['dx']:+.1f} px\n dy   = {st['dy']:+.1f} px\n"
                      f"alpha = {st['alpha']:.2f}\n\nMaus: ziehen\nRad: zoom (Shift=fein)\n"
                      f"Pfeile: 1px\n's': speichern\n'r': reset\n\n-> {A.out}"))

def on_press(e):
    _drag.update(x=e.x, y=e.y, dx0=st["dx"], dy0=st["dy"])
def on_drag(e):
    st["dx"] = _drag["dx0"] + (e.x - _drag["x"]) / DS
    st["dy"] = _drag["dy0"] + (e.y - _drag["y"]) / DS
    compose()
def on_wheel(e):
    fine = (e.state & 0x1)  # Shift
    step = 0.005 if fine else 0.02
    d = step if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else -step
    st["scale"] = max(0.1, st["scale"] + d)
    sc.set(st["scale"]); compose()
def on_scale(v):
    st["scale"] = float(v); compose()
def on_alpha(v):
    st["alpha"] = float(v); compose()
def on_key(e):
    k = e.keysym
    if k == "Left":  st["dx"] -= 1
    elif k == "Right": st["dx"] += 1
    elif k == "Up":   st["dy"] -= 1
    elif k == "Down": st["dy"] += 1
    elif k == "s": save()
    elif k == "r": st.update(scale=1.0, dx=0.0, dy=0.0); sc.set(1.0)
    compose()

def save():
    json.dump({"scale": round(st["scale"], 4), "dx": round(st["dx"], 1), "dy": round(st["dy"], 1)},
              open(A.out, "w"), indent=2)
    info.config(text=info.cget("text") + "\n\nGESPEICHERT ✔")
    root.title(f"GESPEICHERT: scale={st['scale']:.3f} dx={st['dx']:.0f} dy={st['dy']:.0f}")

sc = tk.Scale(root, from_=0.3, to=2.5, resolution=0.005, orient="horizontal",
              label="Skalierung", length=260, command=on_scale)
sc.set(st["scale"]); sc.grid(row=1, column=1, sticky="w", padx=8)
al = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
              label="FG-Deckkraft", length=260, command=on_alpha)
al.set(st["alpha"]); al.grid(row=2, column=1, sticky="w", padx=8)
tk.Button(root, text="Speichern (s)", command=save).grid(row=3, column=1, sticky="w", padx=8, pady=4)

canvas.bind("<ButtonPress-1>", on_press)
canvas.bind("<B1-Motion>", on_drag)
root.bind("<MouseWheel>", on_wheel)   # Windows/mac
root.bind("<Button-4>", on_wheel)     # Linux scroll up
root.bind("<Button-5>", on_wheel)     # Linux scroll down
root.bind("<Key>", on_key)

compose()
root.mainloop()
