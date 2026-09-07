#!/usr/bin/env python3
"""Erzeugt align_tool.html: Splat-BG fix, Punktwolke als FG, die per Azimut-Regler
durch die echten Umrundungs-Frames scrubbt (Frame-Versatz = Orbit-Drehung).
Zusaetzlich Skalierung + Position (2D-Similarity-Rest)."""
import base64, io, os, json
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PCD_DIR = os.path.join(ROOT, "renders/_pcd_frames")
N = 643                      # Frames im vollen Loop
BASE = 161                   # aktueller Clip-Start (= +90 Grad), Azimut-Offset 0 hier
STEP = 5                     # jeder 5. Frame  (~2.8 Grad)
AZ0_RENDER = 360.0 - 360.0*60/N   # Render-Startazimut (nur fuer Info)

def b64_img(path, size, q):
    im = Image.open(path).convert("RGB").resize(size)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

bg = b64_img(os.path.join(ROOT, "renders/align/bg_splat.png"), (1280, 720), 85)

# FG-Samples ueber den ganzen Loop, sortiert nach Azimut-Offset rel. zur Basis
samples = []
for orig in range(0, N, STEP):
    az = ((orig - BASE) * 360.0 / N + 180) % 360 - 180   # [-180,180)
    samples.append((az, orig))
samples.sort()
AZ   = [round(a, 2) for a, _ in samples]
ORIG = [o for _, o in samples]
URLS = [b64_img(os.path.join(PCD_DIR, f"f{o:04d}.png"), (640, 360), 62) for _, o in samples]
k0 = min(range(len(AZ)), key=lambda i: abs(AZ[i]))   # Startindex (Azimut ~0)
print(f"{len(URLS)} FG-Frames eingebettet, Azimut {AZ[0]}..{AZ[-1]}, Start k={k0}")

HTML = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Modell-Ausrichtung</title>
<style>
 body{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif;display:flex}
 #wrap{padding:10px}
 canvas{background:#000;cursor:move;border:1px solid #333}
 #panel{padding:14px;min-width:250px}
 .row{margin:10px 0}
 label{display:block;font-size:13px;margin-bottom:3px}
 input[type=range]{width:240px}
 b{font-family:monospace;font-size:15px;color:#7fd}
 button{margin:4px 6px 4px 0;padding:7px 12px;font-size:14px;cursor:pointer}
 .hint{font-size:12px;color:#999;line-height:1.5;margin-top:14px}
</style></head><body>
<div id="wrap"><canvas id="c" width="1280" height="720"></canvas></div>
<div id="panel">
 <h3>Punktwolke über Splat ausrichten</h3>
 <div class="row"><label>Drehung (Azimut): <b id="vr">0.0</b>°</label>
   <input type="range" id="az" min="0" max="__KMAX__" step="1" value="__K0__"></div>
 <div class="row"><label>Skalierung: <b id="vs">1.000</b></label>
   <input type="range" id="scale" min="0.3" max="2.5" step="0.005" value="1"></div>
 <div class="row"><label>FG-Deckkraft: <b id="va">0.55</b></label>
   <input type="range" id="alpha" min="0" max="1" step="0.05" value="0.55"></div>
 <div class="row">dx = <b id="vdx">0</b> px&nbsp;&nbsp; dy = <b id="vdy">0</b> px</div>
 <div class="row">
   <button onclick="dl()">⬇ align_params.json speichern</button>
   <button onclick="reset()">Reset</button></div>
 <div class="hint">
   • <b>Drehung (Azimut)</b>: Slider oder <b>Q / E</b> — dreht das echte Modell<br>
   • <b>Maus ziehen</b>: verschieben &nbsp;•&nbsp; <b>Mausrad</b>: zoomen (Shift fein)<br>
   • <b>Pfeiltasten</b>: 1&nbsp;px<br>
   • Erst Azimut grob, dann Größe + Position feinjustieren.<br>
   • Decke Halle ↔ Rampe und die Container.<br>
   • Dann <b>Speichern</b> → Datei landet in Downloads.
 </div>
</div>
<script>
const FULLW=1920, FULLH=1080, CW=1280, CH=720, R=CW/FULLW;
const AZ=__AZ__, ORIG=__ORIG__, URLS=__URLS__;
const FG=URLS.map(u=>{const i=new Image(); i.src=u; i.onload=draw; return i;});
const bg=new Image(); bg.src="__BG__"; bg.onload=draw;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let S={k:__K0__, scale:1, dx:0, dy:0, alpha:0.55};
function draw(){
  ctx.clearRect(0,0,CW,CH);
  ctx.globalAlpha=1; if(bg.complete) ctx.drawImage(bg,0,0,CW,CH);
  ctx.globalAlpha=S.alpha;
  const fg=FG[S.k];
  if(fg && fg.complete){
    ctx.save();
    ctx.translate(CW/2 + S.dx*R, CH/2 + S.dy*R);
    ctx.scale(S.scale, S.scale);
    ctx.drawImage(fg, -CW/2, -CH/2, CW, CH);
    ctx.restore();
  }
  ctx.globalAlpha=1;
  vr.textContent=AZ[S.k].toFixed(1); vs.textContent=S.scale.toFixed(3);
  va.textContent=S.alpha.toFixed(2);
  vdx.textContent=Math.round(S.dx); vdy.textContent=Math.round(S.dy);
}
az.oninput=e=>{S.k=+e.target.value; draw();};
scale.oninput=e=>{S.scale=+e.target.value; draw();};
alpha.oninput=e=>{S.alpha=+e.target.value; draw();};
let drag=null;
cv.addEventListener('mousedown',e=>drag={x:e.offsetX,y:e.offsetY,dx:S.dx,dy:S.dy});
window.addEventListener('mouseup',()=>drag=null);
window.addEventListener('mousemove',e=>{ if(!drag)return; const r=cv.getBoundingClientRect();
  S.dx=drag.dx+((e.clientX-r.left)-drag.x)/R; S.dy=drag.dy+((e.clientY-r.top)-drag.y)/R; draw();});
cv.addEventListener('wheel',e=>{e.preventDefault(); const st=e.shiftKey?0.005:0.02;
  S.scale=Math.max(0.1,S.scale+(e.deltaY<0?st:-st)); scale.value=S.scale; draw();},{passive:false});
window.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')S.dx-=1; else if(e.key==='ArrowRight')S.dx+=1;
  else if(e.key==='ArrowUp')S.dy-=1; else if(e.key==='ArrowDown')S.dy+=1;
  else if(e.key==='q'||e.key==='Q'){S.k=Math.max(0,S.k-1); az.value=S.k;}
  else if(e.key==='e'||e.key==='E'){S.k=Math.min(AZ.length-1,S.k+1); az.value=S.k;}
  else if(e.key==='s')dl(); else return; e.preventDefault(); draw();});
function reset(){S={k:__K0__,scale:1,dx:0,dy:0,alpha:S.alpha}; az.value=__K0__; scale.value=1; draw();}
function dl(){
  const o={scale:+S.scale.toFixed(4), dx:Math.round(S.dx), dy:Math.round(S.dy),
           az_off:AZ[S.k], frame0:ORIG[S.k]};
  const b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='align_params.json'; a.click();
}
draw();
</script></body></html>"""
HTML = (HTML.replace("__AZ__", json.dumps(AZ))
            .replace("__ORIG__", json.dumps(ORIG))
            .replace("__URLS__", json.dumps(URLS))
            .replace("__KMAX__", str(len(AZ)-1))
            .replace("__K0__", str(k0))
            .replace("__BG__", bg))
out = os.path.join(ROOT, "align_tool.html")
open(out, "w").write(HTML)
print("geschrieben:", out, f"({len(HTML)//1024} KB)")
