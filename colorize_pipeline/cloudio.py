"""Punktwolken lesen und schreiben: PCD, PLY und ROS-2-Bags.

Alles mit numpy und der Standardbibliothek, damit die Pipeline im
System-Python laeuft.
"""
import os
import re
import sqlite3
import struct
import numpy as np


# --------------------------------------------------------------------- PCD
def _lzf_decompress(src, out_len):
    """libLZF, wie es PCD fuer DATA binary_compressed benutzt."""
    out = bytearray(out_len)
    ip = 0
    op = 0
    n = len(src)
    while ip < n and op < out_len:
        ctrl = src[ip]
        ip += 1
        if ctrl < 32:
            ln = ctrl + 1
            out[op:op + ln] = src[ip:ip + ln]
            ip += ln
            op += ln
        else:
            ln = ctrl >> 5
            ref = op - ((ctrl & 0x1f) << 8) - 1
            if ln == 7:
                ln += src[ip]
                ip += 1
            ref -= src[ip]
            ip += 1
            ln += 2
            if ref < 0:
                raise ValueError("LZF: Rueckverweis vor dem Puffer")
            for _ in range(ln):
                out[op] = out[ref]
                op += 1
                ref += 1
    return bytes(out)


_PCD_NP = {("F", 4): "<f4", ("F", 8): "<f8",
           ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
           ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8"}


def read_pcd(path):
    """PCD lesen, ascii, binary und binary_compressed. Gibt (N,3) float64."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PCD ohne DATA-Zeile")
            s = line.decode("ascii", "replace").strip()
            if not s or s.startswith("#"):
                continue
            key, _, rest = s.partition(" ")
            header[key.upper()] = rest.strip()
            if key.upper() == "DATA":
                break
        body = f.read()

    fields = header.get("FIELDS", "").split()
    sizes = [int(v) for v in header.get("SIZE", "").split()]
    types = [v.upper() for v in header.get("TYPE", "").split()]
    counts = [int(v) for v in header["COUNT"].split()] if "COUNT" in header else [1] * len(fields)
    npts = int(header.get("POINTS", 0)) or int(header["WIDTH"]) * int(header.get("HEIGHT", 1))
    mode = header["DATA"].lower()
    try:
        ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError:
        raise ValueError("PCD ohne x/y/z")

    if mode == "ascii":
        rows = []
        for line in body.decode("ascii", "replace").splitlines():
            t = line.split()
            if len(t) >= len(fields):
                rows.append((float(t[ix]), float(t[iy]), float(t[iz])))
            if len(rows) >= npts:
                break
        return np.asarray(rows, np.float64)

    dt = np.dtype([(f"f{i}", _PCD_NP[(types[i], sizes[i])], (counts[i],))
                   for i in range(len(fields))])
    if mode == "binary":
        rec = np.frombuffer(body, dt, count=min(npts, len(body) // dt.itemsize))
        return np.stack([rec[f"f{ix}"][:, 0], rec[f"f{iy}"][:, 0], rec[f"f{iz}"][:, 0]], 1).astype(np.float64)

    if mode == "binary_compressed":
        comp_len, raw_len = struct.unpack("<II", body[:8])
        raw = _lzf_decompress(body[8:8 + comp_len], raw_len)
        # spaltenweise abgelegt: erst alle x, dann alle y, ...
        cols = []
        off = 0
        for i in range(len(fields)):
            nbytes = npts * sizes[i] * counts[i]
            if i in (ix, iy, iz):
                cols.append((i, np.frombuffer(raw, _PCD_NP[(types[i], sizes[i])],
                                              count=npts * counts[i], offset=off)[::counts[i]]))
            off += nbytes
        d = dict(cols)
        return np.stack([d[ix], d[iy], d[iz]], 1).astype(np.float64)

    raise ValueError(f"PCD-Format {mode} wird nicht unterstuetzt")


# --------------------------------------------------------------------- PLY
_PLY_NP = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
           "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
           "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
           "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(path):
    """PLY lesen, ascii und beide Binaervarianten. Gibt (N,3) float64."""
    with open(path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            chunk = f.readline()
            if not chunk:
                raise ValueError("PLY ohne end_header")
            header += chunk
        fmt = "ascii"
        elements = []
        for line in header.decode("ascii", "replace").splitlines():
            t = line.split()
            if not t:
                continue
            if t[0] == "format":
                fmt = t[1]
            elif t[0] == "element":
                elements.append({"name": t[1], "count": int(t[2]), "props": []})
            elif t[0] == "property" and elements:
                if t[1] == "list":
                    elements[-1]["props"].append(("list", t[2], t[3], t[4]))
                else:
                    elements[-1]["props"].append(("scalar", t[1], t[2]))
        vertex = next(e for e in elements if e["name"] == "vertex")
        names = [p[2] for p in vertex["props"] if p[0] == "scalar"]
        if any(p[0] == "list" for p in vertex["props"]):
            raise ValueError("PLY mit Listeneigenschaften im vertex-Element")

        if fmt == "ascii":
            rows = []
            ixyz = [names.index(c) for c in "xyz"]
            for line in f.read().decode("ascii", "replace").splitlines():
                t = line.split()
                if len(t) >= len(names):
                    rows.append([float(t[i]) for i in ixyz])
                if len(rows) >= vertex["count"]:
                    break
            return np.asarray(rows, np.float64)

        end = "<" if fmt == "binary_little_endian" else ">"
        dt = np.dtype([(p[2], end + _PLY_NP[p[1]]) for p in vertex["props"]])
        rec = np.frombuffer(f.read(dt.itemsize * vertex["count"]), dt, count=vertex["count"])
        return np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)


def write_ply_rgb(path, xyz, rgb):
    """Farbige Wolke als binary_little_endian PLY schreiben."""
    n = len(xyz)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        rec.tofile(f)


# ----------------------------------------------------------------- ROS-Bag
class _Cdr:
    """Minimaler CDR-Leser mit Ausrichtung, reicht fuer PointCloud2."""

    def __init__(self, buf):
        self.b = buf
        self.little = buf[1] in (1, 3)
        self.p = 4  # Kapselungskopf

    def _align(self, n):
        rel = self.p - 4
        pad = (-rel) % n
        self.p += pad

    def u8(self):
        v = self.b[self.p]
        self.p += 1
        return v

    def _num(self, code, size):
        self._align(size)
        v = struct.unpack_from(("<" if self.little else ">") + code, self.b, self.p)[0]
        self.p += size
        return v

    def u32(self):
        return self._num("I", 4)

    def i32(self):
        return self._num("i", 4)

    def string(self):
        n = self.u32()
        s = self.b[self.p:self.p + n - 1].decode("utf-8", "replace") if n else ""
        self.p += n
        return s

    def bytes_seq(self):
        n = self.u32()
        v = self.b[self.p:self.p + n]
        self.p += n
        return v


_PF_NP = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def _parse_pointcloud2(buf):
    c = _Cdr(buf)
    c.i32()          # header.stamp.sec
    c.u32()          # header.stamp.nanosec
    c.string()       # header.frame_id
    height = c.u32()
    width = c.u32()
    nfields = c.u32()
    fields = []
    for _ in range(nfields):
        name = c.string()
        offset = c.u32()
        datatype = c.u8()
        count = c.u32()
        fields.append((name, offset, datatype, count))
    c.u8()                    # is_bigendian
    c._align(4)
    point_step = c.u32()
    c.u32()                   # row_step
    data = c.bytes_seq()

    have = {f[0]: f for f in fields}
    if not all(k in have for k in "xyz"):
        return None
    n = min(height * width, len(data) // point_step) if point_step else 0
    if n <= 0:
        return None
    raw = np.frombuffer(data, np.uint8, count=n * point_step).reshape(n, point_step)
    out = np.empty((n, 3), np.float64)
    for j, k in enumerate("xyz"):
        _, off, dtp, _ = have[k]
        dt = np.dtype(_PF_NP[dtp])
        out[:, j] = raw[:, off:off + dt.itemsize].copy().view(dt).ravel()
    return out


def _bag_dbs(path):
    """Alle .db3-Dateien eines Bags, in Aufnahmereihenfolge.

    Ein ROS-2-Bag wird ab einer gewissen Groesse auf mehrere Dateien
    verteilt, `_0.db3`, `_1.db3` und so weiter.
    """
    if os.path.isdir(path):
        cand = [f for f in os.listdir(path) if f.endswith(".db3")]
        if not cand:
            raise ValueError("Kein .db3 im Bag-Ordner")

        def order(name):
            m = re.search(r"_(\d+)\.db3$", name)
            return (int(m.group(1)) if m else 0, name)

        return [os.path.join(path, f) for f in sorted(cand, key=order)]
    return [path]


def list_bag_topics(path):
    """[(topic, typ, anzahl)] fuer alle PointCloud2-Themen eines Bags."""
    total = {}
    for db in _bag_dbs(path):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT t.name, t.type, COUNT(m.id) FROM topics t "
                "LEFT JOIN messages m ON m.topic_id = t.id GROUP BY t.id").fetchall()
        finally:
            con.close()
        for n, ty, c in rows:
            if "PointCloud2" in ty:
                prev = total.get(n, (ty, 0))
                total[n] = (ty, prev[1] + c)
    return [(n, ty, c) for n, (ty, c) in sorted(total.items())]


def read_bag_cloud(path, topic, voxel=0.05, max_messages=0, progress=None):
    """Alle PointCloud2-Nachrichten eines Themas einsammeln und ausduennen.

    Die Wolken muessen bereits in einem gemeinsamen Rahmen liegen, also etwa
    /cloud_registered oder /Laser_map aus einem SLAM-Lauf. Rohe Sensorwolken
    ohne Trajektorie ergeben keine Karte.
    """
    dbs = _bag_dbs(path)
    parts = []
    seen = 0
    found = False
    for db in dbs:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            tid = con.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
            if not tid:
                continue
            found = True
            cur = con.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp",
                              (tid[0],))
            for (blob,) in cur:
                seen += 1
                if max_messages and seen > max_messages:
                    break
                try:
                    p = _parse_pointcloud2(blob)
                except Exception:
                    p = None
                if p is not None and len(p):
                    p = p[np.isfinite(p).all(1)]
                    if len(p):
                        parts.append(p.astype(np.float32))
                if progress and seen % 50 == 0:
                    progress(f"Bag: {seen} Nachrichten, {sum(len(x) for x in parts)} Punkte")
                    # zwischendurch ausduennen, sonst laeuft der Speicher voll
                    if len(parts) > 400:
                        parts = [voxel_downsample(np.concatenate(parts).astype(np.float64),
                                                  max(voxel, 1e-3)).astype(np.float32)]
        finally:
            con.close()
    if not found:
        raise ValueError(f"Thema {topic} nicht im Bag")
    if not parts:
        raise ValueError(f"Keine brauchbaren Punkte in {topic}")
    P = np.concatenate(parts).astype(np.float64)
    if voxel and voxel > 0:
        P = voxel_downsample(P, voxel)
    return P


def voxel_downsample(P, voxel):
    """Je Wuerfel einen Punkt behalten. Reicht, um Bagwolken zu baendigen."""
    key = np.floor(P / voxel).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 1
    flat = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    _, idx = np.unique(flat, return_index=True)
    return P[np.sort(idx)]


# -------------------------------------------------------------------- Api
def read_points(path, bag_topic=None, voxel=0.05, progress=None):
    """Punktwolke aus PCD, PLY oder ROS-2-Bag lesen."""
    if os.path.isdir(path) or path.endswith(".db3"):
        if not bag_topic:
            raise ValueError("Fuer einen Bag muss ein Thema gewaehlt werden")
        return read_bag_cloud(path, bag_topic, voxel=voxel, progress=progress)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        return read_pcd(path)
    if ext == ".ply":
        return read_ply(path)
    raise ValueError(f"Unbekannte Endung {ext}")


def looks_like_bag(path):
    if os.path.isdir(path):
        return any(f.endswith(".db3") for f in os.listdir(path))
    return path.endswith(".db3")
