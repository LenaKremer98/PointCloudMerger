"""COLMAP über pycolmap aufrufen, in einem eigenen Interpreter."""
import os
import subprocess
import sys

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_colmap_worker.py")

# Orte, an denen ein Interpreter mit pycolmap liegen könnte
_KANDIDATEN = [
    sys.executable,
    "/home/lena/_Data/26.05.21_DRZ_Avata360_Super360Pointcloud/gaussian_splat_avata360/.venv/bin/python",
    os.path.expanduser("~/.venvs/colmap/bin/python"),
]


def has_pycolmap(python_exe):
    if not python_exe or not os.path.exists(python_exe):
        return False
    try:
        r = subprocess.run([python_exe, "-c", "import pycolmap"],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def find_python():
    """Ersten Interpreter mit pycolmap finden, sonst None."""
    for p in _KANDIDATEN:
        if has_pycolmap(p):
            return p
    return None


def _run(python_exe, args, progress=None, cancel=None):
    proc = subprocess.Popen([python_exe, WORKER, *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    last = ""
    for line in proc.stdout:
        last = line.rstrip()
        if progress and last:
            progress(last)
        if cancel and cancel():
            proc.terminate()
            raise RuntimeError("abgebrochen")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"COLMAP-Schritt fehlgeschlagen: {last}")
    return last


def run_sfm(python_exe, image_dir, work_dir, progress=None, cancel=None):
    return _run(python_exe, ["sfm", image_dir, work_dir], progress, cancel)


def export_model(python_exe, sparse_dir, out_npz, progress=None, cancel=None):
    return _run(python_exe, ["export", sparse_dir, out_npz], progress, cancel)
