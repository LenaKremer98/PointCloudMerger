"""Kleine Oberfläche für die Einfärbe-Pipeline.

Links wählt man die Punktwolke und den Ordner mit dem Mäanderflug, dann
läuft der Rest durch. Vor dem Einfärben hält die Pipeline an und zeigt die
Ausrichtung als Draufsicht, dort lässt sie sich von Hand nachziehen, falls
die Automatik danebenliegt.
"""
import os
import queue
import subprocess
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from . import cloudio, preview, register, sfm
from .pipeline import Pipeline, Abbruch


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Punktwolke mit einem Mäanderflug einfärben")
        self.geometry("1120x820")
        self.minsize(900, 640)

        self.q = queue.Queue()
        self.worker = None
        self.pipe = None
        self.cancel_flag = False
        self.tuner = None

        self.v_cloud = tk.StringVar()
        self.v_topic = tk.StringVar()
        self.v_photos = tk.StringVar()
        self.v_work = tk.StringVar()
        self.v_out = tk.StringVar()
        self.v_width = tk.IntVar(value=1600)
        self.v_voxel = tk.DoubleVar(value=0.05)
        self.v_python = tk.StringVar(value=sfm.find_python() or "")
        self.v_sparse = tk.StringVar()
        self.v_stop = tk.BooleanVar(value=True)
        self.v_state = tk.StringVar(value="bereit")

        self._build()
        self.after(100, self._pump)

    # ------------------------------------------------------------- Aufbau
    def _build(self):
        pad = dict(padx=8, pady=4)
        top = ttk.LabelFrame(self, text="Eingaben")
        top.pack(fill="x", **pad)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Punktwolke").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(top, textvariable=self.v_cloud).grid(row=0, column=1, sticky="ew", pady=4)
        bf = ttk.Frame(top)
        bf.grid(row=0, column=2, sticky="w", padx=6)
        ttk.Button(bf, text="Datei…", command=self.pick_cloud).pack(side="left")
        ttk.Button(bf, text="Bag…", command=self.pick_bag).pack(side="left", padx=(4, 0))

        self.topic_row = ttk.Frame(top)
        ttk.Label(self.topic_row, text="Thema").pack(side="left")
        self.topic_box = ttk.Combobox(self.topic_row, textvariable=self.v_topic,
                                      state="readonly", width=52)
        self.topic_box.pack(side="left", padx=6)

        ttk.Label(top, text="Mäanderflug").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(top, textvariable=self.v_photos).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(top, text="Ordner…", command=self.pick_photos)\
            .grid(row=2, column=2, sticky="w", padx=6)
        self.lbl_photos = ttk.Label(top, text="", foreground="#555")
        self.lbl_photos.grid(row=3, column=1, sticky="w")

        ttk.Label(top, text="Ausgabe").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(top, textvariable=self.v_out).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Button(top, text="Speichern…", command=self.pick_out)\
            .grid(row=4, column=2, sticky="w", padx=6)

        ttk.Label(top, text="Arbeitsordner").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(top, textvariable=self.v_work).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Button(top, text="Ordner…", command=self.pick_work)\
            .grid(row=5, column=2, sticky="w", padx=6)

        opt = ttk.LabelFrame(self, text="Optionen")
        opt.pack(fill="x", **pad)
        opt.columnconfigure(1, weight=1)
        r = ttk.Frame(opt)
        r.grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=4)
        ttk.Label(r, text="Bildbreite").pack(side="left")
        ttk.Spinbox(r, from_=800, to=4000, increment=100, width=7,
                    textvariable=self.v_width).pack(side="left", padx=(4, 16))
        ttk.Label(r, text="Voxel für Bags (m)").pack(side="left")
        ttk.Spinbox(r, from_=0.0, to=1.0, increment=0.01, width=6,
                    textvariable=self.v_voxel).pack(side="left", padx=(4, 16))
        ttk.Checkbutton(opt, text="vor dem Einfärben zur Kontrolle anhalten",
                        variable=self.v_stop).grid(row=3, column=0, columnspan=3,
                                                   sticky="w", padx=6, pady=(0, 4))

        ttk.Label(opt, text="Python mit pycolmap").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(opt, textvariable=self.v_python).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(opt, text="Datei…", command=self.pick_python)\
            .grid(row=1, column=2, sticky="w", padx=6)
        ttk.Label(opt, text="fertiges COLMAP-Modell").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(opt, textvariable=self.v_sparse).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(opt, text="sparse/0…", command=self.pick_sparse)\
            .grid(row=2, column=2, sticky="w", padx=6)

        act = ttk.Frame(self)
        act.pack(fill="x", **pad)
        self.btn_start = ttk.Button(act, text="Starten", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(act, text="Abbrechen", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_open = ttk.Button(act, text="Ergebnis öffnen", command=self.open_result,
                                   state="disabled")
        self.btn_open.pack(side="left")
        ttk.Label(act, textvariable=self.v_state, foreground="#0a6",
          anchor="e", width=40).pack(side="right")

        low = ttk.LabelFrame(self, text="Verlauf")
        low.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(low, height=14, wrap="word", font=("monospace", 9))
        sb = ttk.Scrollbar(low, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        if not self.v_python.get():
            self.log("Kein Interpreter mit pycolmap gefunden. Ohne den lässt sich "
                     "keine neue Rekonstruktion rechnen. Ein fertiges COLMAP-Modell "
                     "geht trotzdem.")

    # ------------------------------------------------------------ Auswahl
    def pick_cloud(self):
        p = filedialog.askopenfilename(
            title="Punktwolke wählen",
            filetypes=[("Punktwolken", "*.pcd *.ply"), ("Alle Dateien", "*")])
        if p:
            self.v_cloud.set(p)
            self.topic_row.grid_forget()
            self.v_topic.set("")
            self._defaults(p)

    def pick_bag(self):
        p = filedialog.askdirectory(title="ROS-2-Bag-Ordner wählen")
        if not p:
            return
        try:
            topics = cloudio.list_bag_topics(p)
        except Exception as e:
            messagebox.showerror("Bag", f"Bag nicht lesbar:\n{e}")
            return
        if not topics:
            messagebox.showwarning(
                "Bag", "Keine PointCloud2-Themen im Bag.\n\n"
                       "Die Pipeline braucht eine Wolke in einem gemeinsamen Rahmen, "
                       "also etwa /cloud_registered oder /Laser_map aus einem SLAM-Lauf.")
            return
        self.v_cloud.set(p)
        vals = [f"{n}   ({c} Nachrichten)" for n, _, c in topics]
        self.topic_box["values"] = vals
        self._topic_names = [n for n, _, _ in topics]
        self.topic_box.current(0)
        self.topic_row.grid(row=1, column=1, sticky="w", pady=(0, 4))
        self._defaults(os.path.join(p, "bag"))

    def pick_photos(self):
        p = filedialog.askdirectory(title="Ordner mit den Mäanderflug-Bildern")
        if not p:
            return
        self.v_photos.set(p)
        from . import photos as ph
        info = ph.summarize(p)
        if info["bilder"]:
            txt = f"{info['bilder']} RGB-Bilder"
            if info["thermal"]:
                txt += f", dazu {info['thermal']} Thermalbilder (werden nicht benutzt)"
            txt += f", z. B. {info['beispiel']}"
        else:
            txt = "keine Bilder gefunden"
        self.lbl_photos.configure(text=txt)

    def pick_out(self):
        p = filedialog.asksaveasfilename(title="Farbige Wolke speichern unter",
                                         defaultextension=".ply",
                                         filetypes=[("PLY", "*.ply")])
        if p:
            self.v_out.set(p)

    def pick_work(self):
        p = filedialog.askdirectory(title="Arbeitsordner")
        if p:
            self.v_work.set(p)

    def pick_python(self):
        p = filedialog.askopenfilename(title="Python-Interpreter mit pycolmap")
        if p:
            if not sfm.has_pycolmap(p):
                messagebox.showwarning("pycolmap", "In diesem Interpreter fehlt pycolmap.")
            self.v_python.set(p)

    def pick_sparse(self):
        p = filedialog.askdirectory(title="COLMAP-Modell (Ordner sparse/0)")
        if p:
            self.v_sparse.set(p)

    def _defaults(self, cloud_path):
        base = os.path.splitext(cloud_path)[0]
        if not self.v_out.get():
            self.v_out.set(base + "_farbig.ply")
        if not self.v_work.get():
            self.v_work.set(os.path.join(os.path.dirname(cloud_path), "einfaerben_arbeit"))

    # -------------------------------------------------------------- Lauf
    def log(self, msg):
        self.q.put(("log", msg))

    def _pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.txt.insert("end", payload + "\n")
                    self.txt.see("end")
                    self.v_state.set(payload[:38] + ("…" if len(payload) > 38 else ""))
                elif kind == "aligned":
                    self.open_tuner()
                elif kind == "done":
                    self._finish(payload)
                elif kind == "error":
                    self._finish(None)
                    messagebox.showerror("Fehler", payload)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def start(self):
        if not self.v_cloud.get() or not self.v_photos.get():
            messagebox.showwarning("Fehlt noch", "Punktwolke und Flugordner wählen.")
            return
        if not self.v_out.get():
            messagebox.showwarning("Fehlt noch", "Bitte eine Ausgabedatei angeben.")
            return
        topic = None
        if self.v_topic.get():
            topic = self._topic_names[self.topic_box.current()]
        self.cancel_flag = False
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_open.configure(state="disabled")
        self.txt.delete("1.0", "end")

        self.pipe = Pipeline(
            cloud_path=self.v_cloud.get(),
            photo_dir=self.v_photos.get(),
            work_dir=self.v_work.get(),
            out_path=self.v_out.get(),
            bag_topic=topic,
            voxel=float(self.v_voxel.get()),
            max_width=int(self.v_width.get()),
            sparse_dir=self.v_sparse.get() or None,
            python_exe=self.v_python.get() or None,
            log=self.log,
            cancel=lambda: self.cancel_flag,
        )
        self.worker = threading.Thread(target=self._run_until_align, daemon=True)
        self.worker.start()

    def _run_until_align(self):
        try:
            self.pipe.run(stop_before_color=True)
            if self.v_stop.get():
                self.q.put(("aligned", None))
            else:
                hit = self.pipe.colorize()
                self.q.put(("done", hit))
        except Abbruch:
            self.q.put(("log", "abgebrochen"))
            self.q.put(("error", "Der Lauf wurde abgebrochen."))
        except Exception as e:
            self.q.put(("log", traceback.format_exc()))
            self.q.put(("error", str(e)))

    def _run_color(self):
        try:
            hit = self.pipe.colorize()
            self.q.put(("done", hit))
        except Abbruch:
            self.q.put(("error", "Der Lauf wurde abgebrochen."))
        except Exception as e:
            self.q.put(("log", traceback.format_exc()))
            self.q.put(("error", str(e)))

    def stop(self):
        self.cancel_flag = True
        self.log("Abbruch angefordert…")

    def _finish(self, hit):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if hit is not None:
            self.btn_open.configure(state="normal")
            self.v_state.set(f"fertig, {hit * 100:.1f} % eingefärbt")

    def open_result(self):
        p = self.v_out.get()
        if not os.path.exists(p):
            return
        for cmd in (["code", p], ["xdg-open", p]):
            try:
                subprocess.Popen(cmd)
                return
            except FileNotFoundError:
                continue

    # ------------------------------------------------------- Nachjustieren
    def open_tuner(self):
        if self.tuner is not None and self.tuner.winfo_exists():
            self.tuner.lift()
            return
        self.tuner = Tuner(self, self.pipe, on_accept=self._accept_align)

    def _accept_align(self):
        self.pipe.save_align()
        self.worker = threading.Thread(target=self._run_color, daemon=True)
        self.worker.start()


class Tuner(tk.Toplevel):
    """Draufsicht mit vier Reglern. Grau ist die Punktwolke, orange sind die
    Fotopunkte. Wenn beides deckungsgleich liegt, stimmt die Ausrichtung."""

    def __init__(self, master, pipe, on_accept):
        super().__init__(master)
        self.title("Ausrichtung prüfen")
        self.pipe = pipe
        self.on_accept = on_accept
        self.box = None
        self._job = None

        # Zielwolke ausgedünnt, sonst wird das Zeichnen zäh
        P = pipe.points
        self.target = P[::max(1, len(P) // 250000)]

        self.yaw = float(pipe.yaw)
        self.t = np.array(pipe.t, float)

        top = ttk.Frame(self)
        top.pack(fill="both", expand=True, padx=10, pady=8)
        self.canvas = tk.Label(top, borderwidth=1, relief="solid")
        self.canvas.pack(side="left")

        right = ttk.Frame(top)
        right.pack(side="left", fill="y", padx=(12, 0))
        k = pipe.kennwerte
        ttk.Label(right, text="Automatik", font=("", 10, "bold")).pack(anchor="w")
        ttk.Label(right, text=f"Eindeutigkeit {k.get('verhaeltnis', 0):.2f}\n"
                              f"{k.get('anteil_auf_flaeche', 0) * 100:.1f} % auf der Oberfläche\n"
                              f"Median {k.get('median_abweichung', 0):.2f} m",
                  foreground="#444").pack(anchor="w", pady=(0, 10))

        self.v_yaw = tk.DoubleVar(value=round(np.degrees(self.yaw), 2))
        self.v_x = tk.DoubleVar(value=round(self.t[0], 2))
        self.v_y = tk.DoubleVar(value=round(self.t[1], 2))
        self.v_z = tk.DoubleVar(value=round(self.t[2], 2))
        self.v_info = tk.StringVar()

        # Die Regler decken den groben Bereich ab, feine Schritte macht man
        # mit den Pfeilen am Feld daneben.
        span = float(max(self.target[:, :2].max(0) - self.target[:, :2].min(0))) * 0.35
        for label, var, lo, hi, step in (
                ("Drehung (Grad)", self.v_yaw, -180.0, 180.0, 0.1),
                ("Ost / X (m)", self.v_x, self.t[0] - span, self.t[0] + span, 0.1),
                ("Nord / Y (m)", self.v_y, self.t[1] - span, self.t[1] + span, 0.1),
                ("Höhe / Z (m)", self.v_z, self.t[2] - 20, self.t[2] + 20, 0.1)):
            f = ttk.Frame(right)
            f.pack(fill="x", pady=2)
            ttk.Label(f, text=label).pack(side="left")
            ttk.Spinbox(f, from_=lo, to=hi, increment=step, width=9, textvariable=var,
                        command=self.schedule).pack(side="left")
            s = ttk.Scale(right, from_=lo, to=hi, variable=var, orient="horizontal",
                          command=lambda _e: self.schedule())
            s.pack(fill="x")

        ttk.Label(right, textvariable=self.v_info, foreground="#0a6").pack(anchor="w", pady=8)

        btn = ttk.Frame(right)
        btn.pack(anchor="w", pady=(10, 0))
        ttk.Button(btn, text="Automatik erneut", command=self.redo).pack(fill="x")
        ttk.Button(btn, text="Übernehmen und einfärben", command=self.accept).pack(fill="x", pady=4)
        ttk.Button(btn, text="Schließen", command=self.destroy).pack(fill="x")

        ttk.Label(self, foreground="#555", wraplength=880, justify="left",
                  text="Grau ist die Punktwolke von oben, orange sind die Punkte aus den "
                       "Luftbildern. Deckt sich beides, stimmt die Lage. Sitzt das Orange "
                       "verdreht oder versetzt, mit den Reglern nachziehen.")\
            .pack(fill="x", padx=10, pady=(0, 8))

        self.redraw()

    def schedule(self):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(120, self.redraw)

    def _current(self):
        return np.radians(self.v_yaw.get()), np.array(
            [self.v_x.get(), self.v_y.get(), self.v_z.get()], float)

    def redraw(self):
        self._job = None
        yaw, t = self._current()
        img, self.box = preview.render(self.target, self.pipe.model_enu, yaw, t,
                                       size=(660, 480), box=self.box)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo)
        anteil, med = register.quality(self.pipe.model_enu, self.target, yaw, t)
        self.v_info.set(f"{anteil * 100:.1f} % auf der Oberfläche, Median {med:.2f} m")

    def redo(self):
        yaw, t, k = register.auto_align(self.pipe.model_enu, self.target,
                                        progress=self.pipe.log)
        self.pipe.kennwerte.update(k)
        self.v_yaw.set(round(np.degrees(yaw), 2))
        self.v_x.set(round(t[0], 2))
        self.v_y.set(round(t[1], 2))
        self.v_z.set(round(t[2], 2))
        self.redraw()

    def accept(self):
        yaw, t = self._current()
        self.pipe.yaw, self.pipe.t = yaw, t
        anteil, med = register.quality(self.pipe.model_enu, self.target, yaw, t)
        self.pipe.kennwerte.update({"anteil_auf_flaeche": anteil, "median_abweichung": med})
        self.pipe.log(f"Ausrichtung uebernommen: {np.degrees(yaw):.2f} Grad, "
                      f"{anteil * 100:.1f} % auf der Oberflaeche")
        self.destroy()
        self.on_accept()


def main(argv=None):
    """Optional lassen sich Wolke, Flugordner und Ziel gleich mitgeben:

        python3 -m colorize_pipeline.gui wolke.pcd flugordner ergebnis.ply
    """
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    app = App()
    if len(argv) > 0:
        if cloudio.looks_like_bag(argv[0]):
            app.v_cloud.set(argv[0])
        else:
            app.v_cloud.set(argv[0])
        app._defaults(argv[0])
    if len(argv) > 1:
        app.v_photos.set(argv[1])
        from . import photos as ph
        info = ph.summarize(argv[1])
        app.lbl_photos.configure(text=f"{info['bilder']} RGB-Bilder")
    if len(argv) > 2:
        app.v_out.set(argv[2])
    if len(argv) > 3:
        app.v_work.set(argv[3])
    app.mainloop()


if __name__ == "__main__":
    main()
