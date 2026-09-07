# Punktwolken-Viewer für VS Code

Zeigt `.pcd` und `.ply` direkt im Editor an, statt die Datei als Binärmüll zu
öffnen. Gebaut für die Wolken aus diesem Repo, funktioniert aber mit jeder
Datei in diesen Formaten.

![Höhenfarbe einer Mid360-Aufnahme](../bilder/vscode-viewer.png)

## Was drin ist

- **PCD** in `ascii`, `binary` und `binary_compressed`. Der LZF-Dekompressor
  für die komprimierte Variante steckt mit im Parser.
- **PLY** in `ascii`, `binary_little_endian` und `binary_big_endian`. Meshes
  werden gelesen, es werden nur die Vertices angezeigt, die Faces übersprungen.
- Farbe aus RGB, aus der Höhe mit Turbo-Farbskala, aus einem Skalarfeld wie
  `intensity`, oder einfarbig. Angeboten wird nur, was die Datei hergibt.
- Höhen- und Skalarskala werden auf das 2- bis 98-Perzentil gestreckt, sonst
  fressen einzelne Ausreißer den ganzen Farbbereich.
- **Höhenschnitt** über die Leiste am rechten Rand. Zwei Griffe spannen die
  sichtbare Schicht auf, wie in einem Schichtmodell. Damit lässt sich das Dach
  abnehmen und in ein Gebäude hineinschauen. Die Höhen an den Griffen stehen in
  Metern in den Originalkoordinaten der Datei.
- **Messen** von Punkt zu Punkt. Zwei Klicks setzen die Marken, dazwischen
  liegt eine dünne Linie, und der Abstand steht in Metern direkt an dieser
  Linie. Unten links stehen beide Punkte in Originalkoordinaten, dazu der
  Abstand, der waagerechte Anteil, der Höhenunterschied und die Differenz je
  Achse.
- Achsenkreuz am Ursprung der Originaldaten, Z-up oder Y-up umschaltbar,
  heller oder dunkler Hintergrund.
- Statuszeile mit Punktzahl, Format, Bounds und Schwerpunkt.

Keine Abhängigkeiten. Der Renderer ist reines WebGL, es wird nichts aus dem
Netz nachgeladen und es gibt keinen Build-Schritt.

## Bedienung

| Eingabe | Wirkung |
|---|---|
| linke Maustaste ziehen | drehen |
| rechte Maustaste oder Umschalt und ziehen | verschieben |
| Mausrad | zoomen |
| Griff an der rechten Leiste ziehen | obere oder untere Schnittebene setzen |
| auf die Leiste klicken | den näher liegenden Griff dorthin holen |
| Mausrad über der Leiste | die ganze Schicht nach oben oder unten schieben |
| `alles` | Schnitt aufheben |
| `Messen` oder `m` | Messen ein- und ausschalten |
| Klick bei aktivem Messen | Punkt A, dann Punkt B setzen, ein dritter Klick fängt neu an |
| `Esc` | Messung verwerfen |
| `r` | Ansicht zurücksetzen |

Beim Messen wird der Punkt genommen, welcher dem Klick am nächsten liegt und
dabei der Kamera am nächsten steht. Gesucht wird nur unter den sichtbaren
Punkten, ein aktiver Höhenschnitt schließt also alles Weggeschnittene aus.
Gemessen wird in den Einheiten der Datei, bei den Wolken aus diesem Repo sind
das Meter.

## Installieren

```bash
ln -sfn "$PWD/vscode-pointcloud-viewer" \
        ~/.vscode/extensions/lenakremer.pointcloud-viewer-0.1.0
```

Danach VS Code neu starten. Beim ersten Öffnen fragt VS Code, welcher Editor
zuständig sein soll, dort einmal **Keep Punktwolken-Viewer** wählen. Wer die
Frage überspringen will, trägt das direkt in die `settings.json` ein:

```json
"workbench.editorAssociations": {
  "*.pcd": "pointcloudViewer.cloud",
  "*.ply": "pointcloudViewer.cloud"
}
```

Die Originaldatei lässt sich weiterhin über `Reopen Editor With…` als Text
öffnen.

## Einstellungen

| Schlüssel | Vorgabe | Bedeutung |
|---|---|---|
| `pointcloudViewer.pointSize` | `2` | Punktgröße in Pixeln beim Öffnen |
| `pointcloudViewer.background` | `dunkel` | Hintergrund der 3D-Ansicht |
| `pointcloudViewer.upAxis` | `z` | Welche Achse nach oben zeigt |

## Grenzen

Die Datei wird komplett in den Speicher gelesen und als ein Puffer zur
Grafikkarte geschoben. Bei den 1,03 Mio Punkten aus diesem Repo dauert das
Zerlegen etwa 140 ms. Sehr viel größere Wolken, also deutlich über zehn
Millionen Punkte, sind nicht vorgesehen. Es wird nicht ausgedünnt und es gibt
keinen Level-of-Detail.

Normalen werden gelesen, aber nicht dargestellt, es gibt keine Beleuchtung.
