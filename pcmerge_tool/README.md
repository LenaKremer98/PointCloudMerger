# PointCloud Merge & Edit Tool

Werkzeug, mit welchem mehrere Punktwolken zu einer einzigen zusammengesetzt
werden. Es zeigt die Wolken in einem 3D-Fenster, richtet sie gegeneinander aus
und schreibt das Ergebnis als PCD.

Damit ist `PCD-DRZ_20-05-26/merged_FinlaDRZ.pcd` entstanden, also die
Ausgangskarte der Einfärbung im Hauptordner. Die Super-Drohne hat am 20.05.2026
drei Teilkarten aufgenommen, `Super_Mid360_2026-05-20_15-10-55.pcd`,
`_15-24-25.pcd` und `_15-27-57.pcd`. Zwischen den Aufnahmen wurde FAST-LIO neu
gestartet, weshalb jede Teilkarte ihren eigenen Ursprung hat und keine
gemeinsame Trajektorie existiert. Zusammengeschoben wurden sie hier von Hand.

```bash
./run.sh
```

Das Skript sucht ROS 2 Humble, setzt `DISPLAY` und startet die Oberfläche. Ohne
ROS geht es auch, dann fehlt nur das Laden aus einem Bag.

Links liegt die 3D-Ansicht, rechts die Bedienung. Der Aufbau ist von oben nach
unten der Arbeitsablauf, also laden, ausrichten, säubern, speichern.

## Slots

Jede Wolke liegt in einem eigenen Slot mit eigener Farbe, Punktgröße und einem
Haken für die Sichtbarkeit. Über `＋ Add Slot` kommen weitere dazu, zwei sind
schon da.

Geladen wird entweder eine PCD-Datei oder ein ROS-2-Bag. Beim Bag werden die
enthaltenen PointCloud2-Topics aufgelistet, danach wird gefragt, ob nur die
letzte Nachricht genommen wird oder alle aneinandergehängt werden. Die letzte
Nachricht ist bei fertigen Karten wie `/Laser_map` richtig, das Aneinanderhängen
bei einzelnen Scans. Beim Aneinanderhängen wird ausgedünnt, sonst läuft der
Speicher voll. Ein Intensitätsfeld wird mitgelesen, wenn es im Header steht.

## Ausrichten

Der Kasten `Align — Source → Target` bewegt immer den Source-Slot auf den
Target-Slot zu, der Target bleibt liegen.

Von Hand gibt es sechs Felder für X, Y, Z, Roll, Pitch und Yaw. `Apply manual
transform to Source` rechnet sie auf die Wolke.

`★ Auto-Align (Global + ICP)` probiert eine ganze Reihe Startlagen durch, die
Identität, den Versatz der Schwerpunkte, acht Drehungen um die Hochachse und
zehn Läufe Fast Global Registration über FPFH-Merkmale. Jede Startlage wird
danach mit ICP in drei Stufen von grob nach fein verfeinert, Punkt zu Ebene.
Gewertet wird nach Trefferquote abzüglich des Restfehlers, die beste Lage
gewinnt. `ICP-only refine` lässt die Suche weg und verfeinert nur die aktuelle
Lage, was nach einer Handjustage reicht.

Wenn beides nicht greift, weil sich die Wolken zu wenig überlappen, hilft
`✋ Manual Pick-Pair Align`. Dabei werden abwechselnd zusammengehörende Punkte
in beiden Wolken angeklickt, ab drei Paaren wird daraus per Umeyama eine starre
Transformation gerechnet. Der Haken `Auto-Align after each load` schaltet die
Automatik nach jedem Laden scharf.

## Editieren

Was nicht in die Karte gehört, also Ausreißer, Personen, Fahrzeuge oder der
Nebel über dem Boden, lässt sich vorher wegnehmen. Der aktive Slot ist der, auf
welchen die Werkzeuge wirken.

| Modus | Taste | Wirkung |
|---|---|---|
| Pan/Rotate | Esc | nur Kamera, nichts wird ausgewählt |
| Box Select | B | Rechteck aufziehen |
| Polygon Lasso | P | Umriss klicken, rechte Maustaste schließt ihn |
| Sphere Brush | S | Kugel am Cursor, Radius über das Mausrad |
| Pick Point | K | einzelner Punkt |
| Add Grid | G | ebenes Punktraster einfügen, für Boden oder Wand |

Ausgewählte Punkte leuchten rot und verschwinden mit `Entf`. Dazu kommt eine
Prismenauswahl, bei welcher ein Polygon in der Draufsicht gezogen und über eine
Basishöhe und eine Höhe zu einem Körper aufgezogen wird. Das Prisma lässt sich
danach noch verschieben, alles darin wird gelöscht.

Die Leiste am rechten Rand des Fensters schneidet die Anzeige in der Höhe, mit
zwei Griffen. Was ausgeblendet ist, wird auch nicht ausgewählt, deshalb geht
über `Select All Visible (Z-Slice)` eine ganze Schicht auf einmal. Jeder Schritt
lässt sich mit Strg+Z zurücknehmen.

## Speichern

`Save Active Slot as PCD…` schreibt nur den aktiven Slot. `💾 SAVE MERGED PCD`
schreibt alle sichtbaren Slots als eine Datei, jeder Slot mit seiner aktuellen
Transformation. `⤵ Merge all visible → first slot` macht dasselbe innerhalb der
Anwendung, ohne zu speichern, damit an der zusammengesetzten Wolke weiter
gearbeitet werden kann.

Geschrieben wird binäres PCD, mit Intensität, sofern alle beteiligten Wolken
eine haben.

## Abhängigkeiten

- NumPy und Open3D für ICP, FPFH und das Lesen der Dateien
- PyQt5 und VTK für Oberfläche und Anzeige
- ROS 2 Humble, nur für das Laden aus einem Bag
