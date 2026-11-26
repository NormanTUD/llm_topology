# LLM Topology Lab

Ein lokales Forschungslabor, um die Hypothese zu testen, ob ein von Grund auf
trainierter Transformer topologische Eigenschaften von Punktwolken in seinen
internen Repräsentationen abbildet.

Kernidee:

1. Punktwolken aus bekannten topologischen Strukturen erzeugen.
2. Punktwolken als Zahlensequenz tokenisieren.
3. Ein kleines, vollständig selbst implementiertes Transformer-Modell von
   Grund auf trainieren.
4. Mit linearen Probes, Aktivierungs-Heatmaps, Attention-Maps, Ablationen und
   Activation Patching untersuchen, wo und wie topologische Information im
   Modell repräsentiert ist.
5. Strukturen rotieren, strecken, twisten, verrauschen und die Invarianz
   sowie Out-of-Distribution-Generalisation messen.
6. Filtrationen und persistente Homologie explizit sichtbar und
   konfigurierbar machen.

Das ist ausdrücklich **kein Fine-Tuning von GPT-2 oder einem anderen fremden
Modell**. Modell, Tokenizer, Initialisierung und Trainingsschleife werden
projektspezifisch implementiert.

Die Web-Oberfläche ist auf Englisch; Konzepte und Dokumentation sind deutsch.

## Schnellstart

```bash
python3 app.py
```

Beim ersten Start:

1. `app.py` prüft, ob `uv` vorhanden ist.
2. Falls nötig, wird `uv` per `pip --user` installiert. Es wird kein
   `curl | sh` stillschweigend ausgeführt.
3. `uv run --project .` installiert die Abhängigkeiten in eine lokale
   Umgebung.
4. Der Server startet und öffnet den Browser.

Alternativ:

```bash
uv run app.py
uv run app.py --no-browser --port 8000
```

## Aktueller Stand

**Phase 0 - Data Foundation ist implementiert.**

Enthalten:

- automatischer uv-Bootstrap über `python3 app.py`,
- FastAPI-Server mit Browser-GUI,
- eigener Ziffern-/Zahlen-Tokenizer,
- konfigurierbares Zielgruppen-Schema,
- 12 Punktwolken-Generatoren,
- Transformationen: Rotation, anisotrope Streckung, Twist, Translation,
  Rauschen, Permutation,
- Alpha-Komplex und Vietoris-Rips,
- Persistenzdiagramme, Betti-Zahlen und Betti-Kurven,
- automatische Wahl eines Filtrationsskalenpunkts,
- Z2 und Z3 als Q-Proxy,
- Live-3D-Punktwolke, Persistenzplot, Betti-Kurven, Zielvektor und
  Tokenstrom,
- Temml-gestützte Gleichungsansicht,
- pytest-Suite und GitHub-Actions-CI.

Noch nicht enthalten:

- das eigene Transformer-Modell,
- Training, Checkpoints und Optimierer,
- lineare Probes und Heatmaps,
- Ablation und Activation Patching,
- bedingte Punktwolkengenerierung,
- Betti-Kurven als Trainingsziel,
- exakte Q-Berechnung.

Aktueller Testlauf: **65 Tests bestanden**, Ruff und Formatierung sauber.

## Wissenschaftliche Präzisierung

Die Formulierung "das LLM repräsentiert die Topologie eines Datensatzes"
muss in testbare Teilbehauptungen zerlegt werden:

- **Vorhersagbarkeit:** Kann das Modell die Zielgröße aus der Punktwolke
  vorhersagen?
- **Dekodierbarkeit:** Enthält der Residualstream lineare Information über
  Strukturtyp, Genus, Anzahl der Komponenten oder Betti-Zahlen?
- **Invarianz:** Bleibt diese Information unter Rotation, Translation,
  Skalierung, Streckung, Twist, Punktpermutation und Rauschen erhalten?
- **Kausalität:** Sind gefundene Richtungen/Neuronen tatsächlich notwendig?
  Das testen wir mit Ablation und Activation Patching, nicht nur mit
  linearen Probes.

Eine lineare Probe zeigt **Information**, nicht automatisch **Nutzung** oder
**kausale Repräsentation**.

### Betti-Zahlen allein sind nicht die ganze Topologie

Für eine endliche Punktwolke gibt es ohne Filtration keine eindeutigen
Betti-Zahlen. Deshalb unterscheidet das Projekt explizit:

- **Generator-Label:** Die theoretische Topologie der erzeugten Struktur,
  z. B. geschlossener Torus `(1,2,1)`.
- **PH-Label:** Die aus der konkreten Punktwolke und einer Filtration
  berechnete Topologie.

Beide Werte werden getrennt gespeichert und in der GUI verglichen. So sehen
wir, ob das Modell die erzeugende Struktur oder nur die konkrete Stichprobe
bzw. den Filtrationsradius gelernt hat.

Beispiele geschlossener orientierbarer Flächen vom Genus `g`:

```text
b0 = 1
b1 = 2g
b2 = 1
```

```text
Kreis                  (1, 1, 0)
Sphäre                 (1, 0, 1)
Torus                  (1, 2, 1)
Doppeltorus, Genus 2   (1, 4, 1)
Dreifachtorus, Genus 3 (1, 6, 1)
```

Zwei disjunkte Tori: `(2,4,2)`. Fünf disjunkte Genus-2-Flächen: `(5,20,5)`.

"5 Doppeltori" wird als fünf getrennte Genus-2-Komponenten umgesetzt. Eine
einzelne zusammenhängende Genus-2-Fläche entsteht über `genus=2` und
`copies=1`.

## Zielformat

Es gibt kein `=` im Trainingsstring. Ein explizites Grenz-Token trennt
Eingabe und Ziel. Default:

```text
<endoftext>
```

Die Punktwolke steht davor, die Zielgruppen danach. Koordinaten werden mit
`,` getrennt, Punkte mit `;`. Zielgruppen werden ebenfalls mit `;` getrennt,
Werte innerhalb einer Gruppe mit `,`.

Beispiel:

```text
1.2,3.4,5.6;7.8,9.0,1.2<endoftext>1,2,1;1,3,1;1;1;0,2,0.5,2
```

Default-Reihenfolge der Zielgruppen:

| Gruppe | Felder | Bedeutung |
| --- | --- | --- |
| `betti` | `b0,b1,b2` | theoretische Generator-Topologie |
| `ph` | `b0,b1,b2` | berechnete Topologie bei `epsilon` |
| `genus` | `genus` | Genus bzw. Summe bei Kopien |
| `n_components` | `n_components` | Anzahl Komponenten |
| `filtration` | `method_code,maxdim,epsilon,field_code` | Filtrationsmetadaten |

Codes:

```text
method: alpha=0, rips=1
field:  Z2=2, Z3=3
```

Die Reihenfolge und Auswahl der Gruppen ist in der GUI konfigurierbar. Die
Felder innerhalb einer Gruppe sind fest definiert. Das Modell lernt die
Reihenfolge aus dem Trainingslauf; die GUI zeigt sie explizit an.

Optional zuschaltbar sind `structure_id`, `n_points` und `ambient_dim`.
`structure_id` ist ein numerischer Klassenindex.

## Tokenizer

Der Tokenizer ist bewusst klein und domänenspezifisch:

```text
0 1 2 3 4 5 6 7 8 9 . , ; - ( ) =
betti ph genus meta
<pad> <bos> <eos> <endoftext> <infer> <generate> <unk>
```

Jede Ziffer ist ein eigenes Token. Zahlen werden als Dezimalstrings
serialisiert. `betti`, Klammern und `=` bleiben für spätere Grammatiken und
Vergleichsmodi im Vokabular, auch wenn das aktuelle Zielformat sie nicht
verwendet.

## Punktwolken und Topologien

Implementierter Katalog:

- Kreis / S1
- Sphäre / S2
- Torus / T2
- Genus-g-Fläche (über verschmolzene Tori)
- disjunkte Kopien
- verkettete Ringe (Hopf link)
- Acht / Figure Eight
- Kleinsche Flasche (4D-Einbettung, Z2)
- Möbiusband
- Annulus
- Rauschen
- Vollkugel

Konfigurierbar:

- Anzahl Punkte,
- Ambient-Dimension,
- Kopien,
- Genus,
- Seed,
- Rotation,
- anisotrope Streckung,
- Twist,
- Translation,
- Rauschen,
- Punktpermutation.

Es gibt keine kleinen künstlichen Obergrenzen. Ein lokales Safety-Envelope
von 100.000 Punkten verhindert versehentliche OOM-Runs; oberhalb von 2.000
Punkten erscheinen Warnungen. Höhere Ambient-Dimensionen und
Homologiedimensionen sind erlaubt, werden aber als teuer markiert.

**Bekannte Einschränkung:** Die Genus-g-Fläche wird als Punktwolke auf der
Oberfläche verschmolzener Tori erzeugt. Das theoretische Label ist korrekt,
aber Alpha-Komplexe auf endlichen, verrauschten Oberflächenstichproben
können zusätzliche kurze H1/H2-Klassen zeigen. Genau dieser Unterschied
zwischen Generator- und PH-Label ist Teil des Experiments.

## Filtration und persistente Homologie

Default: **Alpha-Komplex**. Optional: **Vietoris-Rips**.

Alpha-Filtrationswerte sind intern quadratische Radien; die API gibt Radien
zurück. Rips verwendet Distanzen.

Gespeichert und visualisiert:

- Persistenzdiagramme pro Dimension,
- Betti-Zahlen am gewählten `epsilon`,
- Betti-Kurven über einen Radius-Grid,
- Geburts-/Todeswerte,
- Warnungen bei unendlichen positiven Klassen.

Die automatische Epsilon-Wahl ist ein bewusst einfacher, reproduzierbarer
Heuristik-Start:

1. Falls eine signifikante H2-Klasse existiert, wähle deren Intervallmitte.
2. Sonst wähle die längste signifikante H1-Klasse.
3. Sonst falle auf einen Bruchteil des Filtrationsbereichs zurück.

Das ist kein universeller topologischer Schätzer. Die GUI erlaubt deshalb
jederzeit manuelles `epsilon`.

**Koeffizienten:** Z2 ist der Default. Z3 dient als Q-ähnlicher
Vergleichskörper, ist aber **keine exakte Q-Berechnung**. Für orientierbare
Flächen sind die Werte gleich; für nicht-orientierbare Flächen zeigt Z3 das
Q-Verhalten, aber Torsion kann in anderen Räumen abweichen. Exaktes Q ist
für eine spätere Phase vorgesehen.

## Geplantes Modell

Vollständig eigener Decoder-only-Transformer:

- kausale Self-Attention,
- konfigurierbare Layer, Heads, `d_model`, FFN-Breite,
- GELU oder SiLU/SwiGLU,
- Pre- oder Post-LayerNorm,
- gelernte Positionen oder RoPE,
- eigene Initialisierung: Normal, Xavier, orthogonal, uniform, zero,
  reproduzierbar über Seeds und pro Schicht überschreibbar,
- keine vortrainierten Gewichte,
- eigener Token-Embedding- und Ausgabekopf.

Das Modell wird von Anfang an **multitasking** trainiert:

1. **Inferenz:** Punktwolke + `<endoftext>` + Zielgruppen.
2. **Generierung:** Zielgruppen + `<endoftext>` + Punktwolke.
3. **Auxiliäre Köpfe:** Strukturtyp, Genus, Komponenten, Betti-Zahlen.
4. **Trainingsziele:** Cross-Entropy, Ranking/Kontrastivloss und
   Multiple-Choice parallel.

Die genaue Gewichtung und die Frage, ob Generierung zuerst nur als
denoising-/autoencodingartige Aufgabe trainiert wird, ist eine
Phase-1-Entscheidung.

## Interpretierbarkeit

Geplant:

- Ridge-/logistische Probes pro Layer und Token-Position,
- Probe-Gewichte als Neuron × Zielgröße Heatmap,
- Aktivierungs-Heatmaps Layer × Token und Layer × Neuron,
- Attention-Maps mit Zuordnung zu Koordinaten und Punkten,
- FFN-Neuronen-Aktivierungen pro Strukturklasse,
- Ablation einzelner Neuronen/Heads,
- Activation Patching zwischen Strukturen und Transformationen,
- PCA/UMAP-Projektionen des Residualstreams,
- Trajektorien über Trainingsschritte.

t-SNE/UMAP sind Visualisierungen, keine Beweise. Sie werden immer zusammen
mit quantitativen Probes und kausalen Tests gezeigt.

## GUI

Die aktuelle Data-Studio-Ansicht bietet:

- Strukturkatalog und Live-Parameter,
- Transformationsregler,
- Filtrationseinstellungen,
- konfigurierbare Zielgruppen,
- 3D-Punktwolke (PCA-Projektion bei mehr als 3 Dimensionen),
- Persistenzdiagramm,
- Betti-Kurven,
- theoretische vs. berechnete Betti-Zahlen,
- kanonischen Zielstring und vollständigen Modell-Input,
- Tokenstrom mit Token-IDs,
- Temml-Gleichungen mit Live-Werten,
- Warnungen und graceful Fehlerbehandlung.

Geplante Ansichten:

- Model Lab mit Architekturreglern und Parameterzahl,
- Training-Dashboard mit Live-Loss und Checkpoints,
- Interpretability-Dashboard mit Probes, Heatmaps, Attention und Ablation,
- Evaluation mit OOD- und Invarianzmatrix,
- interaktiver Equation Explorer.

## Entwicklung

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests app.py
uv run ruff format --check src tests app.py
```

Projektstruktur:

```text
app.py                         uv-Bootstrap und Serverstart
pyproject.toml                  uv-Projekt
src/llm_topology/
  config.py                     Pydantic-Konfigurationen
  tokenizer.py                  Ziffern-/Zahlen-Tokenizer
  pipeline.py                   Generate -> PH -> Ziele -> Sequenz
  topology/
    generators.py               Punktwolken-Generatoren
    filtration.py               Alpha/Rips, Diagramme, Betti-Kurven
    labels.py                   Zielgruppen und Schema
  web/
    main.py                     FastAPI
    templates/index.html        Data Studio
    static/style.css            Dark-Lab-Design
    static/app.js               Interaktion und Plotly
tests/                          pytest-Suite
.github/workflows/ci.yml        GitHub Actions
```

CI führt Ruff, Formatcheck und pytest aus.

## Grenzen und Risiken

- Vietoris-Rips ist in hohen Dimensionen und mit vielen Punkten teuer.
- Alpha-Komplexe haben bei degenerierten Punktwolken große
  Filtrationswerte; die GUI kappt den sinnvollen Radiusbereich.
- Betti-Zahlen sind eine grobe Zusammenfassung. Verschiedene Strukturen
  können dieselben Betti-Zahlen haben.
- Ein Modell kann Generatorparameter statt Topologie lernen.
- Lineare Probes können Scheinkorrelationen zeigen.
- Endliche, verrauschte Stichproben haben keine saubere Topologie ohne
  Filtrationsentscheidung.
- Punktpermutation ist bei sequenzieller Tokenisierung nicht automatisch
  invariant; Permutationsaugmentation ist eingeplant.
- Der Ziffern-Tokenizer erzeugt lange Sequenzen. 240 Punkte in 3D koennen
  leicht 7.000+ Tokens ergeben. Kontextlaenge, Koordinatenpraezision und
  Punktbudget muessen in Phase 1 gemeinsam kalibriert werden.
- Die Genus-g-Punktwolke über verschmolzene Tori ist topologisch motiviert,
  aber die PH-Signatur kann durch die diskrete Oberfläche verrauscht sein.
- Z3 ist nur ein Q-Proxy.

## Nächste offene Entscheidungen für Phase 1

1. Soll die bedingte Punktwolkengenerierung zuerst als reine
   Sequenzgenerierung oder als separater Decoder-Kopf umgesetzt werden?
2. Welche Default-Modellgröße ist auf CPU sinnvoll?
3. Soll `structure_id` numerisch oder alphanumerisch tokenisiert werden?
4. Soll die Trainingsreihenfolge der Zielgruppen frei permutierbar sein oder
   pro Run fixiert bleiben?
5. Wie wird exaktes Q für kleine Komplexe berechnet: rationale
   Gauß-Elimination, Gudhi-Erweiterung oder externe Bibliothek?
6. Sollen benutzerdefinierte Python-Snippets über AST-Whitelists oder nur
   als lokal vertrauenswürdiger Subprozess laufen?
7. Welche Gleichungen müssen im Equation Explorer zuerst interaktiv werden?
