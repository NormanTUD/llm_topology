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
Die drei Ansichten laufen in einer Single-Page-App ohne Neuladen. UI-Zustand,
letzte Batch-Generierung und Trainingslauf werden serverseitig gespeichert und
nach F5 wiederhergestellt.

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

Danach:

- `http://127.0.0.1:8000/` öffnet das Data Studio mit Batch-Galerie,
- `http://127.0.0.1:8000/lab` öffnet Model Lab und Training,
- `http://127.0.0.1:8000/lab#interpret` öffnet die Probe- und Heatmap-Ansicht.

Alle drei Ansichten sind Tabs derselben Single-Page-App und wechseln ohne
Reload.

Workflow:

1. Im Data Studio Punktwolken generieren und die Rezeptur prüfen.
2. Unten im Data Studio `Build & save dataset` ausführen. Das erzeugt
   `datasets/ds-NNNN/` mit Train-/Val-/Test-Beispielen. Bei zu kurzem
   Kontext wird dieser automatisch auf die nächste Zweierpotenz angehoben.
3. Im Model Lab das gespeicherte Dataset auswählen und trainieren. Val- und
   Test-Split können im Dataset Builder getrennt transformiert werden.
4. Während des Trainings Live-Loss, Validierungspaare und Vorhersagen sehen.
5. Nach dem Training `Run probes & maps` oder frische Multi-Cloud-Predictions
   starten.

Ein Trainingslauf im Model Lab startet im Hintergrund. Die Oberfläche pollt
den Status, zeigt Live-Loss, Validierungsmetriken und generierte Samples und
kann den Lauf stoppen. Nach dem Training startet `Run probes & maps` die
Interpretabilitätsanalyse des besten Checkpoints.

## Aktueller Stand

**Phase 0, Phase 1 und Phase 2 sind implementiert.**

Data Foundation:

- automatischer uv-Bootstrap über `python3 app.py`,
- FastAPI-Server mit Browser-GUI,
- eigener Ziffern-/Zahlen-Tokenizer,
- konfigurierbares Zielgruppen-Schema,
- 12 Punktwolken-Generatoren,
- Transformationen: Rotation, anisotrope Streckung, Twist, Shear, Bend,
  Warp, Translation, Rauschen, Permutation,
- Alpha-Komplex und Vietoris-Rips,
- Persistenzdiagramme, Betti-Zahlen und Betti-Kurven,
- automatische Wahl eines Filtrationsskalenpunkts,
- Z2 und Z3 als Q-Proxy,
- Batch-Galerie mit mehreren Punktwolken,
- Batch-Modus `single` oder `mix` mit expliziter Strukturauswahl,
- Live-Update der Geometrie bei Regleränderungen,
- Trainingsdatensatz-Erzeugung nach `datasets/ds-NNNN/`,
- getrennte Train-/Val-/Test-Augmentierung mit `clean`, `same`, `heldout`,
- Test-Split und Test-Metriken nach dem Training,
- Dataset-Preview pro Split mit Original- und augmentierter Punktwolke,
  Tokenlängen, Zielen und tatsächlich angewendeter Transformation,
- Punkte pro Komponente (z.B. 10 Tori mit je 400 Punkten) zusätzlich zur
  Gesamtpunktzahl,
- Epsilon als Input-Token im manuellen Modus plus Auto-Modus,
- einstellbare String-Präzision mit Default 5 Nachkommastellen,
- automatische Kontextvergrößerung im Dataset Builder, wenn das Token-Budget
  nicht passt,
- Temml-Gleichungsansicht,
- pytest-Suite und GitHub-Actions-CI.

Model Lab:

- eigener Decoder-only-Transformer ohne Pretraining,
- kontrollierbare Initialisierung und Architektur,
- Multi-Task-Training: Inferenz, bedingte Punktwolkengenerierung und
  auxiliäre Köpfe,
- Cross-Entropy, Ranking/Kontrastivloss und Multiple-Choice-Evaluation,
- AdamW/Adam/SGD, LR-Scheduler, Warmup, Early Stopping, Checkpoints,
- CPU/CUDA/MPS-Erkennung, CPU-Default,
- Live-Training im Browser mit Polling, Batch-Loss-Kurven, Validierungspaaren
  (Inputwolke, wahre vs. vorhergesagte Betti-Zahlen) und Samples,
- Start/Stop als ein umschaltender Button,
- Auswahl gespeicherter Datasets statt doppelter Dataset-Regler,
- getrennte Train-/Val-Transformationen im Dataset Builder,
- Experimentordner `runs/exp-NNNN/` mit Checkpoints, Job-Snapshot und History,
- CLI-Logging für Start, Epochen, Batches, Validierung und Checkpoints.

Interpretability:

- Ridge-Probes für Betti/PH/Genus/Komponenten pro Layer,
- logistische Probe für den Strukturtyp,
- Probe-Gewichte und Top-Neuronen,
- Neuron-Aktivierungs-Heatmaps,
- Attention-Maps am Boundary-Token,
- FFN-Neuronen-Aktivierungen,
- PCA-Projektion der Residualstativen,
- Test mit frisch generierten Punktwolken: true vs. predicted targets,
- Trainingshistorie und Checkpoint-Verwaltung.

Noch nicht enthalten:

- Ablation und Activation Patching (Phase 3),
- OOD-/Invarianzmatrix (Phase 3),
- Betti-Kurven/Persistenz als Trainingsziel (Phase 4),
- exakte Q-Berechnung (Phase 4),
- eigene Filtrationen, Formel-/SDF-Eingabe, Uploads (Phase 4),
- interaktiver Equation Explorer (Phase 4).

Aktueller Testlauf: **97 Tests bestanden**, Ruff und Formatierung sauber.

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

Zusätzlich zeigt das Data Studio eine kanonische Referenzwolke und
Referenz-Betti-Zahlen als Ghost-Overlay. Diese Referenz ist ausdrücklich nur
für den Menschen sichtbar und wird **nicht** in den Modell-Input oder die
Trainingssequenz aufgenommen.

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
`copies=1`. Für `structure=torus` und `copies=N` gilt entsprechend
`b0=N`, `b1=2N`, `b2=N`; alle Kopien bleiben Tori. `Components min/max`
zieht pro Wolke eine zufällige Anzahl im Bereich, z.B. 1–5. Für stabile
PH-Labels sollten es etwa 400 Punkte pro Kopie sein. Die Auto-Filtration
zählt nur Persistenzklassen oberhalb einer relativen Signifikanzschwelle;
das reduziert winzige Sampling-Artefakte.

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

Die Default-Reihenfolge im tatsächlichen Trainingsstring ist geprüft:
`betti;ph;genus;n_components;filtration`. Ein Test in `tests/test_training.py`
verhindert, dass diese Reihenfolge unbemerkt auseinanderläuft.

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
- Shear (invertierbar),
- Bend (invertierbar),
- glatter Warp (kleine Amplitude),
- Translation,
- Rauschen,
- Punktpermutation.

Shear, Bend und Twist sind als diffeomorphe Verzerrungen gedacht: Die ideale
Topologie bleibt erhalten, die konkrete Punktwolke wird aber deutlich
variiert. Rauschen ist separat und kann die PH-Signatur bewusst stören. Im
Training steuert `Augment strength`, wie stark diese Transformationen
zufällig angewendet werden.

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

**Auto- oder manuelles Epsilon:** Im Data Studio kann `epsilon` automatisch
gewählt oder explizit gesetzt werden. Der Dataset Builder übernimmt diese
Wahl. Bei `auto` variiert epsilon pro Punktwolke; das Modell muss den Wert
als Teil der `filtration`-Zielgruppe mitlernen. Bei manuellem epsilon ist der
Wert für den Datensatz fixiert und wird zusätzlich als `<epsilon>…;`-Präfix
in den Prompt gegeben. Die Trainings- und Prediction-Ansichten zeigen wahres
und vorhergesagtes epsilon nebeneinander.

**Koeffizienten:** Z2 ist der Default. Z3 dient als Q-ähnlicher
Vergleichskörper, ist aber **keine exakte Q-Berechnung**. Für orientierbare
Flächen sind die Werte gleich; für nicht-orientierbare Flächen zeigt Z3 das
Q-Verhalten, aber Torsion kann in anderen Räumen abweichen. Exaktes Q ist
für eine spätere Phase vorgesehen.

## Modell

Implementiert ist ein vollständig eigener Decoder-only-Transformer:

- kausale Self-Attention,
- konfigurierbare Layer, Heads, `d_model`, FFN-Breite,
- GELU, SiLU oder ReLU,
- Pre- oder Post-LayerNorm,
- gelernte Positionen, RoPE oder keine Positionsinformation,
- eigene Initialisierung: Normal, Xavier, orthogonal, uniform, zero,
  reproduzierbar über Seeds,
- keine vortrainierten Gewichte,
- eigener Token-Embedding- und Ausgabekopf,
- optionale Weight Ties.

Das Modell wird **multitasking** trainiert:

1. **Inferenz:** `<infer>` + Punktwolke + `<endoftext>` + Zielgruppen.
2. **Generierung:** `<generate>` + Zielgruppen + `<endoftext>` + Punktwolke.
3. **Auxiliäre Köpfe:** Strukturtyp, Genus, Komponenten, Betti-Zahlen,
   PH-Betti-Zahlen.
4. **Trainingsziele:** maskierte Cross-Entropy, auxiliäre MSE/CE,
   Ranking/Kontrastivloss und Multiple-Choice-Evaluation.

Default für CPU: 3 Layer, 4 Heads, `d_model=96`, FFN 384, Kontext 1024.
Trainingsdatasets starten mit 32 Punkten und 5 Nachkommastellen; das Data
Studio zeigt standardmäßig 400 Punkte pro Wolke für stabilere PH-Labels.
Alle Werte sind änderbar. Trainiert wird mit konfigurierbarer Batch-Größe;
N Punktwolken werden also gleichzeitig als Batch verarbeitet, nicht als eine
einzige Riesensequenz. Bei mehreren Komponenten kann
`Points per component` gesetzt werden; die Gesamtzahl ist dann
`points_per_component * copies`. `Components min/max` erlaubt z.B. 1–5 Tori
pro Wolke.

Die bedingte Punktwolkengenerierung ist als reine Sequenzgenerierung mit
Task-Token umgesetzt. Ein separater Decoder-Kopf bleibt eine spätere
Alternative. Die genaue Gewichtung von CE, Auxiliary Loss und Ranking ist im
Browser konfigurierbar. Frühe Epochen liefern oft noch keine brauchbaren
Punktwolken; das ist sichtbar und Teil des Experiments.

## Interpretierbarkeit

Implementiert:

- Ridge-Probes pro Layer und Boundary-Token für Betti, PH, Genus und
  Komponenten,
- logistische Probe für den Strukturtyp,
- Probe-Gewichte als Ziel × Neuron Heatmap,
- Top-Neuronen pro Ziel und Layer,
- Aktivierungs-Heatmap Layer × Neuron,
- Attention-Maps am Boundary-Token,
- FFN-Neuronen-Aktivierungen am Boundary-Token,
- PCA-Projektion der Residualstativen,
- Layer-Auswahl, Head-Auswahl und Live-Heatmaps im Browser.

Noch geplant:

- Ablation einzelner Neuronen/Heads,
- Activation Patching zwischen Strukturen und Transformationen,
- UMAP/t-SNE,
- kausale Karten und Invarianzmatrix.

Probes zeigen Dekodierbarkeit, nicht kausale Nutzung. Deshalb folgen kausale
Tests in Phase 3.

## GUI

Es gibt drei Browser-Ansichten:

Alle drei Ansichten laufen in einer Single-Page-App; die Tab-Wechsel laden
nicht neu. UI-Zustand, letzte Punktwolken-Batch und Trainingslauf werden
serverseitig gespeichert und nach F5 wiederhergestellt.

1. **Data Studio**
   - Batch-Galerie mit mehreren Punktwolken,
   - Auswahl einer Wolke für die Detailansicht,
   - Batch-Modus `single`/`mix`, Live-Update, Strukturauswahl,
   - Strukturkatalog, Transformationen inklusive Shear/Bend/Warp,
   - kanonische Referenzwolke als Ghost-Overlay (nicht im Modell-Input),
   - Topology Guard: theoretische vs. berechnete Betti-Zahlen,
   - Trainingsdatensatz-Erzeugung mit getrennten Train-/Val-/Test-Splits,
   - Filtration und Zielgruppen mit Auto- oder manuellem Epsilon,
   - 3D-Punktwolke, Persistenzdiagramm mit Referenz-Betti, Betti-Kurven,
   - theoretische vs. berechnete Betti-Zahlen,
   - Zielstring, Modell-Input und Tokenstrom,
   - Temml-Gleichungen mit Live-Werten.

2. **Model Lab**
   - Auswahl eines gespeicherten Datasets,
   - Dataset-Preview pro Split inklusive Transformationszusammenfassung,
   - Modell- und Trainingsregler,
   - Start/Stop-Toggle, Live-Batch-Loss, Validierungsmetriken,
   - Live-Paare aus Inputwolke, wahrem Ziel und Modellvorhersage,
   - generierte Punktwolken bzw. generierte PH-Betti,
   - Run-Historie und Checkpoints unter `runs/exp-NNNN/`.

3. **Interpretability**
   - Probe-Scores pro Layer,
   - Probe-Gewichts-Heatmap und Top-Neuronen,
   - PCA-Projektion des Residualstreams,
   - Layer × Neuron Heatmap,
   - Attention-Map und FFN-Neuronen am Boundary-Token,
   - frische Multi-Cloud-Predictions mit true vs. predicted.

Noch geplant: Invarianzmatrix, Ablation, Activation Patching, Equation
Explorer und Uploads.

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
  datasets.py                   Persistente Train-/Val-Datasets
  tokenizer.py                  Ziffern-/Zahlen-Tokenizer
  pipeline.py                   Generate -> PH -> Ziele -> Sequenz
  topology/
    generators.py               Punktwolken-Generatoren
    filtration.py               Alpha/Rips, Diagramme, Betti-Kurven
    labels.py                   Zielgruppen und Schema
  model/
    config.py                   Modell-, Daten- und Trainingsconfig
    transformer.py              Eigener Transformer + RoPE + Init
    data.py                     Inferenz-/Generierungssequenzen
    train.py                    Multi-Task-Training + Checkpoints
  analysis/
    probes.py                   Probes, Attention, Neuronen, PCA
  web/
    main.py                     FastAPI
    jobs.py                     Trainings-Jobmanager + Prediction
    state.py                    Server-seitiger UI-Zustand
    templates/index.html        Single-Page-App
    templates/lab_body.html     Model Lab + Interpretability
    static/style.css            Dark-Lab-Design
    static/shell.js             Tab-Routing ohne Reload
    static/state.js             UI-Zustand speichern/laden
    static/app.js               Data-Studio-Interaktion
    static/lab.js               Training, Polling, Probes, Predictions
tests/                          pytest-Suite
.github/workflows/ci.yml        GitHub Actions
```

CI führt Ruff, Formatcheck und pytest aus. PyTorch wird über den CPU-Index
installiert, damit der lokale Bootstrap ohne CUDA-Abhängigkeiten bleibt.

Trainingsartefakte liegen unter `runs/exp-NNNN/` (`best.pt`, `last.pt`,
`history.json`, `job.json`, optional `analysis.json`). Datasets liegen unter
`datasets/ds-NNNN/` (`meta.json`, `data.json`) und enthalten die Rezeptur
inklusive separater Train-/Val-Augmentierung und Filtrationsmodus.
`runs/`, `datasets/` und `state/` sind gitignored. Der UI-Zustand wird unter
`state/ui_state.json` gespeichert.

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
  invariant; die Trainingsdaten nutzen Permutationsaugmentation, der Effekt
  muss aber separat gemessen werden.
- Der Ziffern-Tokenizer erzeugt lange Sequenzen. 240 Punkte in 3D koennen
  leicht 7.000+ Tokens ergeben. Die Defaults sind deshalb klein: 32 Punkte,
  3 Nachkommastellen und Kontext 1024. Groessere Runs brauchen mehr Kontext
  oder eine andere Koordinatenpraezision.
- Die Genus-g-Punktwolke über verschmolzene Tori ist topologisch motiviert,
  aber die PH-Signatur kann durch die diskrete Oberfläche verrauscht sein.
- Z3 ist nur ein Q-Proxy.

## Nächste offene Entscheidungen für Phase 3

1. Welche Ablationsachse zuerst: Neuronen, Heads, Layer oder einzelne
   Residualrichtungen?
2. Soll Activation Patching zwischen Strukturen, Transformationen oder
   Generator- vs. PH-Label laufen?
3. Welche OOD-Achse ist Pflicht: Rotation, Stretch, Twist, Rauschen,
   Punktanzahl, Ambient-Dimension oder ungesehene Genus-Werte?
4. Soll die Invarianzmatrix pro Struktur oder pro Betti-Komponente
   aggregiert werden?
5. Sollen generierte Punktwolken automatisch topologisch ausgewertet und in
   die Trainingsdiagnose aufgenommen werden?
6. Wie wird exaktes Q für kleine Komplexe berechnet: rationale
   Gauß-Elimination, Gudhi-Erweiterung oder externe Bibliothek?
7. Welche Gleichungen müssen im Equation Explorer zuerst interaktiv werden?
