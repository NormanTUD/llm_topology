# LLM Topology Lab

Ein lokales Forschungslabor, um die Hypothese zu testen, ob ein von Grund auf
trainierter Transformer topologische Eigenschaften von Punktwolken in seinen
internen Repräsentationen abbildet.

Kernidee:

1. Punktwolken aus bekannten topologischen Strukturen erzeugen.
2. Punktwolken als Zahlensequenz tokenisieren.
3. Ein kleines, vollständig selbst implementiertes Transformer-Modell von
   Grund auf trainieren, z. B. auf `...=betti(1,4,1)<eot>`.
4. Mit linearen Probes, Aktivierungs-Heatmaps, Attention-Maps und
   Ablationen untersuchen, wo und wie topologische Information im Modell
   repräsentiert ist.
5. Strukturen rotieren, strecken, twisten, verrauschen und die Invarianz
   sowie Out-of-Distribution-Generalisation messen.
6. Filtrationen und persistente Homologie explizit sichtbar und
   konfigurierbar machen.

Das ist ausdrücklich **kein Fine-Tuning von GPT-2 oder einem anderen
fremden Modell**. Das Modell, der Tokenizer, die Initialisierung und die
Trainingsschleife werden projektspezifisch implementiert.

## Wissenschaftliche Präzisierung

Die Formulierung "das LLM repräsentiert die Topologie eines Datensatzes"
muss in testbare Teilbehauptungen zerlegt werden:

- **Vorhersagbarkeit:** Kann das Modell die Zielgröße aus der Punktwolke
  vorhersagen?
- **Dekodierbarkeit:** Enthält der Residualstream lineare Information über
  Strukturtyp, Genus, Anzahl der Komponenten oder Betti-Zahlen?
- **Invarianz:** Bleibt diese Information unter Rotation, Translation,
  Skalierung, Streckung, Twist, Punktpermutation und Rauschen erhalten?
- **Kausalität:** Sind die gefundenen Richtungen/Neuronen tatsächlich
  notwendig? Das testen wir mit Ablation und Activation Patching, nicht nur
  mit linearen Probes.

Wichtig: Eine lineare Probe zeigt **Information**, nicht automatisch
**Nutzung** oder **kausale Repräsentation**.

### Betti-Zahlen allein sind nicht die ganze Topologie

Für eine endliche Punktwolke gibt es ohne Filtration keine eindeutigen
Betti-Zahlen. Deshalb unterscheidet das Projekt zwei Zieltypen:

- **Generator-Label:** Die theoretische Topologie der erzeugten Struktur,
  z. B. geschlossener Torus `(b0,b1,b2)=(1,2,1)`.
- **PH-Label:** Die aus einer expliziten Filtration berechnete Topologie,
  z. B. Vietoris-Rips- oder Alpha-Komplex bei einem gewählten Radius bzw.
  die Betti-Kurve `b_k(epsilon)` oder das Persistenzdiagramm.

Beide Labels werden gespeichert und in der GUI vergleichbar gemacht. So
sehen wir, ob das Modell die erzeugende Struktur oder nur die konkrete
Stichprobe bzw. den Filtrationsradius gelernt hat.

### Wichtige Beispiele für Zielwerte

Geschlossene orientierbare Fläche vom Genus `g`:

```text
b0 = 1
b1 = 2g
b2 = 1
```

Damit:

```text
Kreis                 (1, 1, 0)
Sphäre                (1, 0, 1)
Torus                 (1, 2, 1)
Doppeltorus, Genus 2  (1, 4, 1)
Dreifachtorus, Genus 3(1, 6, 1)
```

Zwei disjunkte Tori:

```text
(2, 4, 2)
```

Fünf disjunkte Doppeltori:

```text
(5, 20, 5)
```

Bei nicht-orientierbaren Flächen und bei Koeffizientenfeld `Z2` müssen die
Zahlen separat dokumentiert werden. Ripser rechnet standardmäßig über `Z2`.

Außerdem: "5 Doppeltori" kann bedeuten:

- fünf getrennte Doppeltorus-Komponenten, oder
- eine verbundene Struktur mit fünf Henkeln.

Das ändert die Betti-Zahlen. Diese Unterscheidung muss im Generator und in
der GUI explizit sein.

## Systemarchitektur

```text
Browser
  |
  |  HTML/CSS/JS, Plotly, Temml
  v
FastAPI + WebSocket
  |
  +-- Data Studio
  |     +-- Struktur-Katalog
  |     +-- Sampling
  |     +-- Transformationen
  |     +-- Filtration / persistente Homologie
  |
  +-- Model Lab
  |     +-- Custom Tokenizer
  |     +-- Custom Transformer
  |     +-- Initialisierung / Seeds
  |
  +-- Training
  |     +-- Subprozess, Live-Logs, Checkpoints
  |     +-- Optimizer, LR, Batch, Split, Early Stopping
  |
  +-- Interpretability
  |     +-- Lineare Probes
  |     +-- Aktivierungs- und Neuron-Heatmaps
  |     +-- Attention-Maps
  |     +-- Latent-Projektionen
  |     +-- Ablation / Activation Patching
  |
  +-- Evaluation
        +-- Test-Set, OOD-Transformationen
        +-- Betti-MAE, Exact Match, Probe-Accuracy
```

### Geplante Projektstruktur

```text
app.py                     # Bootstrap + Start
pyproject.toml             # uv-Projekt und Abhängigkeiten
src/llm_topology/
  config.py                # Pydantic-Konfigurationen
  tokenizer.py             # Ziffern-/Zahlen-Tokenizer
  topology/
    generators.py          # Punktwolken-Generatoren
    transforms.py          # Rotation, Stretch, Twist, Noise
    filtration.py          # Rips/Alpha, Betti-Kurven, Diagramme
  model/
    transformer.py         # eigener Transformer
    init.py                # kontrollierbare Initialisierung
    train.py               # Trainingsschleife
  analysis/
    probes.py              # lineare/ridge/logistische Probes
    attribution.py         # Aktivierungen, Ablation, Patching
    projection.py          # PCA/UMAP/t-SNE
  web/
    main.py                # FastAPI
    jobs.py                # Trainings-/Analyse-Jobs
    static/
  tests/
tests/
.github/workflows/ci.yml
```

## Automatischer Start mit uv

Ziel:

```bash
python3 app.py
```

Ablauf:

1. `app.py` prüft, ob es bereits in einer uv-Umgebung läuft.
2. Falls nicht, wird per `os.execvp` zu `uv run` gewechselt.
3. `uv` installiert bei Bedarf Python-Abhängigkeiten in eine lokale
   Umgebung.
4. Der Server startet und öffnet den Browser.
5. Ein Umgebungsflag verhindert eine Endlosschleife beim Neustart.

Falls `uv` noch nicht vorhanden ist, gibt der Bootstrap eine klare
Installationsanweisung aus. Ein automatisches `curl | sh` wird aus
Sicherheitsgründen nicht stillschweigend ausgeführt.

## Tokenizer und Sequenzgrammatik

Der Tokenizer ist bewusst klein und domänenspezifisch. Vorgesehen sind
mindestens:

```text
0 1 2 3 4 5 6 7 8 9
. , ; ( ) = -
betti
<bos> <eos> <pad>
```

Jede Ziffer ist ein eigenes Token. Zahlen werden als Dezimalstrings
serialisiert. Ein mögliches Format ist:

```text
<bos>1.23,4.56,7.89;2.00,3.10,4.20;...=betti(1,4,1)<eos>
```

Dabei trennt:

- `,` Koordinaten innerhalb eines Punktes und Werte innerhalb von `betti`,
- `;` Punkte,
- `=` Eingabe und Ziel,
- `betti(...)` das Ziel.

Offene Designfrage: Sollen Koordinaten zusätzlich als kontinuierliche
Vektoren embeddings bekommen, oder soll das Modell ausschließlich über
Ziffern-Tokens arbeiten? Für die Treue zur Idee ist die Ziffernvariante
Default; ein numerischer Baseline-Modus ist für Vergleichsexperimente
sinnvoll.

## Modell

Geplant ist ein kleiner, vollständig eigener Decoder-only-Transformer:

- kausale Self-Attention,
- konfigurierbare Layer, Heads, `d_model`, FFN-Breite,
- konfigurierbare Aktivierung (GELU, SiLU/SwiGLU),
- Pre-LayerNorm oder Post-LayerNorm,
- optionale gelernte Positionen oder RoPE,
- eigene Initialisierung: Normal, Xavier, orthogonal, uniform, zero,
  reproduzierbar über Seeds, optional pro Schicht überschreibbar,
- keine vortrainierten Gewichte,
- Token-Embedding und Ausgabekopf exakt für den eigenen Tokenizer.

Zwei Betriebsmodi sind denkbar:

1. **Autoregressives LLM:** erzeugt `betti(...)` als Text. Das entspricht
   der gewünschten Form am direktesten.
2. **Auxiliäre Klassifikationsköpfe:** zusätzliche Köpfe für Strukturtyp,
   Genus, Anzahl Komponenten und Betti-Zahlen. Sie erleichtern Probes und
   Diagnose, ersetzen aber nicht die generative Ausgabe.

Für Permutationsinvarianz wird die Reihenfolge der Punkte trainiert
randomisiert. Zusätzlich kann die Positionseinbettung für den
Punktwolken-Teil abgeschaltet werden. Das ist ein eigenes Experiment, weil
es die Architektur und Interpretierbarkeit verändert.

## Daten und Topologien

Der Katalog wird schrittweise erweitert. Sinnvolle erste Klassen:

- Kreis / S1
- Sphäre / S2
- Torus / T2
- Doppeltorus / Genus 2
- Dreifachtorus / Genus 3
- mehrere disjunkte Kopien
- verschachtelte oder verkettete Kreise
- Acht (wedge of circles)
- Annulus
- Möbiusband
- Kleinsche Flasche (Einbettung in 4D, Z2-Koeffizienten)
- projektive Ebene
- Zufallsrauschen / kontrahierbare Wolke als Negativkontrolle

Konfigurierbar pro Struktur:

- Anzahl der Instanzen,
- Anzahl der Punkte,
- intrinsische Dimension,
- Ambient-Dimension (2D, 3D, 4D, ...),
- Sampling-Verfahren,
- Rauschen,
- Skalierung,
- Rotation,
- anisotrope Streckung,
- Twist,
- Verschiebung,
- Seed.

Die GUI soll natürlichsprachige Presets anbieten, z. B.
"5 Doppeltori, 300 Punkte, 3D, 2% Rauschen", aber intern immer in eine
explizite, reproduzierbare Konfiguration übersetzen.

## Filtration und persistente Homologie

Geplante Verfahren:

- **Vietoris-Rips:** allgemein, aber teuer bei vielen Punkten und hoher
  Homologiedimension.
- **Alpha-Komplex:** für euklidische Punktwolken in niedriger Dimension
  effizienter; abhängig von Delaunay-Komplex.
- **Gewichtete/angepasste Filtrationen:** später, um Strukturen gezielt zu
  betonen.

Für jede Punktwolke werden gespeichert:

- Distanzmatrix,
- Persistenzdiagramme pro Dimension,
- Betti-Kurven `b_k(epsilon)`,
- optional Repräsentanten von Homologieklassen,
- Geburts-/Todesradien.

Die GUI zeigt interaktiv:

- Punktwolke in 3D,
- filtrierte Simplizes bei variablem `epsilon`,
- Betti-Kurven,
- Persistenzdiagramm,
- Vergleich Generator-Label vs. PH-Label.

## Training und Evaluation

Konfigurierbar:

- Train/Validation/Test-Split, auch gruppiert nach Strukturinstanz,
- Batch-Größe,
- Epochen,
- Optimizer (AdamW, Adam, SGD),
- Learning Rate, Weight Decay, Warmup, Scheduler,
- Dropout,
- Gradient Clipping,
- Early Stopping,
- Seed,
- Checkpoint-Intervall,
- Gerät (CPU/CUDA/MPS, automatische Erkennung).

Wichtige Evaluationsachsen:

- Betti-Zahlen exakt,
- mittlerer absoluter Fehler pro Betti-Komponente,
- Strukturklassifikation,
- Generalisierung auf ungesehene Punktanzahl,
- Generalisierung auf ungesehene Ambient-Dimension,
- Generalisierung auf ungesehene Strukturen,
- Invarianz unter Transformationen,
- lineare Probe-Accuracy pro Layer,
- kausale Ablationseffekte.

Train/Test-Trennung muss nach **Instanz** und nicht nur nach Punktwolke
erfolgen. Sonst lernt das Modell möglicherweise fast identische
Nachbarstichproben auswendig.

## Interpretierbarkeit

### Lineare Probes

Pro Layer, Token-Position und Zielgröße:

- Ridge-Regression für kontinuierliche Ziele (Genus, Betti-Zahlen),
- logistische Regression für Klassen (Strukturtyp, Anzahl Komponenten),
- Kreuzvalidierung,
- `R^2`, Accuracy, Konfusionsmatrix.

Visualisierung:

- Layer × Zielgröße Heatmap,
- Probe-Gewichte als Neuron × Zielgröße Heatmap,
- Top-N Neuronen pro Konzept,
- Vergleich Original vs. transformierte Struktur.

### Neuronale und attentionale Lokalisierung

- Aktivierungs-Heatmaps: Layer × Token, Layer × Neuron.
- Attention-Maps: Head × Token, inklusive Zuordnung der Tokens zu
  Koordinaten und Punkten.
- FFN-Neuronen: Aktivierungsstatistik pro Strukturklasse.
- Ablation einzelner Neuronen/Heads mit Effekt auf Betti-Vorhersage.
- Activation Patching zwischen z. B. Torus und Doppeltorus.

### Latente Projektion

- PCA (schnell, deterministisch) als Default,
- UMAP/t-SNE optional,
- 2D/3D-Scatter, gefärbt nach Struktur, Genus, Betti, Transformation,
  Split oder Vorhersagefehler,
- Trajektorien über Trainingsschritte.

Wichtig: t-SNE/UMAP sind Visualisierungen, keine Beweise für topologische
Struktur. Deshalb werden sie immer zusammen mit quantitativen Probes
gezeigt.

## GUI

Geplante Ansichten:

1. **Data Studio**
   - Strukturkatalog, Presets, natürliche Spracheingabe,
   - Live-Punktwolke,
   - Filtration, Betti-Kurven, Persistenzdiagramm,
   - Transformationsregler,
   - Generator vs. PH-Label.

2. **Model Lab**
   - Architekturregler,
   - Parameterzahl,
   - Initialisierung und Seeds,
   - Tokenizer-Vorschau,
   - interaktive Gleichungen mit Temml,
   - Shape-/FLOP-Vorschau.

3. **Training**
   - Live-Loss, LR, Accuracy,
   - generierte Beispiele,
   - Checkpoints,
   - Stop/Pause/Resume,
   - Log-Konsole.

4. **Interpretability**
   - Probe-Dashboard,
   - Heatmaps,
   - Attention-Maps,
   - Latent-3D,
   - Neuron Explorer,
   - Ablation/Patching.

5. **Evaluation**
   - Testmetriken,
   - OOD-Transformationen,
   - Invarianzmatrix,
   - Fehlerbeispiele,
   - Export als JSON/CSV.

6. **Equation Explorer**
   - Tokenizer,
   - Attention,
   - FFN,
   - LayerNorm/Residual,
   - Loss,
   - Vietoris-Rips-Komplex,
   - Betti-Zahlen,
   - Persistenz,
   - lineare Probe,
   - jeweils mit live berechneten Beispielwerten und Sliders.

## Ästhetik

- dunkles, hochwertiges Labor-Interface,
- großzügige Typografie,
- Plotly für interaktive 2D/3D-Plots,
- Temml für Formeln,
- konsistente Farbcodierung für Betti-Dimensionen und Strukturklassen,
- keine dekorativen Elemente, die Daten verdecken,
- responsiv, aber primär für Desktop.

## Test Driven Development

Geplante erste Tests:

- Tokenizer: Roundtrip, Sonderzeichen, negative Zahlen, Sonder-Tokens.
- Generatoren: Form, Determinismus bei Seed, korrekte theoretische Labels.
- Transformationen: Topologie-Label bleibt erhalten, Punktzahl bleibt
  erhalten, Determinismus.
- Filtration: Kreis erkennt H1, Sphäre erkennt H2, Torus erkennt H1 und H2
  bei geeigneter Stichprobe und Filtrationswahl.
- Modell: Forward-Shape, kausale Maske, Parameterzahl, reproduzierbare
  Initialisierung.
- Probes: synthetische linear separable Daten.
- API: Healthcheck, Datensatz-Erzeugung, ungültige Konfiguration.
- CI: `uv sync --dev`, `ruff`, `pytest`.

Topologische Tests sind stochastisch. Sie verwenden feste Seeds und
tolerante Schwellen, damit CI nicht flaky wird. Langsame PH-Tests werden
markiert und können separat laufen.

## Roadmap

- **Phase 0:** Bootstrap, Projektgerüst, Tokenizer, Generatoren,
  Transformationen, Filtration, Tests, CI.
- **Phase 1:** eigener Transformer, Init-Kontrolle, Training, Checkpoints,
  CLI-Experimente.
- **Phase 2:** FastAPI/WebSocket, Data Studio, Model Lab, Live-Training.
- **Phase 3:** Probes, Heatmaps, Attention-Maps, Latent-Projektion.
- **Phase 4:** Invarianz-/OOD-Benchmarks, Ablation, Activation Patching.
- **Phase 5:** Betti-Kurven/Persistenz als Lernziel, Equation Explorer,
  visueller Feinschliff.

## Risiken und Grenzen

- Vietoris-Rips in hohen Dimensionen und mit vielen Punkten ist teuer.
- Betti-Zahlen sind eine grobe Zusammenfassung; verschiedene Strukturen
  können dieselben Betti-Zahlen haben.
- Ein Modell kann Generatorparameter statt Topologie lernen.
- Lineare Probes können Scheinkorrelationen zeigen.
- Punktwolken mit endlicher Stichprobe und Rauschen haben keine saubere
  Topologie ohne Filtrationsentscheidung.
- Sehr lange Zahlenfolgen erhöhen Kontextlänge und Trainingskosten.
- Permutationsinvarianz ist bei sequenzieller Tokenisierung nicht
  automatisch gegeben.
- "Stunning" darf nicht auf Kosten von Reproduzierbarkeit und
  wissenschaftlicher Aussagekraft gehen.

## Offene Fragen

Die wichtigsten Entscheidungen, die vor der ersten Implementierung geklärt
werden sollten:

1. Soll das erste Modell zwingend autoregressiv `betti(...)` generieren,
   oder ist ein Encoder mit Klassifikationsköpfen als erste Stufe okay?
2. Sind mit "Doppeltorus" zwei getrennte Tori oder eine zusammenhängende
   Genus-2-Fläche gemeint?
3. Sollen die Trainingslabels theoretische Generator-Labels, PH-Labels oder
   beide sein?
4. Welche Filtration ist Default: Vietoris-Rips, Alpha-Komplex oder beides?
5. Soll das Modell statische Betti-Zahlen, Betti-Kurven oder
   Persistenzdiagramme vorhersagen?
6. Welche Strukturen sind für den ersten Katalog Pflicht?
7. Soll der Tokenizer ausschließlich Ziffern-Tokens nutzen, oder zusätzlich
   numerische Embeddings für ein Vergleichsexperiment?
8. Wie viele Punkte, welche Ambient-Dimensionen und welche
   Kontextlängen sind realistisch?
9. Soll die GUI zuerst funktional oder zuerst visuell poliert werden?
10. Welche Invarianz-Tests sind Pflicht: Rotation, Streckung, Twist,
    Translation, Skalierung, Permutation, Rauschen?
11. Soll es echte kausale Tests (Ablation/Patching) von Anfang an geben?
12. Gibt es eine GPU? Falls nein, muss die Standardkonfiguration auf CPU
    laufen.
