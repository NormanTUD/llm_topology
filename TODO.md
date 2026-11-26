# Projekt-TODO

## Aktueller Stand

- **Aktuelle Phase:** Phase 0 - Discovery und Basisfunktionen
- **Status:** Kernentscheidungen geklaert; Implementierung beginnt
- **Naechster Meilenstein:** Phase 0 abschliessen, dann Phase 1 (Modell + Training)

Legende: `[ ]` offen, `[~]` in Arbeit, `[x]` erledigt

## Bereits entschieden

- [x] Vollstaendig eigenes Modell, kein Pretraining, keine GPT-2-Gewichte
- [x] Automatischer Bootstrap ueber `uv`
- [x] Browser-GUI mit Live-Visualisierungen
- [x] TDD und CI von Anfang an
- [x] Tokenizer mit einzelnen Ziffern, `.` `,` `;` `(` `)` `=` `-`,
      `betti`, `<bos>`, `<eos>`, `<pad>`
- [x] Autoregressive Ausgabe `...=betti(1,4,1)` plus auxiliaere Koepfe
- [x] Ziffern-Tokenizer als Default, numerische Embeddings als Baseline
- [x] Generator- und PH-Labels beide vorsehen
- [x] Alpha-Komplex als Default, Vietoris-Rips optional
- [x] Statische Betti-Zahlen, Betti-Kurven und Persistenz spaeter
- [x] Strukturkatalog: Kreis, Sphaere, Torus, Genus 2/3, disjunkte Kopien,
      verkettete Ringe/Acht, Klein bottle/Moebius/Annulus, Rauschen
- [x] "Doppeltorus" ist im ersten Test eine zusammenhaengende Genus-2-Flaeche
- [x] Weitere, frei definierbare Strukturen sind ausdruecklich erwuenscht
- [x] Phase 0 zuerst: Basis, Tests, CI
- [x] Ziel ist eine feste, in der GUI konfigurierbare Liste numerischer
      Zielgruppen, getrennt durch `;`; Werte innerhalb einer Gruppe durch `,`
- [x] Default-Zielgruppen: `betti`, `ph`, `genus`, `n_components`,
      `filtration`
- [x] Kein `=` im Trainingsstring; ein explizites Grenz-Token trennt
      Eingabe und Ziel
- [x] Training mit Cross-Entropy, Ranking und Multiple-Choice parallel
- [x] Modell soll von Anfang an sowohl inferieren als auch Punktwolken
      generieren
- [x] Python-Snippets lokal und explizit, aber mit sicheren Defaults
- [x] Eigene Strukturen per Preset, Formel/SDF, Python-Ausdruck, Mesh- und
      Punktwolken-Upload
- [x] Oberflaechen-Sampling als Default
- [x] Keine harten Compute-Limits, aber Warnungen und konservative Defaults
- [x] Transformationen werden dem Modell nicht mitgeteilt; Invarianz wird
      getestet
- [x] Leichte und strenge OOD-Invarianztests parallel
- [x] Alpha/Rips mit Parametern in der ersten Version; eigene Filtration
      spaeter
- [x] Z2 und rationale Koeffizienten vergleichen
- [x] UI komplett Englisch

## Noch zu praezisieren

- [ ] Exakter Name und Verhalten des Grenz-Tokens (`<eot>` als Default)
- [ ] Genaue Spaltenbelegung der Zielgruppen `filtration` und `structure_id`
- [ ] Ob Struktur-IDs numerisch oder alphanumerisch tokenisiert werden
- [ ] Default fuer maximale Homologiedimension (vorlaeufig 2)
- [ ] Grenzen fuer Warnungen statt harter Limits
- [ ] Konkrete Default-Modellgroesse fuer CPU

## Phase 0 - Discovery und Basis

- [~] README mit wissenschaftlichem Rahmen
- [ ] Offene Fragen beantworten
- [ ] Projektgeruest mit `pyproject.toml`
- [ ] Tokenizer + Tests
- [ ] Topologie-Generatoren + Tests
- [ ] Transformationen + Tests
- [ ] Filtration (Alpha/Rips) + Tests
- [ ] Bootstrap `python3 app.py` + Smoke-Test
- [ ] Minimales Web-UI mit Datenvorschau
- [ ] CI-Workflow

## Phase 1 - Modell und Training

- [ ] Eigener Transformer
- [ ] Kontrollierbare Initialisierung
- [ ] Autoregressive Trainingsschleife
- [ ] Auxiliaere Koepfe
- [ ] Checkpoints, Resume, Seeds
- [ ] Modelltests

## Phase 2 - Web-GUI und Live-Training

- [ ] FastAPI/WebSocket
- [ ] Data Studio
- [ ] Model Lab
- [ ] Training-Dashboard
- [ ] Live-Punktwolken und Trainingskurven

## Phase 3 - Interpretierbarkeit

- [ ] Lineare Probes pro Layer
- [ ] Aktivierungs- und Neuron-Heatmaps
- [ ] Attention-Maps
- [ ] Latent-Projektionen (PCA/UMAP)
- [ ] Neuron Explorer

## Phase 4 - Invarianz und Kausalitaet

- [ ] OOD-Transformationsbenchmarks
- [ ] Ablation einzelner Neuronen/Heads
- [ ] Activation Patching
- [ ] Invarianzmatrix und Fehleranalyse

## Phase 5 - Filtration als Lernziel und Politur

- [ ] Betti-Kurven als Ziel
- [ ] Persistenzdiagramme/Images
- [ ] Benutzerdefinierte Filtrationen
- [ ] Equation Explorer mit Temml
- [ ] Visueller Feinschliff
