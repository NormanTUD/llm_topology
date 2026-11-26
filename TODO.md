# Projekt-TODO

## Aktueller Stand

- **Aktuelle Phase:** Phase 1 - Custom Transformer und Training
- **Phase 0:** abgeschlossen (Data Foundation)
- **Status:** Basis, GUI, Filtration, Tests und CI laufen
- **Letzter Testlauf:** 65 Tests bestanden, Ruff und Formatierung sauber
- **Naechster Meilenstein:** eigener Transformer, Multi-Task-Training,
  Checkpoints und erste Probes

Legende: `[ ]` offen, `[~]` in Arbeit, `[x]` erledigt

## Bereits entschieden

- [x] Vollstaendig eigenes Modell, kein Pretraining, keine GPT-2-Gewichte
- [x] Automatischer Bootstrap ueber uv
- [x] Browser-GUI mit Live-Visualisierungen
- [x] TDD und CI von Anfang an
- [x] Tokenizer mit einzelnen Ziffern, `.` `,` `;` `(` `)` `=` `-`,
      `betti`, `<bos>`, `<eos>`, `<endoftext>`, `<infer>`, `<generate>`
- [x] Autoregressive Ausgabe plus auxiliaere Koepfe
- [x] Ziffern-Tokenizer als Default, numerische Embeddings als Baseline spaeter
- [x] Generator- und PH-Labels getrennt vorsehen
- [x] Alpha-Komplex als Default, Vietoris-Rips optional
- [x] Statische Betti-Zahlen jetzt, Betti-Kurven und Persistenz als Ziele spaeter
- [x] Strukturkatalog: Kreis, Sphaere, Torus, Genus 2/3, disjunkte Kopien,
      verkettete Ringe, Acht, Klein bottle, Moebiusband, Annulus, Rauschen,
      Vollkugel
- [x] "Doppeltorus" als zusammenhaengende Genus-2-Flaeche, plus freie Strukturen
- [x] Phase 0 zuerst: Basis, Tests, CI
- [x] Feste, in der GUI konfigurierbare numerische Zielgruppen
- [x] Optionale Zielgruppen `structure_id`, `n_points`, `ambient_dim`
- [x] Kein `=`: `<endoftext>` trennt Eingabe und Ziel
- [x] Default-Zielgruppen: `betti`, `ph`, `genus`, `n_components`,
      `filtration`
- [x] Training mit Cross-Entropy, Ranking und Multiple-Choice parallel geplant
- [x] Modell soll inferieren und Punktwolken generieren
- [x] Python-Snippets lokal und explizit, mit sicheren Defaults
- [x] Eigene Strukturen per Preset, Formel/SDF, Python-Ausdruck, Mesh- und
      Punktwolken-Upload geplant
- [x] Oberflaechen-Sampling als Default
- [x] Keine harten Compute-Limits, aber Warnungen und konservative Defaults
- [x] Transformationen werden nicht mitgeteilt; Invarianz wird getestet
- [x] Leichte und strenge OOD-Invarianztests parallel geplant
- [x] Alpha/Rips mit Parametern; eigene Filtrationen spaeter
- [x] Z2 und Z3 als Q-Proxy vergleichen; exaktes Q spaeter
- [x] UI komplett Englisch

## Phase 0 - Data Foundation

- [x] README mit wissenschaftlichem Rahmen
- [x] Projektgeruest mit `pyproject.toml`
- [x] uv-Bootstrap `python3 app.py`
- [x] Tokenizer + Tests
- [x] Zielgruppen-Schema + Parser + Tests
- [x] Topologie-Generatoren + Tests
- [x] Transformationen + Tests
- [x] Filtration Alpha/Rips + Tests
- [x] Automatische Epsilon-Heuristik
- [x] Betti-Kurven und Persistenzdiagramme
- [x] FastAPI + Web-GUI mit Data Studio
- [x] 3D-Punktwolke, Persistenzplot, Betti-Kurven
- [x] Temml-Gleichungen mit Live-Werten
- [x] Graceful Fehlerbehandlung
- [x] CI-Workflow
- [x] uv.lock

## Phase 1 - Modell und Training

- [ ] Eigener Decoder-only-Transformer
- [ ] Kontrollierbare Initialisierung
- [ ] Token-Embedding und Ausgabekopf
- [ ] Multi-Task-Ziele: Inferenz und Generierung
- [ ] Auxiliaere Koepfe
- [ ] Cross-Entropy, Ranking, Multiple-Choice
- [ ] Optimizer, LR, Batch, Split, Early Stopping
- [ ] Checkpoints, Resume, Seeds
- [ ] Live-Training im Browser
- [ ] Modelltests und Shape-Tests
- [ ] Default-Modellgroesse fuer CPU festlegen
- [ ] Kontextlaenge, Koordinatenpraezision und Punktbudget kalibrieren
- [ ] CPU/CUDA/MPS-Erkennung

## Phase 2 - Interpretierbarkeit

- [ ] Lineare Probes pro Layer
- [ ] Probe-Gewichts-Heatmaps
- [ ] Aktivierungs- und Neuron-Heatmaps
- [ ] Attention-Maps
- [ ] FFN-Neuronen-Explorer
- [ ] Latent-Projektionen (PCA/UMAP)
- [ ] Trajektorien ueber Trainingsschritte

## Phase 3 - Invarianz und Kausalitaet

- [ ] OOD-Transformationsbenchmarks
- [ ] Leichte und strenge Invarianzmatrix
- [ ] Ablation einzelner Neuronen/Heads
- [ ] Activation Patching
- [ ] Fehleranalyse und Candidate Negatives

## Phase 4 - Filtration als Lernziel und Politur

- [ ] Epsilon als Modell-Input
- [ ] Betti-Kurven als Ziel
- [ ] Persistenzdiagramme/Images als Ziel
- [ ] Exaktes Q fuer kleine Komplexe
- [ ] Benutzerdefinierte Filtrationen
- [ ] Mesh-/Punktwolken-Upload
- [ ] Formel-/SDF-Eingabe
- [ ] Python-Ausdruck/Snippet
- [ ] Interaktiver Equation Explorer
- [ ] Visueller Feinschliff

## Offene Phase-1-Entscheidungen

- [ ] Generierung zuerst als Sequenzmodell oder separater Decoder-Kopf?
- [ ] Konkrete Default-Modellgroesse fuer CPU?
- [ ] `structure_id` numerisch oder alphanumerisch?
- [ ] Zielgruppen-Reihenfolge pro Run fixiert oder permutierbar?
- [ ] Exaktes Q: rationale Gauss-Elimination, Gudhi oder externe Bibliothek?
- [ ] Python-Snippets: AST-Whitelist oder vertrauenswuerdiger Subprozess?
- [ ] Welche Gleichungen werden zuerst interaktiv?
