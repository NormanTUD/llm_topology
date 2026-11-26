# Projekt-TODO

## Aktueller Stand

- **Aktuelle Phase:** Phase 3 - Invarianz und Kausalitaet
- **Phase 0:** abgeschlossen (Data Foundation)
- **Phase 1:** abgeschlossen (Custom Transformer + Training)
- **Phase 2:** abgeschlossen (Probes + Heatmaps + Latent-PCA)
- **Status:** Browser-GUI mit Data Studio, Model Lab und Interpretability
- **Letzter Testlauf:** 91 Tests bestanden, Ruff und Formatierung sauber
- **Naechster Meilenstein:** OOD-/Invarianzmatrix, Ablation, Activation Patching

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

- [x] Eigener Decoder-only-Transformer
- [x] Kontrollierbare Initialisierung
- [x] Token-Embedding und Ausgabekopf
- [x] Multi-Task-Ziele: Inferenz und Generierung
- [x] Auxiliaere Koepfe
- [x] Cross-Entropy, Ranking, Multiple-Choice
- [x] Optimizer, LR, Batch, Split, Early Stopping
- [x] Checkpoints, Resume, Seeds
- [x] Live-Training im Browser
- [x] Modelltests und Shape-Tests
- [x] Default-Modellgroesse fuer CPU festlegen
- [x] Kontextlaenge, Koordinatenpraezision und Punktbudget kalibrieren
- [x] CPU/CUDA/MPS-Erkennung
- [x] Batch-Training mit N Punktwolken
- [x] Verzerrungen (Shear/Bend/Warp/Twist) + Augmentationsstaerke
- [x] Run-Ordner `runs/exp-NNNN` und CLI-Logging
- [x] Live-Validierungspaare mit Inputwolke und true/predicted Betti
- [x] Frische Multi-Cloud-Predictions im Browser
- [x] Start/Stop-Toggle
- [x] Single-Page-App ohne Reload
- [x] Server-seitiger UI-Zustand + F5-Restore
- [x] Referenzwolke nur fuer die GUI, nicht fuer das Modell
- [x] Batch-Modus single/mix mit expliziter Strukturauswahl
- [x] Live-Update der Geometrie-Regler
- [x] Persistente Datasets `datasets/ds-NNNN` + Auswahl im Model Lab
- [x] Doppelte Dataset-Regler im Model Lab entfernt
- [x] Augmentierungs-TypeError behoben und getestet
- [x] Topology Guard und Normalisierung fuer stabile Filtrationsskalen
- [x] Getrennte Train-/Val-Transformationen fuer gespeicherte Datasets
- [x] Mehrere Tori/Kopien mit stabilen b0-Labels
- [x] Auto- und manuelles Epsilon in Dataset-Erzeugung
- [x] Filtration wird als Zielgruppe mitgelernt und in Predictions angezeigt
- [x] GUI-Smoke-Tests: 31 Checks fuer Tabs, Galerie, Dataset-Builder, Training, Analyse, F5
- [x] GUI-Test fuer Stop-Button und Token-Budget-Guard

## Phase 2 - Interpretierbarkeit

- [x] Lineare Probes pro Layer
- [x] Probe-Gewichts-Heatmaps
- [x] Aktivierungs- und Neuron-Heatmaps
- [x] Attention-Maps
- [x] FFN-Neuronen-Explorer
- [x] Latent-Projektionen (PCA; UMAP spaeter)
- [ ] Latente Trajektorien ueber Trainingsschritte (Loss-Historie ist da)

## Phase 3 - Invarianz und Kausalitaet (aktuell)

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

## Entschieden fuer die naechste Ausbaustufe

- [x] Val-Transformationen: drei Modi `clean`, `same`, `heldout`
- [x] Punkte: sowohl Gesamtpunkte als auch Punkte pro Komponente anbieten
- [x] Filtration: Epsilon als Input plus Auto-Modus
- [x] Generierte Punktwolken: echtes 3D nebeneinander (Input vs. generiert)
- [x] Dataset-Bibliothek: duplizieren, loeschen, exportieren, importieren
- [x] Naechster grosser Fokus: Phase 3 (OOD/Invarianz, Ablation, Patching)

## Offene Phase-3-Entscheidungen

- [ ] Ablationsachse: Neuronen, Heads, Layer oder Residualrichtungen?
- [ ] Activation Patching zwischen Strukturen, Transformationen oder Labels?
- [ ] Pflicht-OOD-Achsen: Rotation, Stretch, Twist, Rauschen, Punktanzahl,
      Ambient-Dimension, ungesehene Genus-Werte?
- [ ] Invarianzmatrix pro Struktur oder pro Betti-Komponente?
- [ ] Generierte Punktwolken automatisch topologisch auswerten?
- [ ] Exaktes Q: rationale Gauss-Elimination, Gudhi oder externe Bibliothek?
- [ ] Welche Gleichungen werden zuerst interaktiv?
