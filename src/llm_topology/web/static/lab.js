import { applyById, collectById, loadUIState, saveUIState } from "./state.js";

const labState = {
  catalog: null,
  jobId: null,
  job: null,
  pollTimer: null,
  analysis: null,
  runs: [],
  datasets: [],
  datasetId: null,
  view: "train",
  running: false,
  stopping: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const COLORS = ["#5eead4", "#fb923c", "#a78bfa", "#f472b6", "#60a5fa", "#fbbf24"];

function numberValue(id, fallback = 0) {
  const element = document.getElementById(id);
  if (!element) return fallback;
  const value = Number(element.value);
  return Number.isFinite(value) ? value : fallback;
}

function intValue(id, min, max, fallback) {
  const value = Number.parseInt(document.getElementById(id)?.value, 10);
  if (Number.isNaN(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function setError(message) {
  const banner = $("#lab-error");
  if (!banner) return;
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function clearError() {
  const banner = $("#lab-error");
  if (!banner) return;
  banner.classList.add("hidden");
  banner.textContent = "";
}

function toast(message, isError = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.classList.add("show");
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => element.classList.remove("show"), 2600);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_error) {
      // Keep status text.
    }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function plotlyReady() {
  return typeof window.Plotly !== "undefined";
}

function plotlyLayout(extra = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: "#8b9bb4", size: 10 },
    margin: { l: 46, r: 18, t: 18, b: 42 },
    hoverlabel: {
      bgcolor: "#0b1220",
      bordercolor: "rgba(94,234,212,.28)",
      font: { color: "#e9eff9", family: "JetBrains Mono, monospace", size: 10 },
    },
    ...extra,
  };
}

function plotConfig() {
  return { responsive: true, displayModeBar: false };
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

async function loadCatalog() {
  labState.catalog = await fetchJSON("/api/catalog");
  $("#phase-chip").textContent = labState.catalog.phase;
  $("#version-chip").textContent = `v${labState.catalog.version}`;
  const defaultData = labState.catalog.data_defaults;
  const defaultModel = labState.catalog.model_defaults;
  const defaultTrain = labState.catalog.training_defaults;

  setValue("context-train", defaultData.max_seq_len);

  setValue("n-layers", defaultModel.n_layers);
  setValue("n-heads", defaultModel.n_heads);
  setValue("d-model", defaultModel.d_model);
  setValue("d-ff", defaultModel.d_ff);
  setValue("dropout", defaultModel.dropout);
  setValue("activation", defaultModel.activation);
  setValue("norm-placement", defaultModel.norm_placement);
  setValue("position-encoding", defaultModel.position_encoding);
  setValue("init-scheme", defaultModel.init.scheme);
  setValue("init-std", defaultModel.init.std);
  setValue("init-seed", defaultModel.init.seed);
  $("#tie-embeddings").checked = Boolean(defaultModel.tie_embeddings);
  $("#use-bias").checked = Boolean(defaultModel.use_bias);

  setValue("epochs", defaultTrain.epochs);
  setValue("batch-size", defaultTrain.batch_size);
  setValue("lr", defaultTrain.lr);
  setValue("weight-decay", defaultTrain.weight_decay);
  setValue("optimizer", defaultTrain.optimizer);
  setValue("warmup", defaultTrain.warmup_steps);
  setValue("grad-clip", defaultTrain.grad_clip);
  setValue("patience", defaultTrain.patience);
  setValue("train-seed", defaultTrain.seed);
  setValue("aux-weight", defaultTrain.aux_weight);
  setValue("ranking-weight", defaultTrain.ranking_weight);
  setValue("mc-negatives", defaultTrain.mc_negatives);
  setValue("max-new-tokens", defaultTrain.max_new_tokens);
  setValue("device", defaultTrain.device);

  const freshSelect = $("#fresh-structure");
  if (freshSelect) {
    freshSelect.innerHTML = "";
    for (const structure of labState.catalog.structures) {
      if (structure.value === "disjoint") continue;
      const option = document.createElement("option");
      option.value = structure.value;
      option.textContent = structure.label;
      freshSelect.appendChild(option);
    }
    freshSelect.value = "torus";
  }

  setValue("analysis-examples", labState.catalog.analysis_defaults.max_examples);
  setValue("probe-alpha", labState.catalog.analysis_defaults.probe_alpha);
  setValue("attention-example", labState.catalog.analysis_defaults.attention_example);
  setValue("pca-dim", labState.catalog.analysis_defaults.pca_dim);
  setValue("top-neurons", labState.catalog.analysis_defaults.top_neurons);
}

function setValue(id, value) {
  const element = document.getElementById(id);
  if (element) element.value = String(value);
}

function persistLabState() {
  saveUIState({
    labForm: collectById(".lab-sidebar"),
    freshForm: collectById(".fresh-controls"),
    datasetId: labState.datasetId,
    jobId: labState.jobId,
  });
}

function schemaForMaxDim(maxDim) {
  const base = labState.catalog.data_defaults.target_schema;
  return {
    blocks: base.blocks.map((block) => {
      if (block.name === "betti" || block.name === "ph") {
        return {
          name: block.name,
          fields: Array.from({ length: maxDim + 1 }, (_, index) => `b${index}`),
        };
      }
      return { name: block.name, fields: block.fields };
    }),
  };
}

function buildSpec() {
  const datasetId = $("#dataset-select").value || labState.datasetId;
  if (!datasetId) throw new Error("Select a saved dataset first.");
  const context = intValue("context-train", 64, 32768, 1024);
  return {
    dataset_id: datasetId,
    model: {
      max_seq_len: context,
      n_layers: intValue("n-layers", 1, 24, 3),
      n_heads: intValue("n-heads", 1, 16, 4),
      d_model: intValue("d-model", 8, 1024, 96),
      d_ff: intValue("d-ff", 8, 4096, 384),
      dropout: Math.max(0, Math.min(0.8, numberValue("dropout", 0))),
      activation: $("#activation").value,
      norm_placement: $("#norm-placement").value,
      position_encoding: $("#position-encoding").value,
      tie_embeddings: $("#tie-embeddings").checked,
      use_bias: $("#use-bias").checked,
      init: {
        scheme: $("#init-scheme").value,
        std: Math.max(0.0001, numberValue("init-std", 0.02)),
        mean: 0,
        seed: intValue("init-seed", -2147483648, 2147483647, 0),
      },
    },
    training: {
      epochs: intValue("epochs", 1, 1000, 6),
      batch_size: intValue("batch-size", 1, 512, 8),
      lr: Math.max(0.000001, numberValue("lr", 0.0003)),
      weight_decay: Math.max(0, numberValue("weight-decay", 0.01)),
      optimizer: $("#optimizer").value,
      warmup_steps: intValue("warmup", 0, 100000, 20),
      grad_clip: Math.max(0, numberValue("grad-clip", 1)),
      patience: intValue("patience", 1, 1000, 3),
      seed: intValue("train-seed", -2147483648, 2147483647, 7),
      aux_weight: Math.max(0, numberValue("aux-weight", 0.2)),
      ranking_weight: Math.max(0, numberValue("ranking-weight", 0.05)),
      ranking_margin: 0.5,
      mc_negatives: intValue("mc-negatives", 1, 16, 2),
      device: $("#device").value,
      checkpoint_root: "runs",
      max_new_tokens: intValue("max-new-tokens", 1, 2048, 80),
      eval_every_batches: 20,
    },
  };
}

async function startTraining() {
  clearError();
  const button = $("#train-toggle");
  button.disabled = true;
  button.classList.add("loading");
  button.querySelector(".btn-label").textContent = "Preparing dataset…";
  try {
    const spec = buildSpec();
    const job = await fetchJSON("/api/train/start", {
      method: "POST",
      body: JSON.stringify(spec),
    });
    labState.jobId = job.id;
    labState.job = job;
    labState.analysis = null;
    labState.running = true;
    labState.stopping = false;
    persistLabState();
    renderJob(job);
    schedulePoll(250);
    toast(`Run ${job.id} started`);
  } catch (error) {
    setError(error.message);
    toast(error.message, true);
    labState.running = false;
    setTrainButton(false);
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
  }
}

async function stopTraining() {
  if (!labState.jobId) return;
  const button = $("#train-toggle");
  labState.stopping = true;
  button.disabled = true;
  button.classList.add("danger");
  button.querySelector(".btn-label").textContent = "Stopping…";
  try {
    await fetchJSON(`/api/train/stop/${labState.jobId}`, { method: "POST" });
    toast("Stop requested");
  } catch (error) {
    labState.stopping = false;
    toast(error.message, true);
    button.disabled = false;
    setTrainButton(labState.running);
  }
}

function toggleTraining() {
  if (labState.running) stopTraining();
  else startTraining();
}

function setTrainButton(running) {
  labState.running = running;
  const button = $("#train-toggle");
  if (!button) return;
  button.classList.toggle("danger", running);
  button.querySelector(".btn-label").textContent = running ? "Stop training" : "Start training";
  button.disabled = false;
}

function schedulePoll(delay = 900) {
  window.clearTimeout(labState.pollTimer);
  labState.pollTimer = window.setTimeout(pollStatus, delay);
}

async function pollStatus() {
  if (!labState.jobId) return;
  try {
    const job = await fetchJSON(`/api/train/status/${labState.jobId}`);
    labState.job = job;
    renderJob(job);
    if (["completed", "early_stopped", "stopped", "failed"].includes(job.status)) {
      window.clearTimeout(labState.pollTimer);
      labState.stopping = false;
      setTrainButton(false);
      refreshRuns();
      return;
    }
    schedulePoll(900);
  } catch (error) {
    setError(error.message);
    schedulePoll(2500);
  }
}

function renderJob(job) {
  const running = ["queued", "preparing", "running"].includes(job.status);
  if (running && labState.stopping) {
    const button = $("#train-toggle");
    button.disabled = true;
    button.classList.add("danger");
    button.querySelector(".btn-label").textContent = "Stopping…";
  } else {
    if (!running) labState.stopping = false;
    setTrainButton(running);
  }
  $("#run-status").textContent = job.status;
  $("#run-id").textContent = job.id ? `run ${job.id}` : "no run";
  $("#best-val").textContent = formatNumber(job.best_val_loss, 4);
  $("#best-val-note").textContent =
    job.best_val_loss !== null && job.best_val_loss !== undefined
      ? "lower is better"
      : "waiting for validation";
  $("#param-count").textContent = job.parameter_count
    ? Number(job.parameter_count).toLocaleString()
    : "—";
  $("#param-note").textContent = job.device ? `${job.device} · custom transformer` : "custom transformer";
  $("#device-chip").textContent = job.device || "cpu";
  const current = job.current || {};
  const total = current.total_steps || 0;
  const step = current.step || 0;
  const progress = total > 0 ? Math.min(100, (step / total) * 100) : 0;
  $("#run-progress").textContent = `${progress.toFixed(0)}%`;
  $("#progress-note").textContent = current.epoch
    ? `epoch ${current.epoch}${total ? ` · step ${step}/${total}` : ""}`
    : "preparing";
  $("#train-step").textContent = total ? `step ${step}/${total}` : "preparing";
  if (job.error) setError(job.error);
  if (job.samples) {
    $("#sample-infer").textContent = job.samples.infer || "—";
    $("#sample-generate").textContent = job.samples.generate || "—";
  }
  renderLoss(job);
  renderValMetrics(job);
  renderPredictionPairs(job);
  populateAnalysisJob(job);
}

function renderLoss(job) {
  const element = $("#loss-plot");
  if (!plotlyReady()) return;
  element.innerHTML = "";
  const history = job.history || [];
  const batchHistory = job.loss_history || [];
  const current = job.current || {};
  const batchSteps = batchHistory.map((entry) => entry.step);
  const batchLosses = batchHistory.map((entry) => entry.loss);
  const batchCe = batchHistory.map((entry) => entry.ce);
  const epochSteps = history.map((entry) => entry.step);
  const trainLoss = history.map((entry) => entry.train_loss);
  const valLoss = history.map((entry) => entry.val_loss);
  const traces = [];
  if (batchHistory.length) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "batch loss",
      x: batchSteps,
      y: batchLosses,
      line: { color: "#fb923c", width: 1.6 },
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "batch CE",
      x: batchSteps,
      y: batchCe,
      line: { color: "#a78bfa", width: 1, dash: "dot" },
    });
  }
  if (history.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: "epoch train loss",
      x: epochSteps,
      y: trainLoss,
      line: { color: "#fdba74", width: 1.4 },
      marker: { size: 4 },
    });
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: "epoch val loss",
      x: epochSteps,
      y: valLoss,
      line: { color: "#5eead4", width: 2.2 },
      marker: { size: 6 },
    });
  }
  if (!traces.length && current.loss) {
    traces.push({
      type: "scatter",
      mode: "markers",
      name: "live",
      x: [current.step || 0],
      y: [current.loss],
      marker: { color: "#f472b6", size: 7 },
    });
  }
  if (!traces.length) {
    element.innerHTML = '<div class="plot-placeholder">Start a run to see live loss.</div>';
    return;
  }
  const layout = plotlyLayout({
    showlegend: true,
    legend: { orientation: "h", y: 1.08, x: 0, font: { size: 9 } },
    xaxis: { title: "step", gridcolor: "rgba(148,163,184,.08)", zeroline: false },
    yaxis: { title: "loss", gridcolor: "rgba(148,163,184,.08)", zeroline: false },
  });
  window.Plotly.newPlot(element, traces, layout, plotConfig());
}

function renderPredictionPairs(job) {
  const container = $("#prediction-pairs");
  const predictions =
    (job.current && job.current.prediction_samples) ||
    (job.history && job.history.length
      ? job.history[job.history.length - 1].prediction_samples
      : null) ||
    [];
  container.innerHTML = "";
  if (!predictions.length) {
    container.innerHTML = '<span class="muted">Validation pairs appear after the first epoch.</span>';
    return;
  }
  for (const prediction of predictions) {
    const card = document.createElement("div");
    card.className = "prediction-card";
    const head = document.createElement("div");
    head.className = "prediction-head";
    const name = document.createElement("strong");
    name.textContent = `${prediction.task} · ${prediction.structure_name}`;
    const status = document.createElement("span");
    const correct = prediction.predicted_betti && prediction.true_betti &&
      prediction.predicted_betti.slice(0, 3).join(",") === prediction.true_betti.slice(0, 3).join(",");
    status.textContent = prediction.task === "infer" ? (correct ? "match" : "readout") : "generation";
    status.style.color = correct ? "var(--cyan)" : "var(--muted)";
    head.appendChild(name);
    head.appendChild(status);
    card.appendChild(head);

    if (prediction.input_projection && prediction.input_projection.length) {
      const mini = document.createElement("div");
      mini.className = "prediction-mini";
      card.appendChild(mini);
      window.requestAnimationFrame(() => plotMiniPoints(mini, prediction.input_projection));
    }
    if (prediction.generated_projection && prediction.generated_projection.length) {
      const mini = document.createElement("div");
      mini.className = "prediction-mini";
      mini.style.borderTop = "1px solid var(--line)";
      card.appendChild(mini);
      window.requestAnimationFrame(() => plotMiniPoints(mini, prediction.generated_projection));
    }

    const values = document.createElement("div");
    values.className = "prediction-values";
    if (prediction.task === "infer") {
      const trueBetti = prediction.true_betti || [];
      const predBetti = prediction.predicted_betti || [];
      const trueRow = document.createElement("div");
      trueRow.className = "true";
      trueRow.innerHTML = `<span class="label">true</span> b=(${trueBetti.slice(0, 3).join(",")})`;
      const predRow = document.createElement("div");
      predRow.className = `predicted${correct ? "" : " bad"}`;
      predRow.innerHTML = `<span class="label">pred</span> b=(${predBetti.slice(0, 3).join(",")})`;
      values.appendChild(trueRow);
      values.appendChild(predRow);
      const trueFiltration = prediction.true_filtration || [];
      const predictedFiltration = prediction.predicted_filtration || [];
      if (trueFiltration.length || predictedFiltration.length) {
        const filt = document.createElement("div");
        filt.className = "predicted";
        filt.innerHTML = `<span class="label">filtration ε</span> true ${trueFiltration[2] ?? "—"} · pred ${predictedFiltration[2] ?? "—"}`;
        values.appendChild(filt);
      }
      if (prediction.predicted_text) {
        const text = document.createElement("div");
        text.className = "prediction-condition";
        text.textContent = prediction.predicted_text;
        values.appendChild(text);
      }
    } else {
      const condition = document.createElement("div");
      condition.className = "prediction-condition";
      condition.textContent = `condition: ${prediction.condition_text}`;
      values.appendChild(condition);
      const generated = document.createElement("div");
      generated.className = "prediction-condition";
      generated.textContent = `generated: ${prediction.generated_text}`;
      values.appendChild(generated);
      if (prediction.generated_betti) {
        const topology = document.createElement("div");
        topology.className = "predicted";
        topology.innerHTML = `<span class="label">generated PH</span> b=(${prediction.generated_betti.slice(0, 3).join(",")})`;
        values.appendChild(topology);
      }
    }
    card.appendChild(values);
    container.appendChild(card);
  }
}

function plotMiniPoints(element, points) {
  if (!plotlyReady() || !points || !points.length) return;
  const trace = {
    type: "scattergl",
    mode: "markers",
    x: points.map((point) => point[0]),
    y: points.map((point) => point[1]),
    marker: { size: 2.2, color: points.map((_, index) => index), colorscale: [[0, "#60a5fa"], [0.5, "#5eead4"], [1, "#fb923c"]], opacity: 0.85 },
    hoverinfo: "skip",
  };
  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { visible: false, fixedrange: true },
    yaxis: { visible: false, fixedrange: true, scaleanchor: "x", scaleratio: 1 },
    showlegend: false,
  };
  window.Plotly.newPlot(element, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderValMetrics(job) {
  const current = job.current || {};
  const rows = [
    ["val loss", current.val_loss],
    ["val CE", current.val_ce],
    ["token accuracy", current.val_token_accuracy],
    ["multiple choice", current.val_multiple_choice],
    ["aux loss", current.val_aux_loss],
    ["train loss", current.train_loss],
  ];
  const container = $("#val-metrics");
  container.innerHTML = "";
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "metric-row";
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("span");
    if (label.includes("accuracy") || label.includes("choice")) {
      number.textContent = formatPercent(value);
      number.className = value > 0.5 ? "good" : "";
    } else {
      number.textContent = formatNumber(value, 4);
    }
    row.appendChild(name);
    row.appendChild(number);
    container.appendChild(row);
  }
}

function populateAnalysisJob(job) {
  if (!job || !job.id) return;
  const select = $("#analysis-job");
  const existing = Array.from(select.options).find((option) => option.value === job.id);
  if (!existing) {
    const option = document.createElement("option");
    option.value = job.id;
    option.textContent = `${job.id} · ${job.status}`;
    select.appendChild(option);
  }
  select.value = job.id;
}

async function refreshRuns() {
  try {
    const runs = await fetchJSON("/api/train/list");
    labState.runs = runs;
    renderRuns(runs);
    const analysisSelect = $("#analysis-job");
    analysisSelect.innerHTML = "";
    const completed = runs.filter((run) => run.best_checkpoint);
    if (!completed.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No completed run";
      analysisSelect.appendChild(option);
    }
    for (const run of completed) {
      const option = document.createElement("option");
      option.value = run.id;
      option.textContent = `${run.id} · ${run.status} · best ${formatNumber(run.best_val_loss, 3)}`;
      analysisSelect.appendChild(option);
    }
    if (labState.jobId) analysisSelect.value = labState.jobId;
  } catch (error) {
    setError(error.message);
  }
}

function renderRuns(runs) {
  const container = $("#run-list");
  container.innerHTML = "";
  if (!runs.length) {
    container.innerHTML = '<span class="muted">No runs yet.</span>';
    return;
  }
  for (const run of runs.slice().reverse()) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `run-item${run.id === labState.jobId ? " active" : ""}`;
    const main = document.createElement("div");
    main.className = "run-item-main";
    const title = document.createElement("strong");
    title.textContent = run.id;
    const note = document.createElement("span");
    note.textContent = `${run.parameter_count || "?"} params · best ${formatNumber(run.best_val_loss, 3)} · ${run.device || "?"}`;
    main.appendChild(title);
    main.appendChild(note);
    const status = document.createElement("span");
    status.className = `run-item-status${run.status === "failed" ? " failed" : ""}`;
    status.textContent = run.status;
    item.appendChild(main);
    item.appendChild(status);
    item.addEventListener("click", () => {
      labState.jobId = run.id;
      labState.analysis = null;
      persistLabState();
      pollStatus();
      renderRuns(labState.runs);
    });
    container.appendChild(item);
  }
}

async function runAnalysis() {
  const jobId = $("#analysis-job").value || labState.jobId;
  if (!jobId) {
    toast("Select a run with a checkpoint first", true);
    return;
  }
  clearError();
  const button = $("#run-analysis");
  button.disabled = true;
  button.classList.add("loading");
  button.querySelector(".btn-label").textContent = "Running probes…";
  try {
    const spec = {
      max_examples: intValue("analysis-examples", 8, 1024, 96),
      probe_alpha: Math.max(0, numberValue("probe-alpha", 1)),
      attention_example: intValue("attention-example", 0, 1024, 0),
      pca_dim: intValue("pca-dim", 2, 3, 3),
      top_neurons: intValue("top-neurons", 1, 128, 12),
    };
    const analysis = await fetchJSON(`/api/analyze/${jobId}`, {
      method: "POST",
      body: JSON.stringify(spec),
    });
    labState.analysis = analysis;
    labState.jobId = jobId;
    renderAnalysis(analysis);
    setTab("interpret");
    toast("Probe analysis complete");
  } catch (error) {
    setError(error.message);
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
    button.querySelector(".btn-label").textContent = "Run probes & maps";
  }
}

function renderAnalysis(analysis) {
  const summary = analysis.summary;
  $("#probe-structure").textContent = formatPercent(summary.best_structure_accuracy);
  $("#probe-betti").textContent = formatNumber(summary.best_betti_r2, 3);
  $("#probe-layers").textContent = summary.layers;
  $("#probe-examples").textContent = `${summary.examples_train} / ${summary.examples_val}`;
  populateLayerSelects(analysis);
  renderProbeScores(analysis);
  renderProbeHeatmap(analysis);
  renderTopNeurons(analysis);
  renderLatent(analysis);
  renderNeuronMap(analysis);
  renderAttention(analysis);
  renderFFN(analysis);
}

function populateLayerSelects(analysis) {
  const layers = analysis.probes.map((probe) => probe.layer);
  for (const id of ["probe-layer", "latent-layer", "ffn-layer"]) {
    const select = document.getElementById(id);
    select.innerHTML = "";
    for (const layer of layers) {
      const option = document.createElement("option");
      option.value = String(layer);
      option.textContent = `Layer ${layer}`;
      select.appendChild(option);
    }
    select.value = String(layers[layers.length - 1] ?? 0);
  }
  const heads = analysis.attention?.attention?.[0]?.length || 1;
  const headSelect = $("#attention-head");
  headSelect.innerHTML = "";
  for (let head = 0; head < heads; head += 1) {
    const option = document.createElement("option");
    option.value = String(head);
    option.textContent = `Head ${head}`;
    headSelect.appendChild(option);
  }
}

function renderProbeScores(analysis) {
  if (!plotlyReady()) return;
  $("#probe-plot").innerHTML = "";
  const layers = analysis.probes.map((probe) => probe.layer);
  const structure = analysis.probes.map((probe) =>
    Number.isFinite(probe.structure_accuracy) ? probe.structure_accuracy : null
  );
  const betti = analysis.probes.map((probe) => {
    const values = Object.entries(probe.r2)
      .filter(([key, value]) => key.startsWith("betti") && Number.isFinite(value))
      .map(([, value]) => value);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  });
  const ph = analysis.probes.map((probe) => {
    const values = Object.entries(probe.r2)
      .filter(([key, value]) => key.startsWith("ph") && Number.isFinite(value))
      .map(([, value]) => value);
    return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  });
  const traces = [
    { name: "structure accuracy", x: layers, y: structure, type: "scatter", mode: "lines+markers", line: { color: "#5eead4", width: 2 } },
    { name: "Betti mean R²", x: layers, y: betti, type: "scatter", mode: "lines+markers", line: { color: "#fb923c", width: 2 } },
    { name: "PH mean R²", x: layers, y: ph, type: "scatter", mode: "lines+markers", line: { color: "#a78bfa", width: 2 } },
  ];
  const layout = plotlyLayout({
    showlegend: true,
    legend: { orientation: "h", y: 1.1, x: 0, font: { size: 9 } },
    xaxis: { title: "layer", dtick: 1, gridcolor: "rgba(148,163,184,.08)" },
    yaxis: { title: "score", gridcolor: "rgba(148,163,184,.08)", range: [-0.2, 1.05] },
  });
  window.Plotly.newPlot($("#probe-plot"), traces, layout, plotConfig());
}

function renderProbeHeatmap(analysis) {
  if (!plotlyReady()) return;
  $("#probe-heatmap").innerHTML = "";
  const layer = intValue("probe-layer", 0, 1024, analysis.probes.length - 1);
  const probe = analysis.probes.find((entry) => entry.layer === layer) || analysis.probes[0];
  const targets = Object.keys(probe.r2);
  const weights = probe.weights;
  const z = weights.map((row) => row.slice(0, 64));
  if (!z.length || !z[0]?.length) {
    $("#probe-heatmap").innerHTML = '<div class="plot-placeholder">No probe weights available.</div>';
    return;
  }
  const layout = plotlyLayout({
    margin: { l: 104, r: 12, t: 12, b: 34 },
    xaxis: { title: "neuron", gridcolor: "rgba(148,163,184,.06)" },
    yaxis: { title: "", automargin: true, tickfont: { size: 8 } },
  });
  window.Plotly.newPlot(
    $("#probe-heatmap"),
    [{ type: "heatmap", z, x: Array.from({ length: z[0]?.length || 0 }, (_, index) => index), y: targets, colorscale: "RdBu", zmid: 0, colorbar: { thickness: 8 } }],
    layout,
    plotConfig()
  );
}

function renderTopNeurons(analysis) {
  const layer = intValue("probe-layer", 0, 1024, analysis.probes.length - 1);
  const probe = analysis.probes.find((entry) => entry.layer === layer) || analysis.probes[0];
  const container = $("#top-neuron-list");
  container.innerHTML = "";
  const targets = ["betti.1", "ph.1", "genus"].filter((target) => probe.top_neurons[target]);
  if (!targets.length) {
    container.innerHTML = '<span class="muted">No probe weights for this layer.</span>';
    return;
  }
  for (const target of targets) {
    const row = document.createElement("div");
    row.className = "metric-row";
    const label = document.createElement("span");
    label.textContent = target;
    const value = document.createElement("span");
    value.textContent = probe.top_neurons[target]
      .slice(0, 4)
      .map((entry) => `${entry.index}:${entry.weight.toFixed(2)}`)
      .join("  ");
    row.appendChild(label);
    row.appendChild(value);
    container.appendChild(row);
  }
}

function renderLatent(analysis) {
  if (!plotlyReady()) return;
  $("#latent-plot").innerHTML = "";
  const layer = intValue("latent-layer", 0, 1024, analysis.probes.length - 1);
  const projection = analysis.projection;
  const coords = projection.coordinates[layer] || [];
  const names = projection.structure_names;
  const structureIds = projection.structure_ids;
  const dims = coords[0]?.length || 2;
  if (!coords.length) {
    $("#latent-plot").innerHTML = '<div class="plot-placeholder">No projection available.</div>';
    return;
  }
  const trace = {
    type: dims === 3 ? "scatter3d" : "scatter",
    mode: "markers",
    x: coords.map((point) => point[0]),
    y: coords.map((point) => point[1]),
    ...(dims === 3 ? { z: coords.map((point) => point[2]) } : {}),
    marker: {
      size: dims === 3 ? 4 : 7,
      color: structureIds,
      colorscale: [[0, "#60a5fa"], [0.35, "#5eead4"], [0.7, "#a78bfa"], [1, "#fb923c"]],
      opacity: 0.85,
    },
    text: names,
    hovertemplate: "%{text}<extra></extra>",
  };
  const layout = dims === 3
    ? plotlyLayout({
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
          bgcolor: "rgba(0,0,0,0)",
          xaxis: { visible: false },
          yaxis: { visible: false },
          zaxis: { visible: false },
          aspectmode: "data",
        },
      })
    : plotlyLayout({
        margin: { l: 40, r: 16, t: 16, b: 36 },
        xaxis: { title: "PC1", gridcolor: "rgba(148,163,184,.08)" },
        yaxis: { title: "PC2", gridcolor: "rgba(148,163,184,.08)" },
      });
  window.Plotly.newPlot($("#latent-plot"), [trace], layout, plotConfig());
}

function renderNeuronMap(analysis) {
  if (!plotlyReady()) return;
  $("#neuron-plot").innerHTML = "";
  const z = analysis.neuron_maps.normalized;
  if (!z.length || !z[0]?.length) {
    $("#neuron-plot").innerHTML = '<div class="plot-placeholder">No neuron map available.</div>';
    return;
  }
  const layout = plotlyLayout({
    margin: { l: 50, r: 12, t: 12, b: 34 },
    xaxis: { title: "neuron" },
    yaxis: { title: "layer", dtick: 1, autorange: "reversed" },
  });
  window.Plotly.newPlot(
    $("#neuron-plot"),
    [{ type: "heatmap", z, colorscale: [[0, "#071019"], [0.5, "#155e75"], [1, "#5eead4"]], colorbar: { thickness: 8 } }],
    layout,
    plotConfig()
  );
}

function renderAttention(analysis) {
  if (!plotlyReady()) return;
  $("#attention-plot").innerHTML = "";
  const attention = analysis.attention?.attention;
  if (!attention || !attention.length) return;
  const head = intValue("attention-head", 0, 32, 0);
  const z = attention.map((layer) => layer[head] || []);
  const tokens = analysis.attention.tokens;
  const boundary = analysis.attention.boundary_index;
  if (!z.length || !tokens || !tokens.length) {
    $("#attention-plot").innerHTML = '<div class="plot-placeholder">No attention map available.</div>';
    return;
  }
  const layout = plotlyLayout({
    margin: { l: 50, r: 12, t: 12, b: 64 },
    xaxis: { title: "key token", tickfont: { size: 7 }, showticklabels: true },
    yaxis: { title: "layer", dtick: 1, autorange: "reversed" },
    shapes: [
      {
        type: "line",
        x0: boundary,
        x1: boundary,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "rgba(251,191,36,.7)", width: 1, dash: "dash" },
      },
    ],
  });
  window.Plotly.newPlot(
    $("#attention-plot"),
    [{ type: "heatmap", z, x: tokens, y: z.map((_, index) => index), colorscale: [[0, "#071019"], [0.55, "#1e3a5f"], [1, "#5eead4"]], colorbar: { thickness: 8 } }],
    layout,
    plotConfig()
  );
}

function renderFFN(analysis) {
  if (!plotlyReady()) return;
  $("#ffn-plot").innerHTML = "";
  const ffn = analysis.attention?.ffn_activation;
  if (!ffn || !ffn.length) return;
  const layer = intValue("ffn-layer", 0, 1024, ffn.length - 1);
  const values = ffn[layer] || [];
  if (!values.length) {
    $("#ffn-plot").innerHTML = '<div class="plot-placeholder">No FFN activation available.</div>';
    return;
  }
  const indices = values.map((_, index) => index);
  const order = indices.slice().sort((a, b) => Math.abs(values[b]) - Math.abs(values[a])).slice(0, 28);
  const traces = [
    {
      type: "bar",
      x: order.map((index) => `n${index}`),
      y: order.map((index) => values[index]),
      marker: { color: order.map((index) => (values[index] >= 0 ? "#5eead4" : "#fb923c")) },
      hovertemplate: "%{x}<br>%{y:.3f}<extra></extra>",
    },
  ];
  const layout = plotlyLayout({
    margin: { l: 46, r: 12, t: 12, b: 52 },
    xaxis: { tickangle: -45, tickfont: { size: 7 } },
    yaxis: { title: "activation", gridcolor: "rgba(148,163,184,.08)" },
  });
  window.Plotly.newPlot($("#ffn-plot"), traces, layout, plotConfig());
}

function freshTransform(seed, distort) {
  if (!distort) {
    return {
      rotation_deg: [0, 0, 0],
      stretch: [1, 1, 1],
      twist: 0,
      shear: [0, 0, 0],
      bend: 0,
      warp: 0,
      translation: [0, 0, 0],
      noise: 0,
      permute: false,
      seed: seed + 1,
    };
  }
  const rng = (() => {
    let value = (seed + 1) % 2147483647;
    if (value <= 0) value += 2147483646;
    return () => {
      value = (value * 16807) % 2147483647;
      return (value - 1) / 2147483646;
    };
  })();
  const signed = (scale) => (rng() * 2 - 1) * scale;
  return {
    rotation_deg: [signed(180), signed(180), signed(180)],
    stretch: [1 + signed(0.18), 1 + signed(0.18), 1 + signed(0.18)],
    twist: signed(0.5),
    shear: [signed(0.35), signed(0.35), signed(0.35)],
    bend: signed(0.4),
    warp: signed(0.4),
    translation: [0, 0, 0],
    noise: 0,
    permute: true,
    seed: seed + 1,
  };
}

async function runFreshPredictions() {
  const jobId = $("#analysis-job").value || labState.jobId;
  if (!jobId) {
    toast("Select a run with a checkpoint first", true);
    return;
  }
  const button = $("#run-predictions");
  button.disabled = true;
  button.classList.add("loading");
  button.querySelector(".btn-label").textContent = "Predicting…";
  try {
    const count = intValue("fresh-count", 1, 24, 6);
    const ambient = intValue("fresh-ambient", 2, 12, 3);
    const seed = intValue("fresh-seed", -2147483648, 2147483647, 100);
    const mix = $("#fresh-mix").checked;
    const distort = $("#fresh-distort").checked;
    const structures = mix
      ? labState.catalog.structures
          .filter((structure) => structure.value !== "disjoint")
          .filter((structure) => structure.needs_ambient <= ambient)
          .map((structure) => structure.value)
      : null;
    const spec = {
      structure: $("#fresh-structure").value,
      n_points: intValue("fresh-points", 8, 512, 32),
      ambient_dim: ambient,
      copies: 1,
      genus: 2,
      seed,
      params: {},
      transform: freshTransform(seed, distort),
      filtration: {
        ...labState.catalog.data_defaults.filtration,
        epsilon: null,
        max_edge_length: null,
      },
      target_schema: labState.catalog.data_defaults.target_schema,
    };
    const result = await fetchJSON(`/api/predict/${jobId}`, {
      method: "POST",
      body: JSON.stringify({ spec, count, structures }),
    });
    renderFreshPredictions(result.items || [], result.warnings || []);
    toast(`Predicted ${result.count} clouds`);
  } catch (error) {
    setError(error.message);
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
    button.querySelector(".btn-label").textContent = "Generate & predict";
  }
}

function renderFreshPredictions(items, warnings) {
  const container = $("#fresh-predictions");
  container.innerHTML = "";
  for (const warning of warnings) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = warning;
    container.appendChild(note);
  }
  if (!items.length) {
    container.innerHTML = '<span class="muted">No predictions returned.</span>';
    return;
  }
  for (const item of items) {
    const card = document.createElement("div");
    card.className = "prediction-card";
    const head = document.createElement("div");
    head.className = "prediction-head";
    const name = document.createElement("strong");
    name.textContent = item.structure_name || item.sample?.structure_name || "cloud";
    const status = document.createElement("span");
    const trueBetti = item.true_groups?.[0] || [];
    const predBetti = item.predicted_groups?.[0] || [];
    const correct = trueBetti.slice(0, 3).join(",") === predBetti.slice(0, 3).join(",");
    status.textContent = correct ? "match" : "miss";
    status.style.color = correct ? "var(--cyan)" : "var(--red)";
    head.appendChild(name);
    head.appendChild(status);
    card.appendChild(head);
    if (item.projection?.length) {
      const mini = document.createElement("div");
      mini.className = "prediction-mini";
      card.appendChild(mini);
      window.requestAnimationFrame(() => plotMiniPoints(mini, item.projection));
    }
    const values = document.createElement("div");
    values.className = "prediction-values";
    const trueRow = document.createElement("div");
    trueRow.className = "true";
    trueRow.innerHTML = `<span class="label">true</span> b=(${trueBetti.slice(0, 3).join(",")})`;
    const predRow = document.createElement("div");
    predRow.className = `predicted${correct ? "" : " bad"}`;
    predRow.innerHTML = `<span class="label">pred</span> b=(${predBetti.slice(0, 3).join(",")})`;
    values.appendChild(trueRow);
    values.appendChild(predRow);
    const trueFiltration = item.true_groups?.[4] || [];
    const predictedFiltration = item.predicted_groups?.[4] || [];
    if (trueFiltration.length || predictedFiltration.length) {
      const filt = document.createElement("div");
      filt.className = "predicted";
      filt.innerHTML = `<span class="label">filtration ε</span> true ${trueFiltration[2] ?? "—"} · pred ${predictedFiltration[2] ?? "—"}`;
      values.appendChild(filt);
    }
    const predicted = document.createElement("div");
    predicted.className = "prediction-condition";
    predicted.textContent = item.predicted_text || "";
    values.appendChild(predicted);
    card.appendChild(values);
    container.appendChild(card);
  }
}

async function loadDatasets(preferredId = null) {
  const select = $("#dataset-select");
  const previous = preferredId || select.value || labState.datasetId;
  labState.datasets = await fetchJSON("/api/datasets");
  select.innerHTML = "";
  if (!labState.datasets.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No saved dataset";
    select.appendChild(option);
    labState.datasetId = null;
    $("#dataset-meta").textContent = "No dataset selected. Build one in Data Studio first.";
    return;
  }
  for (const dataset of labState.datasets.slice().reverse()) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.name} · ${dataset.id} · ${dataset.train_count}/${dataset.val_count}`;
    select.appendChild(option);
  }
  const valid = labState.datasets.some((dataset) => dataset.id === previous);
  const chosen = valid ? previous : labState.datasets[labState.datasets.length - 1].id;
  select.value = chosen;
  labState.datasetId = chosen;
  updateDatasetMeta();
}

function updateDatasetMeta() {
  const dataset = labState.datasets.find((item) => item.id === labState.datasetId);
  const meta = $("#dataset-meta");
  if (!dataset) {
    meta.textContent = "No dataset selected.";
    return;
  }
  const valAugment = dataset.data_spec?.val_augment ? "val transforms" : "clean val";
  meta.textContent = `${dataset.name} · ${dataset.id} · train ${dataset.train_count} / val ${dataset.val_count} · ${dataset.structures.join(", ")} · max_dim ${dataset.max_dim} · context ${dataset.max_seq_len} · ${valAugment}`;
  if (dataset.max_seq_len > intValue("context-train", 64, 32768, 1024)) {
    setValue("context-train", dataset.max_seq_len);
  }
  persistLabState();
}

function setTab(tab) {
  labState.view = tab;
  saveUIState({ labTab: tab });
  $$(".lab-tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $("#train-controls").classList.toggle("active", tab === "train");
  $("#interpret-controls").classList.toggle("active", tab === "interpret");
  $("#train-view").classList.toggle("active", tab === "train");
  $("#interpret-view").classList.toggle("active", tab === "interpret");
}

function bindEvents() {
  $("#train-toggle").addEventListener("click", toggleTraining);
  $("#refresh-runs").addEventListener("click", refreshRuns);
  $("#refresh-datasets").addEventListener("click", () => loadDatasets(labState.datasetId));
  $("#dataset-select").addEventListener("change", () => {
    labState.datasetId = $("#dataset-select").value || null;
    updateDatasetMeta();
  });
  $("#go-data-studio").addEventListener("click", () => {
    document.querySelector("#nav-data")?.click();
    document.querySelector("#dataset-builder")?.scrollIntoView({ behavior: "smooth", block: "center" });
  });
  window.addEventListener("llm-datasets-changed", () => loadDatasets(labState.datasetId));
  $("#run-analysis").addEventListener("click", runAnalysis);
  $("#analysis-job").addEventListener("change", () => {
    labState.jobId = $("#analysis-job").value || labState.jobId;
    persistLabState();
  });
  $(".lab-sidebar").addEventListener("input", persistLabState);
  $(".lab-sidebar").addEventListener("change", persistLabState);
  $(".fresh-controls").addEventListener("input", persistLabState);
  $(".fresh-controls").addEventListener("change", persistLabState);
  $("#run-predictions").addEventListener("click", runFreshPredictions);
  $$(".lab-tab").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });
  window.addEventListener("llm-lab-tab", (event) => setTab(event.detail));
  $("#probe-layer").addEventListener("change", () => {
    if (labState.analysis) {
      renderProbeHeatmap(labState.analysis);
      renderTopNeurons(labState.analysis);
    }
  });
  $("#latent-layer").addEventListener("change", () => {
    if (labState.analysis) renderLatent(labState.analysis);
  });
  $("#attention-head").addEventListener("change", () => {
    if (labState.analysis) renderAttention(labState.analysis);
  });
  $("#ffn-layer").addEventListener("change", () => {
    if (labState.analysis) renderFFN(labState.analysis);
  });
}

async function boot() {
  bindEvents();
  try {
    const health = await fetchJSON("/api/health");
    $("#health-chip").textContent = `torch ${health.torch}${health.cuda ? " · cuda" : ""}`;
    if (health.cuda) {
      $("#device").value = "cuda";
    }
  } catch (_error) {
    $("#health-chip").textContent = "offline";
  }
  try {
    await loadCatalog();
    const saved = await loadUIState();
    if (saved.labTab) setTab(saved.labTab);
    applyById(saved.labForm || {});
    applyById(saved.freshForm || {});
    await loadDatasets(saved.datasetId);
    await refreshRuns();
    if (saved.jobId) {
      labState.jobId = saved.jobId;
      pollStatus();
    } else {
      const running = labState.runs.find((run) => ["preparing", "running", "queued"].includes(run.status));
      const latest = running || labState.runs[labState.runs.length - 1];
      if (latest) {
        labState.jobId = latest.id;
        pollStatus();
      }
    }
  } catch (error) {
    setError(error.message);
  }
  if (window.location.hash === "#interpret") setTab("interpret");
}

boot();
