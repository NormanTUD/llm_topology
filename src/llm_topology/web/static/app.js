import { applyById, collectById, loadUIState, saveUIState } from "./state.js";

const state = {
  catalog: null,
  last: null,
  batch: [],
  selected: 0,
  batchWarnings: [],
  batchMode: "single",
  batchStructures: [],
  equationValues: {},
  generateToken: 0,
  abortController: null,
};

const COLORS = ["#5eead4", "#fb923c", "#a78bfa", "#f472b6"];
const DIM_NAMES = ["H0", "H1", "H2", "H3"];
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function optionalInt(id, min, max) {
  const raw = document.getElementById(id)?.value?.trim();
  if (!raw) return null;
  return clampInt(raw, min, max, null);
}

function numberValue(name, fallback = 0) {
  const element = $(`[name="${name}"]`);
  if (!element) return fallback;
  const parsed = Number(element.value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function setStatus(text, isError = false) {
  const footer = $("#footer-state");
  if (footer) {
    footer.textContent = text;
    footer.style.color = isError ? "var(--red)" : "";
  }
}

let toastTimer = null;
function toast(message, isError = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => element.classList.remove("show"), 2600);
}

function showError(message) {
  const banner = $("#error-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  setStatus("generation failed", true);
}

function hideError() {
  const banner = $("#error-banner");
  banner.textContent = "";
  banner.classList.add("hidden");
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
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch (_error) {
      // Keep the HTTP status text.
    }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function plotlyReady() {
  return typeof window.Plotly !== "undefined";
}

let webglCache = null;
function hasWebGL() {
  if (webglCache !== null) return webglCache;
  try {
    const canvas = document.createElement("canvas");
    webglCache = Boolean(canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
  } catch (_error) {
    webglCache = false;
  }
  return webglCache;
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

async function loadCatalog() {
  state.catalog = await fetchJSON("/api/catalog");
  const structureSelect = $("#structure-select");
  const baseSelect = $("#base-structure-select");
  structureSelect.innerHTML = "";
  baseSelect.innerHTML = "";
  for (const structure of state.catalog.structures) {
    const option = document.createElement("option");
    option.value = structure.value;
    option.textContent = `${structure.label}`;
    structureSelect.appendChild(option);
    if (structure.value !== "disjoint") {
      const baseOption = option.cloneNode(true);
      baseSelect.appendChild(baseOption);
    }
  }
  structureSelect.value = state.catalog.default_spec.structure;
  baseSelect.value = "torus";
  $("#version-chip").textContent = `v${state.catalog.version}`;
  $("#phase-chip").textContent = state.catalog.phase;
  $("#projection-method").textContent = "—";
  renderTargetBlocks();
  renderBatchStructures();
  syncStructureVisibility();
  syncMethodVisibility();
}

function selectedTargetNames() {
  return $$(".target-check:checked").map((input) => input.dataset.block);
}

function applyTargetBlockState(names) {
  if (!Array.isArray(names) || !names.length) return;
  const selected = new Set(names);
  $$(".target-check").forEach((input) => {
    input.checked = selected.has(input.dataset.block);
  });
}

function selectedBatchStructures() {
  return $$(".batch-structure:checked").map((input) => input.value);
}

function applyBatchStructures(values) {
  if (!Array.isArray(values) || !values.length) return;
  const selected = new Set(values);
  $$(".batch-structure").forEach((input) => {
    input.checked = selected.has(input.value);
  });
}

function renderBatchStructures() {
  const container = $("#batch-structure-checks");
  if (!container) return;
  container.innerHTML = "";
  const defaults = new Set(["circle", "sphere", "torus", "genus_surface"]);
  for (const structure of state.catalog.structures) {
    if (structure.value === "disjoint") continue;
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "batch-structure";
    checkbox.value = structure.value;
    checkbox.checked = defaults.has(structure.value);
    const text = document.createElement("span");
    text.textContent = structure.label;
    label.appendChild(checkbox);
    label.appendChild(text);
    container.appendChild(label);
  }
}

function syncBatchMode() {
  const mode = $("#batch-mode").value;
  const mix = mode === "mix";
  $("#batch-structure-wrap").classList.toggle("hidden", !mix);
  $("#structure-select").disabled = mix;
  if (mix && !selectedBatchStructures().length) {
    applyBatchStructures(["circle", "sphere", "torus", "genus_surface"]);
  }
}

function syncBatchStructureAvailability() {
  const ambient = clampInt($("#ambient-dim").value, 2, 64, 3);
  $$(".batch-structure").forEach((checkbox) => {
    const structure = state.catalog.structures.find((item) => item.value === checkbox.value);
    const unavailable = structure && structure.needs_ambient > ambient;
    checkbox.disabled = Boolean(unavailable);
    if (unavailable) checkbox.checked = false;
  });
}

function persistDataState() {
  saveUIState({
    dataForm: collectById("#generate-form"),
    targetBlocks: selectedTargetNames(),
    datasetForm: collectById("#dataset-builder"),
    lastBatch: state.batch,
    selected: state.selected,
    batchWarnings: state.batchWarnings,
    batchMode: $("#batch-mode")?.value || state.batchMode,
    batchStructures: selectedBatchStructures(),
  });
}

function renderTargetBlocks() {
  const container = $("#target-blocks");
  if (!state.catalog) return;
  const checkedBefore = new Set(selectedTargetNames());
  const initial = checkedBefore.size ? checkedBefore : new Set(["betti"]);
  const maxDim = clampInt($("#max-dim")?.value, 0, 6, 2);
  container.innerHTML = "";
  for (const block of state.catalog.target_blocks) {
    const fields = ["betti", "ph"].includes(block.name)
      ? Array.from({ length: maxDim + 1 }, (_, index) => `b${index}`)
      : block.fields;
    const row = document.createElement("div");
    row.className = "target-block";
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "target-check";
    checkbox.dataset.block = block.name;
    checkbox.checked = initial.has(block.name);
    const text = document.createElement("span");
    text.textContent = block.name;
    label.appendChild(checkbox);
    label.appendChild(text);
    const fieldText = document.createElement("span");
    fieldText.className = "fields";
    fieldText.textContent = fields.join(",");
    row.appendChild(label);
    row.appendChild(fieldText);
    container.appendChild(row);
  }
}

function syncStructureVisibility() {
  const structure = $("#structure-select").value;
  $("#base-structure-wrap").classList.toggle("hidden", structure !== "disjoint");
  const ambient = $("#ambient-dim");
  if (structure === "klein_bottle" && Number(ambient.value) < 4) {
    ambient.value = "4";
  }
  const needsGenus = structure === "genus_surface";
  $("#genus").disabled = !needsGenus;
  if (!needsGenus) $("#genus").value = "2";
}

function syncMethodVisibility() {
  const method = $("#method-select").value;
  $("#rips-edge-wrap").classList.toggle("hidden", method !== "rips");
}

function syncEpsilonVisibility() {
  const auto = $("#auto-epsilon").checked;
  const epsilon = $("#epsilon");
  epsilon.disabled = auto;
  if (auto) epsilon.value = "";
}

function buildSchema() {
  const maxDim = clampInt($("#max-dim").value, 0, 6, 2);
  const selected = selectedTargetNames();
  if (!selected.length) throw new Error("Select at least one target group.");
  const blocks = selected.map((name) => {
    const template = state.catalog.target_blocks.find((block) => block.name === name);
    const fields = ["betti", "ph"].includes(name)
      ? Array.from({ length: maxDim + 1 }, (_, index) => `b${index}`)
      : template.fields;
    return { name, fields };
  });
  return { blocks };
}

function buildSpec() {
  const structure = $("#structure-select").value;
  const maxDim = clampInt($("#max-dim").value, 0, 6, 2);
  const autoEpsilon = $("#auto-epsilon").checked;
  const method = $("#method-select").value;
  const seed = numberValue("seed", 0);
  const edgeRaw = $("#max-edge-length").value.trim();
  const epsilonRaw = $("#epsilon").value.trim();
  const params = {};
  if (structure === "disjoint") params.base_structure = $("#base-structure-select").value;

  return {
    structure,
    n_points: clampInt($("#n-points").value, 4, 100000, 240),
    points_per_component: optionalInt("points-per-component", 4, 2048),
    precision: clampInt($("#precision").value, 1, 8, 5),
    ambient_dim: clampInt($("#ambient-dim").value, 2, 64, 3),
    copies: 1,
    copies_min: optionalInt("copies-min", 1, 1000),
    copies_max: optionalInt("copies-max", 1, 1000),
    genus: clampInt($("#genus").value, 1, 32, 2),
    seed,
    params,
    transform: {
      rotation_deg: [
        numberValue("rot_x", 0),
        numberValue("rot_y", 0),
        numberValue("rot_z", 0),
      ],
      stretch: [
        Math.max(0.01, numberValue("stretch_x", 1)),
        Math.max(0.01, numberValue("stretch_y", 1)),
        Math.max(0.01, numberValue("stretch_z", 1)),
      ],
      twist: numberValue("twist", 0),
      shear: [
        numberValue("shear_xy", 0),
        numberValue("shear_xz", 0),
        numberValue("shear_yz", 0),
      ],
      bend: numberValue("bend", 0),
      warp: numberValue("warp", 0),
      translation: [0, 0, 0],
      noise: Math.max(0, numberValue("noise", 0)),
      permute: $('[name="permute"]').checked,
      seed: seed + 1,
    },
    filtration: {
      method,
      field: $('[name="field"]').value,
      max_dim: maxDim,
      epsilon: autoEpsilon || epsilonRaw === "" ? null : Math.max(0, Number(epsilonRaw)),
      max_edge_length: method === "rips" && edgeRaw !== "" ? Math.max(0.0001, Number(edgeRaw)) : null,
      rips_threshold_quantile: 0.35,
    },
    target_schema: buildSchema(),
  };
}

function setLoading(loading) {
  const button = $("#generate-btn");
  button.disabled = loading;
  button.classList.toggle("loading", loading);
  $(".btn-label").textContent = loading ? "Computing filtration…" : "Generate point cloud";
}

function buildBatchPayload() {
  const spec = buildSpec();
  const count = clampInt($("#batch-count").value, 1, 24, 6);
  const mode = $("#batch-mode").value;
  let structures = null;
  if (mode === "mix") {
    structures = selectedBatchStructures();
    const available = new Set(
      state.catalog.structures
        .filter((structure) => structure.needs_ambient <= spec.ambient_dim)
        .map((structure) => structure.value)
    );
    structures = structures.filter((value) => available.has(value));
    if (!structures.length) {
      throw new Error("Select at least one structure compatible with the ambient dimension.");
    }
  }
  return {
    spec,
    count,
    mode,
    structures,
    vary_transform: $("#vary-transform").checked,
    variation_strength: Math.max(0, Math.min(1, numberValue("variation_strength", 0.35))),
  };
}

async function generate({ live = false } = {}) {
  hideError();
  setLoading(true);
  setStatus(live ? "updating preview" : "generating");
  const token = ++state.generateToken;
  try {
    const payload = buildBatchPayload();
    const { mode, structures } = payload;
    const previousSelected = state.selected;
    state.abortController?.abort();
    const controller = new AbortController();
    state.abortController = controller;
    const data = await fetchJSON("/api/generate/batch", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify(payload),
    });
    if (token !== state.generateToken) return;
    state.batch = data.items;
    state.batchWarnings = data.warnings || [];
    state.batchMode = data.mode || mode;
    state.batchStructures = data.structures || structures || [];
    state.selected = Math.min(previousSelected, state.batch.length - 1);
    renderGallery(state.batch);
    renderSelection();
    persistDataState();
    setStatus("ready");
  } catch (error) {
    if (error.name === "AbortError") return;
    showError(error.message || String(error));
    toast(error.message || "Generation failed", true);
  } finally {
    if (token === state.generateToken) {
      setLoading(false);
      updateDataPlausibility();
    }
  }
}

function render(data) {
  renderWarnings([...(data.warnings || []), ...state.batchWarnings]);
  renderNotes(data.notes || []);
  renderMetrics(data);
  renderCloud(data);
  renderPersistence(data);
  renderBetti(data);
  renderTargets(data);
  renderSequence(data);
  renderEquationValues(data);
}

function renderWarnings(warnings) {
  const container = $("#warning-list");
  container.innerHTML = "";
  const unique = [...new Set(warnings)];
  for (const warning of unique) {
    const item = document.createElement("div");
    item.className = "warning";
    item.textContent = warning;
    container.appendChild(item);
  }
}

function renderNotes(notes) {
  const container = $("#note-list");
  container.innerHTML = "";
  for (const note of [...new Set(notes)]) {
    const item = document.createElement("div");
    item.className = "note";
    item.textContent = note;
    container.appendChild(item);
  }
}

function renderGallery(items) {
  const container = $("#cloud-gallery");
  container.innerHTML = "";
  $("#gallery-count").textContent = `${items.length} cloud${items.length === 1 ? "" : "s"}`;
  const recipe = state.batchMode === "mix"
    ? `mix: ${(state.batchStructures || []).join(", ")}`
    : `single: ${items[0]?.sample?.structure_name || "cloud"}`;
  $("#batch-recipe").textContent = recipe.length > 54 ? `${recipe.slice(0, 51)}…` : recipe;
  items.forEach((item, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "cloud-card";
    card.dataset.index = String(index);
    const title = document.createElement("div");
    title.className = "cloud-card-title";
    const name = document.createElement("strong");
    name.textContent = item.sample.structure_name;
    const betti = document.createElement("span");
    betti.textContent = `b=(${item.persistence.betti.join(",")})`;
    title.appendChild(name);
    title.appendChild(betti);
    const mini = document.createElement("div");
    mini.className = "cloud-mini";
    card.appendChild(title);
    card.appendChild(mini);
    card.addEventListener("click", () => {
      state.selected = index;
      renderSelection();
    });
    container.appendChild(card);
    window.requestAnimationFrame(() => plotMiniCloud(mini, item));
  });
  updateGallerySelection();
}

function plotMiniCloud(element, item) {
  if (!plotlyReady()) return;
  const points = item.projection;
  const trace = {
    type: hasWebGL() ? "scattergl" : "scatter",
    mode: "markers",
    x: points.map((point) => point[0]),
    y: points.map((point) => point[1]),
    marker: {
      size: item.sample.n_points > 500 ? 1.6 : 2.3,
      color: points.map((_, index) => index),
      colorscale: [[0, "#60a5fa"], [0.5, "#5eead4"], [1, "#fb923c"]],
      opacity: 0.85,
    },
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

function updateGallerySelection() {
  $$(".cloud-card").forEach((card, index) => {
    card.classList.toggle("active", index === state.selected);
  });
}

function renderSelection() {
  if (!state.batch.length) return;
  state.selected = Math.max(0, Math.min(state.selected, state.batch.length - 1));
  state.last = state.batch[state.selected];
  render(state.last);
  updateGallerySelection();
  saveUIState({ selected: state.selected });
}

function renderMetrics(data) {
  const sample = data.sample;
  const persistence = data.persistence;
  $("#metric-theory").textContent = `(${sample.theoretical_betti.join(", ")})`;
  $("#metric-ph").textContent = `(${persistence.betti.join(", ")})`;
  $("#metric-field").textContent = `${persistence.field} · ${persistence.method}`;
  const guard = $("#metric-guard");
  guard.textContent = data.topology_guard ? "match" : "differs";
  guard.style.color = data.topology_guard ? "var(--cyan)" : "var(--red)";
  $("#metric-guard-note").textContent = data.topology_guard
    ? "theoretical = computed PH"
    : "finite sample / filtration scale";
  $("#metric-epsilon").textContent = Number(persistence.epsilon).toFixed(4);
  $("#metric-epsilon-note").textContent = "auto-selected filtration scale";
  $("#metric-structure").textContent = sample.structure_name;
  const batchInfo = state.batch.length > 1 ? ` · cloud ${state.selected + 1}/${state.batch.length}` : "";
  $("#metric-counts").textContent =
    `${sample.n_points} points · ${sample.n_components} components · genus ${sample.genus}${batchInfo}`;
  $("#projection-method").textContent = data.projection_method;
}

function renderCloud(data) {
  const element = $("#cloud-plot");
  if (!plotlyReady()) {
    element.innerHTML = '<div class="plot-placeholder">Plotly could not be loaded.</div>';
    return;
  }
  element.innerHTML = "";
  const projected = data.projection;
  const colors = projected.map((_, index) => index);
  const use3d = hasWebGL();
  const traces = [{
    type: use3d ? "scatter3d" : "scatter",
    mode: "markers",
    x: projected.map((point) => point[0]),
    y: projected.map((point) => point[1]),
    ...(use3d ? { z: projected.map((point) => point[2]) } : {}),
    marker: {
      size: data.sample.n_points > 900 ? 2.1 : 3.2,
      color: colors,
      colorscale: [
        [0, "#60a5fa"],
        [0.35, "#5eead4"],
        [0.7, "#a78bfa"],
        [1, "#fb923c"],
      ],
      opacity: 0.88,
      line: { width: 0 },
    },
    hovertemplate: "p %{pointNumber}<br>x %{x:.3f}<br>y %{y:.3f}<br>z %{z:.3f}<extra></extra>",
  }];
  const reference = data.reference_projection || [];
  const showReference = Boolean($("#show-reference")?.checked);
  if (showReference && reference.length) {
    traces.push({
      type: use3d ? "scatter3d" : "scatter",
      mode: "markers",
      name: "canonical reference",
      x: reference.map((point) => point[0]),
      y: reference.map((point) => point[1]),
      ...(use3d ? { z: reference.map((point) => point[2]) } : {}),
      marker: { size: 2, color: "rgba(180,195,220,.55)", symbol: "circle-open", opacity: 0.7 },
      hovertemplate: "reference<extra></extra>",
    });
  }
  const layout = use3d
    ? plotlyLayout({
        margin: { l: 0, r: 0, t: 0, b: 0 },
        showlegend: true,
        legend: { orientation: "h", y: 1.02, x: 0, font: { size: 9 } },
        scene: {
          bgcolor: "rgba(0,0,0,0)",
          xaxis: { title: "", showgrid: true, gridcolor: "rgba(148,163,184,.10)", zeroline: false, showticklabels: false },
          yaxis: { title: "", showgrid: true, gridcolor: "rgba(148,163,184,.10)", zeroline: false, showticklabels: false },
          zaxis: { title: "", showgrid: true, gridcolor: "rgba(148,163,184,.10)", zeroline: false, showticklabels: false },
          camera: { eye: { x: 1.35, y: 1.15, z: 1.05 } },
          aspectmode: "data",
        },
      })
    : plotlyLayout({
        margin: { l: 40, r: 16, t: 16, b: 36 },
        showlegend: true,
        legend: { orientation: "h", y: 1.02, x: 0, font: { size: 9 } },
        xaxis: { title: "x", gridcolor: "rgba(148,163,184,.08)" },
        yaxis: { title: "y", gridcolor: "rgba(148,163,184,.08)" },
      });
  window.Plotly.newPlot(element, traces, layout, plotConfig());
  const referenceChip = $("#reference-chip");
  if (referenceChip) {
    referenceChip.textContent = !showReference
      ? "reference: hidden"
      : data.reference_structure_name
        ? `reference: ${data.reference_structure_name} (not model input)`
        : "reference: —";
  }
}

function renderPersistence(data) {
  const element = $("#persistence-plot");
  if (!plotlyReady()) {
    element.innerHTML = '<div class="plot-placeholder">Plotly could not be loaded.</div>';
    return;
  }
  element.innerHTML = "";
  const diagrams = data.persistence.diagrams;
  const allDeaths = diagrams.flat().map((pair) => pair[1]).filter((value) => value !== null);
  const maxFinite = allDeaths.length ? Math.max(...allDeaths) : data.persistence.epsilon;
  const upper = Math.max(maxFinite * 1.08, data.persistence.epsilon * 1.08, 0.1);
  const traces = [];
  diagrams.forEach((diagram, dimension) => {
    if (!diagram.length) return;
    const finite = diagram.filter((pair) => pair[1] !== null);
    const infinite = diagram.filter((pair) => pair[1] === null);
    if (finite.length) {
      traces.push({
        type: hasWebGL() ? "scattergl" : "scatter",
        mode: "markers",
        name: DIM_NAMES[dimension] || `H${dimension}`,
        x: finite.map((pair) => pair[0]),
        y: finite.map((pair) => pair[1]),
        marker: { color: COLORS[dimension % COLORS.length], size: 6, opacity: 0.82 },
        hovertemplate: `${DIM_NAMES[dimension] || `H${dimension}`} birth %{x:.3f}<br>death %{y:.3f}<extra></extra>`,
      });
    }
    if (infinite.length) {
      traces.push({
        type: hasWebGL() ? "scattergl" : "scatter",
        mode: "markers",
        name: `${DIM_NAMES[dimension] || `H${dimension}`} infinite`,
        x: infinite.map((pair) => pair[0]),
        y: infinite.map(() => upper),
        marker: { color: COLORS[dimension % COLORS.length], size: 8, symbol: "diamond", opacity: 0.9 },
        hovertemplate: `${DIM_NAMES[dimension] || `H${dimension}`} infinite<extra></extra>`,
      });
    }
  });
  traces.push({
    type: "scatter",
    mode: "lines",
    name: "diagonal",
    x: [0, upper],
    y: [0, upper],
    line: { color: "rgba(139,155,180,.25)", width: 1, dash: "dot" },
    hoverinfo: "skip",
    showlegend: false,
  });
  const layout = plotlyLayout({
    showlegend: true,
    legend: { orientation: "h", y: 1.08, x: 0, font: { size: 9 } },
    xaxis: { title: "birth", gridcolor: "rgba(148,163,184,.08)", zeroline: false },
    yaxis: { title: "death", gridcolor: "rgba(148,163,184,.08)", zeroline: false },
    annotations: data.reference_betti?.length
      ? [
          {
            x: 0.02,
            y: 0.98,
            xref: "paper",
            yref: "paper",
            text: `reference b=(${data.reference_betti.join(",")})`,
            showarrow: false,
            font: { color: "#8b9bb4", size: 9, family: "JetBrains Mono, monospace" },
          },
        ]
      : [],
  });
  window.Plotly.newPlot(element, traces, layout, plotConfig());
}

function renderBetti(data) {
  const element = $("#betti-plot");
  if (!plotlyReady()) {
    element.innerHTML = '<div class="plot-placeholder">Plotly could not be loaded.</div>';
    return;
  }
  element.innerHTML = "";
  const grid = data.persistence.epsilon_grid;
  const curve = data.persistence.betti_curve;
  const traces = [];
  const dims = curve.length ? curve[0].length : 0;
  for (let dimension = 0; dimension < dims; dimension += 1) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: DIM_NAMES[dimension] || `H${dimension}`,
      x: grid,
      y: curve.map((row) => row[dimension]),
      line: { color: COLORS[dimension % COLORS.length], width: 2, shape: "hv" },
      hovertemplate: `${DIM_NAMES[dimension] || `H${dimension}`} = %{y}<br>epsilon %{x:.3f}<extra></extra>`,
    });
  }
  const reference = data.reference_betti || [];
  if (reference.length) {
    reference.forEach((value, dimension) => {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `reference b${dimension}`,
        x: [0, grid[grid.length - 1] || 1],
        y: [value, value],
        line: { color: COLORS[dimension % COLORS.length], width: 1, dash: "dot" },
        hoverinfo: "skip",
        opacity: 0.55,
      });
    });
  }
  const layout = plotlyLayout({
    showlegend: true,
    legend: { orientation: "h", y: 1.08, x: 0, font: { size: 9 } },
    xaxis: { title: "epsilon", gridcolor: "rgba(148,163,184,.08)", zeroline: false },
    yaxis: { title: "betti", gridcolor: "rgba(148,163,184,.08)", zeroline: false, rangemode: "tozero" },
    shapes: [
      {
        type: "line",
        x0: data.persistence.epsilon,
        x1: data.persistence.epsilon,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "rgba(94,234,212,.5)", width: 1, dash: "dash" },
      },
    ],
  });
  window.Plotly.newPlot(element, traces, layout, plotConfig());
}

function renderTargets(data) {
  const tbody = $("#target-table tbody");
  tbody.innerHTML = "";
  data.target_schema.blocks.forEach((block, index) => {
    const values = data.target_groups[index] || [];
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    nameCell.textContent = block.name;
    const fieldsCell = document.createElement("td");
    fieldsCell.textContent = block.fields.join(", ");
    const valuesCell = document.createElement("td");
    valuesCell.textContent = values.map((value) => Number(value).toString()).join(", ");
    row.appendChild(nameCell);
    row.appendChild(fieldsCell);
    row.appendChild(valuesCell);
    tbody.appendChild(row);
  });
  $("#target-text").textContent = data.target_text;
  const sequence = data.sequence;
  $("#sequence-text").textContent = sequence.length > 18000 ? `${sequence.slice(0, 18000)}\n…` : sequence;
  renderTokenStream(data.target_text);
}

async function renderTokenStream(text) {
  const container = $("#token-stream");
  container.innerHTML = "";
  try {
    const payload = await fetchJSON("/api/tokenize", {
      method: "POST",
      body: JSON.stringify({ text, strict: true }),
    });
    const limit = 260;
    payload.tokens.slice(0, limit).forEach((token, index) => {
      const chip = document.createElement("span");
      chip.className = `token${token.startsWith("<") ? " special" : ""}`;
      const value = document.createElement("span");
      value.textContent = token;
      const id = document.createElement("span");
      id.className = "id";
      id.textContent = payload.ids[index];
      chip.appendChild(value);
      chip.appendChild(id);
      container.appendChild(chip);
    });
    if (payload.tokens.length > limit) {
      const more = document.createElement("span");
      more.className = "muted";
      more.textContent = `+${payload.tokens.length - limit} more tokens`;
      container.appendChild(more);
    }
    if (!payload.tokens.length) {
      container.innerHTML = '<span class="muted">Empty target.</span>';
    }
  } catch (error) {
    container.innerHTML = `<span class="muted">${error.message}</span>`;
  }
}

function renderSequence(data) {
  const sequence = data.sequence;
  $("#sequence-text").textContent = sequence.length > 18000 ? `${sequence.slice(0, 18000)}\n…` : sequence;
}

function renderEquations() {
  const equations = [
    {
      key: "betti",
      title: "Betti number",
      tex: String.raw`b_k(\varepsilon) = \dim_{\mathbb{F}} H_k\!\left(\mathrm{VR}_\varepsilon(X)\right)`,
    },
    {
      key: "sequence",
      title: "Token stream",
      tex: String.raw`\mathcal{S} = x_1,y_1,z_1;\ldots;\;<\mathrm{endoftext}>\;b_0,b_1,b_2;\ldots`,
    },
    {
      key: "persistence",
      title: "Persistence diagram",
      tex: String.raw`\mathrm{PD}_k = \left\{(\mathrm{birth}_i,\mathrm{death}_i)\right\}_{i=1}^{m_k}`,
    },
    {
      key: "attention",
      title: "Attention · Phase 1",
      tex: String.raw`\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V`,
    },
    {
      key: "ffn",
      title: "FFN · Phase 1",
      tex: String.raw`\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2`,
    },
    {
      key: "probe",
      title: "Linear probe",
      tex: String.raw`\hat b_k = w_k^\top h_\ell + c_k`,
    },
  ];
  const grid = $("#equation-grid");
  grid.innerHTML = "";
  for (const equation of equations) {
    const card = document.createElement("div");
    card.className = "equation-card";
    const title = document.createElement("div");
    title.className = "equation-title";
    title.textContent = equation.title;
    const body = document.createElement("div");
    body.className = "equation-body";
    const value = document.createElement("div");
    value.className = "equation-value";
    value.dataset.equation = equation.key;
    try {
      if (window.temml && typeof window.temml.render === "function") {
        window.temml.render(equation.tex, body);
      } else {
        body.textContent = equation.tex;
      }
    } catch (_error) {
      body.textContent = equation.tex;
    }
    card.appendChild(title);
    card.appendChild(body);
    card.appendChild(value);
    grid.appendChild(card);
    state.equationValues[equation.key] = value;
  }
}

function renderEquationValues(data) {
  const values = state.equationValues;
  if (values.betti) {
    values.betti.textContent =
      `current b = (${data.persistence.betti.join(", ")}) · epsilon = ${data.persistence.epsilon.toFixed(4)}`;
  }
  if (values.sequence) {
    values.sequence.textContent =
      `boundary = <endoftext> · target groups = ${data.target_schema.blocks.length} · tokens ≈ ${data.sequence.length}`;
  }
  if (values.persistence) {
    const finite = data.persistence.diagrams.flat().filter((pair) => pair[1] !== null).length;
    values.persistence.textContent = `${finite} finite persistence pairs`;
  }
  if (values.attention) values.attention.textContent = "custom transformer arrives in Phase 1";
  if (values.ffn) values.ffn.textContent = "layer-wise neuron maps arrive in Phase 1";
  if (values.probe) {
    values.probe.textContent = `planned target: ${data.target_schema.blocks.map((block) => block.name).join(" · ")}`;
  }
}

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${label} copied`);
  } catch (_error) {
    toast("Clipboard unavailable", true);
  }
}

function updateDatasetBudget() {
  const copies = optionalInt("copies-max", 1, 1000) || clampInt($("#copies")?.value, 1, 1000, 1);
  const per = optionalInt("dataset-points-per-component", 8, 2048);
  const points = per ? per * copies : clampInt($("#dataset-n-points")?.value, 8, 4096, 32);
  const ambient = clampInt($("#ambient-dim")?.value, 2, 64, 3);
  const precision = clampInt($("#dataset-precision")?.value, 1, 8, 5);
  let context = clampInt($("#dataset-context")?.value, 64, 32768, 1024);
  const estimated = points * ambient * (precision + 3) + 96;
  let autoFitted = false;
  if (estimated > context && estimated <= 32768) {
    context = 2 ** Math.ceil(Math.log2(estimated));
    const contextInput = $("#dataset-context");
    if (contextInput) contextInput.value = String(context);
    autoFitted = true;
  }
  const budget = $("#dataset-budget");
  const button = $("#build-dataset");
  if (budget) {
    const fits = estimated <= context;
    budget.textContent = `Estimated token budget: ~${estimated} / context ${context}. ` +
      (fits ? (autoFitted ? "Context auto-fitted." : "Fits.") : "Reduce points/precision or raise the context cap.");
    budget.style.color = fits ? "" : "var(--red)";
  }
  if (button) button.disabled = estimated > context;
}

function structuresForAmbient(structures, ambient) {
  const available = new Set(
    state.catalog.structures
      .filter((structure) => structure.needs_ambient <= ambient)
      .map((structure) => structure.value)
  );
  return structures.filter((value) => available.has(value));
}

async function buildTrainingDataset() {
  hideError();
  const button = $("#build-dataset");
  button.disabled = true;
  button.classList.add("loading");
  button.querySelector(".btn-label").textContent = "Building dataset…";
  try {
    const base = buildSpec();
    updateDatasetBudget();
    const perComponent = optionalInt("dataset-points-per-component", 8, 2048);
    const totalPoints = perComponent
      ? perComponent * Math.max(1, base.copies_max || base.copies || 1)
      : clampInt($("#dataset-n-points").value, 8, 4096, 32);
    const estimated = totalPoints *
      base.ambient_dim * (clampInt($("#dataset-precision").value, 1, 8, 5) + 3) + 96;
    if (estimated > clampInt($("#dataset-context").value, 64, 32768, 1024)) {
      throw new Error("Dataset token budget exceeds context. Increase context or reduce points/precision.");
    }
    const mode = $("#batch-mode").value;
    let structures = mode === "mix" ? selectedBatchStructures() : [base.structure];
    structures = structuresForAmbient(structures, base.ambient_dim);
    if (!structures.length) {
      throw new Error("No compatible structure selected for this dataset.");
    }
    const filtration = { ...base.filtration };
    const dataSpec = {
      structures,
      n_train: clampInt($("#dataset-n-train").value, 4, 5000, 48),
      n_val: clampInt($("#dataset-n-val").value, 2, 1000, 16),
      n_test: clampInt($("#dataset-n-test").value, 2, 1000, 8),
      n_points: totalPoints,
      points_per_component: perComponent,
      ambient_dim: base.ambient_dim,
      copies: base.copies,
      copies_min: base.copies_min,
      copies_max: base.copies_max,
      genus: base.genus,
      precision: clampInt($("#dataset-precision").value, 1, 8, 5),
      max_seq_len: clampInt($("#dataset-context").value, 64, 32768, 1024),
      task_mix: Math.max(0, Math.min(1, numberValue("dataset-task-mix", 0.5))),
      augment: $("#dataset-augment").checked,
      augment_strength: Math.max(0, Math.min(1, numberValue("dataset-augment-strength", 0.35))),
      val_transform_mode: $("#dataset-val-transform-mode").value,
      test_transform_mode: $("#dataset-test-transform-mode").value,
      val_augment_strength: Math.max(0, Math.min(1, numberValue("dataset-val-augment-strength", 0.35))),
      test_augment_strength: Math.max(0, Math.min(1, numberValue("dataset-test-augment-strength", 0.35))),
      seed: base.seed,
      filtration,
      target_schema: base.target_schema,
    };
    const meta = await fetchJSON("/api/datasets", {
      method: "POST",
      body: JSON.stringify({
        name: $("#dataset-name").value || "topology-set",
        data_spec: dataSpec,
      }),
    });
    $("#dataset-status").textContent = `saved ${meta.id} · ${meta.train_count}/${meta.val_count}`;
    $("#dataset-status").style.color = "var(--cyan)";
    window.dispatchEvent(new CustomEvent("llm-datasets-changed"));
    toast(`Dataset ${meta.id} saved`);
  } catch (error) {
    $("#dataset-status").textContent = "build failed";
    $("#dataset-status").style.color = "var(--red)";
    showError(error.message);
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
    button.querySelector(".btn-label").textContent = "Build & save dataset";
  }
}

function renderPlausibility(container, result) {
  if (!container) return;
  container.innerHTML = "";
  if (!result) return;
  if (!result.warnings?.length) return;
  for (const warning of result.warnings) {
    const item = document.createElement("div");
    item.className = `plaus-item ${result.level === "block" ? "block" : result.level === "warn" ? "warn" : ""}`;
    item.textContent = warning;
    container.appendChild(item);
  }
}

let plausibilityTimer = null;
async function updateDataPlausibility() {
  window.clearTimeout(plausibilityTimer);
  plausibilityTimer = window.setTimeout(async () => {
    const container = $("#data-plausibility");
    const button = $("#generate-btn");
    try {
      const payload = buildBatchPayload();
      const result = await fetchJSON("/api/estimate/generate", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderPlausibility(container, result);
      if (result.level === "block") {
        button.disabled = true;
        button.title = result.warnings.join(" ");
      } else if (!button.classList.contains("loading")) {
        button.disabled = false;
        button.title = "";
      }
    } catch (error) {
      renderPlausibility(container, { level: "warn", warnings: [error.message] });
    }
  }, 450);
}

function setControl(id, value) {
  const element = document.getElementById(id);
  if (!element) return;
  if (element.type === "checkbox") element.checked = Boolean(value);
  else element.value = String(value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

async function quickDemo() {
  setControl("structure-select", "torus");
  setControl("n-points", 400);
  setControl("copies", 1);
  setControl("copies-min", "");
  setControl("copies-max", "");
  setControl("points-per-component", "");
  setControl("precision", 5);
  setControl("batch-mode", "single");
  setControl("batch-count", 3);
  setControl("live-preview", false);
  applyTargetBlockState(["betti"]);
  syncStructureVisibility();
  syncBatchMode();
  await generate();
}

let liveTimer = null;
function scheduleLiveGenerate() {
  if (!$("#live-preview")?.checked) return;
  window.clearTimeout(liveTimer);
  liveTimer = window.setTimeout(() => {
    generate({ live: true });
  }, 750);
}

async function boot() {
  renderEquations();
  try {
    const health = await fetchJSON("/api/health");
    $("#health-chip").textContent = `py ${health.python} · ${(health.ram_available_bytes/1024**3).toFixed(1)} GiB RAM · ${health.gpu ? "GPU" : "CPU"}`;
  } catch (_error) {
    $("#health-chip").textContent = "offline";
  }
  try {
    await loadCatalog();
  } catch (error) {
    showError(`Could not load catalog: ${error.message}`);
    return;
  }
  const saved = await loadUIState();
  applyById(saved.dataForm || {});
  applyById(saved.datasetForm || {});
  applyTargetBlockState(saved.targetBlocks);
  applyBatchStructures(saved.batchStructures);
  syncStructureVisibility();
  syncBatchMode();
  syncBatchStructureAvailability();
  syncMethodVisibility();
  syncEpsilonVisibility();
  updateDatasetBudget();
  updateDataPlausibility();
  $("#structure-select").addEventListener("change", syncStructureVisibility);
  $("#ambient-dim").addEventListener("input", syncBatchStructureAvailability);
  $("#ambient-dim").addEventListener("change", syncBatchStructureAvailability);
  $("#batch-mode").addEventListener("change", () => {
    syncBatchMode();
    persistDataState();
    scheduleLiveGenerate();
    updateDataPlausibility();
  });
  $("#method-select").addEventListener("change", syncMethodVisibility);
  $("#auto-epsilon").addEventListener("change", syncEpsilonVisibility);
  $("#max-dim").addEventListener("change", renderTargetBlocks);
  $("#randomize-btn").addEventListener("click", () => {
    $("#seed").value = String(Math.floor(Math.random() * 1_000_000));
    toast("Seed randomized");
  });
  $("#generate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    generate();
  });
  $("#copy-target").addEventListener("click", () => {
    if (state.last) copyText(state.last.target_text, "Target");
  });
  $("#copy-sequence").addEventListener("click", () => {
    if (state.last) copyText(state.last.sequence, "Sequence");
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      generate();
    }
  });
  const form = $("#generate-form");
  const onControlChange = () => {
    persistDataState();
    scheduleLiveGenerate();
    updateDataPlausibility();
  };
  form.addEventListener("input", onControlChange);
  form.addEventListener("change", onControlChange);
  $("#target-blocks").addEventListener("change", onControlChange);
  $("#dataset-builder").addEventListener("input", () => {
    persistDataState();
    updateDatasetBudget();
  });
  $("#dataset-builder").addEventListener("change", () => {
    persistDataState();
    updateDatasetBudget();
  });
  $("#ambient-dim").addEventListener("input", updateDatasetBudget);
  $("#ambient-dim").addEventListener("change", updateDatasetBudget);
  $("#batch-structure-checks").addEventListener("change", onControlChange);
  $("#build-dataset").addEventListener("click", buildTrainingDataset);
  $("#show-reference").addEventListener("change", () => {
    if (state.last) render(state.last);
    persistDataState();
  });
  $("#top-generate").addEventListener("click", () => generate());
  $("#quick-demo").addEventListener("click", quickDemo);
  $("#top-build-dataset").addEventListener("click", () => {
    document.querySelector("#dataset-builder")?.scrollIntoView({ behavior: "smooth", block: "center" });
  });
  if (saved.lastBatch && Array.isArray(saved.lastBatch) && saved.lastBatch.length) {
    state.batch = saved.lastBatch;
    state.selected = Number(saved.selected) || 0;
    state.batchWarnings = saved.batchWarnings || [];
    state.batchMode = saved.batchMode || "single";
    state.batchStructures = saved.batchStructures || [];
    renderGallery(state.batch);
    renderSelection();
    setStatus("restored");
  } else {
    await generate();
  }
}

boot();
