import { loadUIState, saveUIState } from "./state.js";

let currentView = "data";
let uiMode = "simple";

function applyMode(mode, { persist = true } = {}) {
  uiMode = mode;
  document.body.classList.toggle("simple-mode", mode === "simple");
  const button = document.getElementById("mode-toggle");
  if (button) {
    button.textContent = mode === "simple" ? "Simple" : "Expert";
    button.classList.toggle("expert", mode === "expert");
    button.title = mode === "simple" ? "Switch to expert mode" : "Switch to simple mode";
  }
  if (persist) saveUIState({ uiMode: mode });
}

function setView(view, { persist = true, labTab = null } = {}) {
  currentView = view;
  document.getElementById("data-view")?.classList.toggle("hidden", view !== "data");
  document.getElementById("lab-view")?.classList.toggle("hidden", view !== "lab");
  document.getElementById("nav-data")?.classList.toggle("active", view === "data");
  document.getElementById("nav-lab")?.classList.toggle("active", view === "lab" && labTab !== "interpret");
  document.getElementById("interpret-nav")?.classList.toggle("active", view === "lab" && labTab === "interpret");
  if (view === "lab" && labTab) {
    window.dispatchEvent(new CustomEvent("llm-lab-tab", { detail: labTab }));
  }
  if (persist) {
    saveUIState({ appView: view });
  }
}

function locationView() {
  if (window.location.hash === "#interpret") return { view: "lab", labTab: "interpret" };
  if (window.location.pathname.startsWith("/lab")) return { view: "lab", labTab: "train" };
  return { view: "data", labTab: null };
}

function bind() {
  document.getElementById("nav-data")?.addEventListener("click", (event) => {
    event.preventDefault();
    setView("data");
    history.replaceState(null, "", "/");
  });
  document.getElementById("nav-lab")?.addEventListener("click", (event) => {
    event.preventDefault();
    setView("lab", { labTab: "train" });
    history.replaceState(null, "", "/lab");
  });
  document.getElementById("interpret-nav")?.addEventListener("click", (event) => {
    event.preventDefault();
    setView("lab", { labTab: "interpret" });
    history.replaceState(null, "", "/lab#interpret");
  });
  window.addEventListener("hashchange", () => {
    const target = locationView();
    setView(target.view, { persist: false, labTab: target.labTab });
  });
  window.addEventListener("popstate", () => {
    const target = locationView();
    setView(target.view, { persist: false, labTab: target.labTab });
  });
  document.getElementById("mode-toggle")?.addEventListener("click", () => {
    applyMode(uiMode === "simple" ? "expert" : "simple");
  });
}

async function boot() {
  bind();
  const state = await loadUIState();
  applyMode(state.uiMode === "expert" ? "expert" : "simple", { persist: false });
  const fromLocation = locationView();
  if (fromLocation.view === "lab" || window.location.hash) {
    setView(fromLocation.view, { persist: false, labTab: fromLocation.labTab });
  } else if (state.appView === "lab") {
    setView("lab", { persist: false });
  } else {
    setView("data", { persist: false });
  }
}

boot();
