let cache = null;
let saveTimer = null;

export async function loadUIState() {
  if (cache) return cache;
  try {
    const response = await fetch("/api/ui-state");
    cache = response.ok ? await response.json() : {};
  } catch (_error) {
    cache = {};
  }
  return cache;
}

export function saveUIState(patch) {
  if (!cache) cache = {};
  Object.assign(cache, patch);
  window.clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => {
    fetch("/api/ui-state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).catch(() => {});
  }, 250);
}

export function collectById(rootSelector) {
  const root = document.querySelector(rootSelector);
  if (!root) return {};
  const state = {};
  for (const element of root.querySelectorAll(
    "input[name], select[name], textarea[name], input[id], select[id], textarea[id]"
  )) {
    if (element.type === "button" || element.type === "submit") continue;
    const key = element.id || `name:${element.name}`;
    if (!key || key === "name:") continue;
    state[key] = {
      value: element.value,
      checked: Boolean(element.checked),
      type: element.type || element.tagName.toLowerCase(),
    };
  }
  return state;
}

export function applyById(state) {
  if (!state) return;
  for (const [key, entry] of Object.entries(state)) {
    const element = key.startsWith("name:")
      ? document.querySelector(`[name="${CSS.escape(key.slice(5))}"]`)
      : document.getElementById(key);
    if (!element) continue;
    if (entry.type === "checkbox" || element.type === "checkbox") {
      element.checked = Boolean(entry.checked);
    } else {
      element.value = entry.value;
    }
  }
}
