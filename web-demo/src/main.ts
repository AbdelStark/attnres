/**
 * Attention Residuals — Interactive Web Demo
 *
 * Loads the attnres-wasm WASM module and drives all interactive
 * visualizations. Mirrors the attnres crate's algorithm faithfully.
 */

import init, { AttnResEngine } from "../crate/pkg/attnres_wasm.js";
import { drawHeatmap, drawBarChart, drawLossCurve, drawNormsChart } from "./viz.js";
import { drawStandardResidual, drawComparisonStandard, drawComparisonAttnRes } from "./diagrams.js";

// ─── State ─────────────────────────────────────────────────────────────

let engine: AttnResEngine | null = null;
let trainingRafId: number | null = null;
let lastTrainTime = 0;
let lossHistory: number[] = [];
let normsHistory: number[][] = [];

const MAX_TRAINING_STEPS = 500;
const TRAIN_INTERVAL = 80;
const TRAIN_INTERVAL_REDUCED = 300;
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

// ─── Toast Notifications ───────────────────────────────────────────────

function showToast(message: string, type: "info" | "error" | "success" = "info", duration = 4000) {
  const container = document.getElementById("toast-container")!;
  const toast = document.createElement("div");
  toast.className = `toast${type !== "info" ? ` toast-${type}` : ""}`;
  toast.textContent = message;
  container.appendChild(toast);

  const dismiss = () => {
    toast.classList.add("toast-exit");
    toast.addEventListener("animationend", () => toast.remove(), { once: true });
  };

  toast.addEventListener("click", dismiss);
  setTimeout(dismiss, duration);
}

// ─── Navigation ────────────────────────────────────────────────────────

function setupNavigation() {
  const nav = document.querySelector(".nav")!;
  const toggle = document.querySelector(".nav-toggle") as HTMLButtonElement;
  const links = document.getElementById("nav-links")!;
  const navAnchors = links.querySelectorAll("a[href^='#']");

  // Scrolled state — add subtle shadow
  const updateScrollState = () => {
    nav.classList.toggle("scrolled", window.scrollY > 10);
  };
  window.addEventListener("scroll", updateScrollState, { passive: true });
  updateScrollState();

  // Mobile toggle
  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });

  // Close mobile nav helper
  const closeNav = () => {
    links.classList.remove("open");
    toggle.setAttribute("aria-expanded", "false");
  };

  // Close mobile nav on link click
  navAnchors.forEach((a) => a.addEventListener("click", closeNav));

  // Close mobile nav on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && links.classList.contains("open")) {
      closeNav();
      toggle.focus();
    }
  });

  // Active nav link via IntersectionObserver
  const sections = document.querySelectorAll<HTMLElement>("section[id]");
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const id = entry.target.id;
          navAnchors.forEach((a) => {
            a.classList.toggle("active", a.getAttribute("href") === `#${id}`);
          });
        }
      }
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );

  sections.forEach((s) => observer.observe(s));
}

// ─── Section Fade-In ───────────────────────────────────────────────────

function setupFadeIn() {
  const containers = document.querySelectorAll<HTMLElement>(".section .container");

  // Immediately show hero container
  const heroContainer = document.querySelector<HTMLElement>(".hero .container");
  if (heroContainer) heroContainer.classList.add("visible");

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          (entry.target as HTMLElement).classList.add("visible");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.1 }
  );

  containers.forEach((c) => observer.observe(c));
}

// ─── Initialization ────────────────────────────────────────────────────

async function main() {
  const statusEl = document.getElementById("wasm-status")!;

  // Start non-WASM setup immediately
  setupNavigation();
  setupFadeIn();

  try {
    await init();
    statusEl.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>WASM engine ready</span>`;
    showToast("WASM engine loaded successfully", "success", 2500);

    // Draw static diagrams
    drawStandardResidual("canvas-standard");
    drawComparisonStandard("canvas-cmp-standard");
    drawComparisonAttnRes("canvas-cmp-attnres");

    // Wire up controls
    setupDemoControls();
    setupTrainingControls();
  } catch (e) {
    statusEl.innerHTML = `<span class="status-dot error" aria-hidden="true"></span><span>Failed to load WASM engine</span>`;
    showToast(`WASM load error: ${e}`, "error", 8000);
    console.error(e);
  }
}

// ─── Demo Panel ────────────────────────────────────────────────────────

function setupDemoControls() {
  const btnInit = document.getElementById("btn-init-model") as HTMLButtonElement;
  const slider = document.getElementById("query-magnitude") as HTMLInputElement;
  const magDisplay = document.getElementById("query-mag-display")!;

  btnInit.addEventListener("click", () => {
    const dModel = parseInt((document.getElementById("cfg-d-model") as HTMLSelectElement).value);
    const numLayers = parseInt((document.getElementById("cfg-layers") as HTMLSelectElement).value);
    const numBlocks = parseInt((document.getElementById("cfg-blocks") as HTMLSelectElement).value);
    const numHeads = parseInt((document.getElementById("cfg-heads") as HTMLSelectElement).value);

    // Validate blocks divides layers
    if (numLayers % numBlocks !== 0) {
      showToast(`num_layers (${numLayers}) must be divisible by num_blocks (${numBlocks}). Try layers=8, blocks=2.`, "error");
      return;
    }
    if (dModel % numHeads !== 0) {
      showToast(`d_model (${dModel}) must be divisible by num_heads (${numHeads}). Try d_model=32, heads=4.`, "error");
      return;
    }

    try {
      engine = new AttnResEngine({
        d_model: dModel,
        num_layers: numLayers,
        num_blocks: numBlocks,
        num_heads: numHeads,
        vocab_size: 256,
      });

      document.getElementById("query-controls")!.classList.remove("hidden");
      document.getElementById("demo-results")!.classList.remove("hidden");
      document.getElementById("btn-train-start")!.removeAttribute("disabled");
      document.getElementById("btn-train-reset")!.removeAttribute("disabled");

      slider.value = "0";
      magDisplay.textContent = "0.00";
      slider.setAttribute("aria-valuenow", "0");
      slider.setAttribute("aria-valuetext", "0.00 (uniform)");

      updateDemo();
      showModelInfo();
      showToast(`Model initialized: ${numLayers} sublayers, ${numBlocks} blocks`, "success", 2500);
    } catch (e) {
      showToast(`Failed to create model: ${e}`, "error");
    }
  });

  slider.addEventListener("input", () => {
    const mag = parseInt(slider.value) / 100;
    magDisplay.textContent = mag.toFixed(2);
    slider.setAttribute("aria-valuenow", mag.toFixed(2));
    const label = mag === 0 ? "0.00 (uniform)" : mag >= 0.95 ? `${mag.toFixed(2)} (selective)` : mag.toFixed(2);
    slider.setAttribute("aria-valuetext", label);
    updateDemo();
  });
}

function showModelInfo() {
  if (!engine) return;
  const info = engine.model_info();
  const el = document.getElementById("model-info-display")!;

  const fields: [string, string | number][] = [
    ["d_model", info.d_model],
    ["sublayers", info.num_layers],
    ["transformer layers", info.num_transformer_layers],
    ["blocks", info.num_blocks],
    ["block size", info.block_size],
    ["heads", info.num_heads],
    ["d_ff", info.d_ff],
    ["AttnRes ops", info.total_attnres_ops],
    ["total params", info.total_params?.toLocaleString() ?? "—"],
    ["variant", info.is_full_attnres ? "Full" : "Block"],
  ];

  const grid = document.createElement("div");
  grid.className = "model-info-grid";
  for (const [k, v] of fields) {
    const key = document.createElement("span");
    key.className = "model-info-key";
    key.textContent = k;
    const val = document.createElement("span");
    val.className = "model-info-val";
    val.textContent = String(v);
    grid.appendChild(key);
    grid.appendChild(val);
  }
  el.replaceChildren(grid);
}

function updateDemo() {
  if (!engine) return;

  const mag = parseInt((document.getElementById("query-magnitude") as HTMLInputElement).value) / 100;

  // Reset and train to desired magnitude
  engine.reset_training();
  const stepsNeeded = Math.round(mag * 80);
  let result: { depth_weights: number[][] } | null = null;
  for (let i = 0; i < stepsNeeded; i++) {
    result = engine.train_step();
  }

  // If zero steps, do a forward pass to get uniform weights
  if (!result) {
    const dModel = parseInt((document.getElementById("cfg-d-model") as HTMLSelectElement).value);
    const input = new Array(dModel).fill(0.1);
    const fwd = engine.forward(input);
    result = { depth_weights: fwd.depth_weights };
  }

  // Reset training state (demo slider shouldn't affect training panel)
  engine.reset_training();

  // Draw heatmap
  drawHeatmap("canvas-heatmap", result.depth_weights);

  // Draw bar chart for the last sublayer
  if (result.depth_weights.length > 0) {
    const lastWeights = result.depth_weights[result.depth_weights.length - 1];
    drawBarChart("canvas-bar", lastWeights);
  }
}

// ─── Training Panel ────────────────────────────────────────────────────

function showTrainingCanvas(id: string, emptyId: string) {
  document.getElementById(id)?.classList.remove("hidden");
  document.getElementById(emptyId)?.classList.add("hidden");
}

function hideTrainingCanvas(id: string, emptyId: string) {
  document.getElementById(id)?.classList.add("hidden");
  document.getElementById(emptyId)?.classList.remove("hidden");
}

function trainingLoop(btnStart: HTMLButtonElement, timestamp: number) {
  if (!trainingRafId || !engine) return;

  const interval = prefersReducedMotion.matches ? TRAIN_INTERVAL_REDUCED : TRAIN_INTERVAL;
  if (timestamp - lastTrainTime < interval) {
    trainingRafId = requestAnimationFrame((t) => trainingLoop(btnStart, t));
    return;
  }
  lastTrainTime = timestamp;

  const snapshot = engine.train_step();

  // Update stats
  document.getElementById("train-step")!.textContent = String(snapshot.step);
  document.getElementById("train-loss")!.textContent = snapshot.loss.toFixed(4);

  // Record history
  lossHistory.push(snapshot.loss);
  normsHistory.push(snapshot.pseudo_query_norms);

  // Draw charts
  drawLossCurve("canvas-loss", lossHistory);
  drawHeatmap("canvas-train-heatmap", snapshot.depth_weights);
  drawNormsChart("canvas-norms", normsHistory);

  // Auto-stop at max steps to prevent unbounded memory growth
  if (snapshot.step >= MAX_TRAINING_STEPS) {
    trainingRafId = null;
    btnStart.textContent = "Training Complete";
    btnStart.setAttribute("aria-label", "Training complete — reset to restart");
    showToast(`Training stopped at ${MAX_TRAINING_STEPS} steps`, "info", 3000);
    return;
  }

  trainingRafId = requestAnimationFrame((t) => trainingLoop(btnStart, t));
}

function stopTraining() {
  if (trainingRafId) {
    cancelAnimationFrame(trainingRafId);
    trainingRafId = null;
  }
}

function setupTrainingControls() {
  const btnStart = document.getElementById("btn-train-start") as HTMLButtonElement;
  const btnReset = document.getElementById("btn-train-reset") as HTMLButtonElement;

  btnStart.addEventListener("click", () => {
    if (trainingRafId) {
      // Pause
      stopTraining();
      btnStart.textContent = "Resume Training";
      btnStart.setAttribute("aria-label", "Resume training simulation");
      return;
    }

    if (!engine) {
      // Auto-init with default config
      document.getElementById("btn-init-model")!.click();
    }

    btnStart.textContent = "Pause";
    btnStart.setAttribute("aria-label", "Pause training simulation");

    // Show canvases, hide empty states
    showTrainingCanvas("canvas-loss", "loss-empty");
    showTrainingCanvas("canvas-train-heatmap", "heatmap-empty");
    showTrainingCanvas("canvas-norms", "norms-empty");

    // Start rAF loop (synced to browser paint cycle)
    lastTrainTime = 0;
    trainingRafId = requestAnimationFrame((t) => trainingLoop(btnStart, t));
  });

  btnReset.addEventListener("click", () => {
    stopTraining();

    if (engine) {
      engine.reset_training();
    }

    lossHistory = [];
    normsHistory = [];

    document.getElementById("train-step")!.textContent = "0";
    document.getElementById("train-loss")!.textContent = "\u2014";
    btnStart.textContent = "Start Training";
    btnStart.setAttribute("aria-label", "Start training simulation");

    // Show empty states, hide canvases
    hideTrainingCanvas("canvas-loss", "loss-empty");
    hideTrainingCanvas("canvas-train-heatmap", "heatmap-empty");
    hideTrainingCanvas("canvas-norms", "norms-empty");

    // Clear canvases
    clearCanvas("canvas-loss");
    clearCanvas("canvas-train-heatmap");
    clearCanvas("canvas-norms");
  });
}

function clearCanvas(id: string) {
  const canvas = document.getElementById(id) as HTMLCanvasElement;
  if (!canvas) return;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ─── Cleanup ──────────────────────────────────────────────────────────

window.addEventListener("beforeunload", () => {
  stopTraining();
});

// ─── Entry Point ───────────────────────────────────────────────────────

main();
