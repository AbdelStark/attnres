/**
 * Attention Residuals — Interactive Web Demo
 *
 * Loads the attnres-wasm WASM module and drives all interactive
 * visualizations. Mirrors the attnres-rs crate's algorithm faithfully.
 */

import init, { AttnResEngine } from "../crate/pkg/attnres_wasm.js";
import { drawHeatmap, drawBarChart, drawLossCurve, drawNormsChart } from "./viz.js";
import { drawStandardResidual, drawComparisonStandard, drawComparisonAttnRes } from "./diagrams.js";

// ─── State ─────────────────────────────────────────────────────────────

let engine: AttnResEngine | null = null;
let trainingInterval: ReturnType<typeof setInterval> | null = null;
let lossHistory: number[] = [];
let normsHistory: number[][] = [];

// ─── Initialization ────────────────────────────────────────────────────

async function main() {
  const statusEl = document.getElementById("wasm-status")!;

  try {
    await init();
    statusEl.innerHTML = `<span class="status-dot"></span><span>WASM engine ready</span>`;

    // Draw static diagrams
    drawStandardResidual("canvas-standard");
    drawComparisonStandard("canvas-cmp-standard");
    drawComparisonAttnRes("canvas-cmp-attnres");

    // Wire up controls
    setupDemoControls();
    setupTrainingControls();
  } catch (e) {
    statusEl.innerHTML = `<span class="status-dot" style="background:#ef4444"></span><span>Failed to load WASM: ${e}</span>`;
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
      alert(`num_layers (${numLayers}) must be divisible by num_blocks (${numBlocks})`);
      return;
    }
    if (dModel % numHeads !== 0) {
      alert(`d_model (${dModel}) must be divisible by num_heads (${numHeads})`);
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

      document.getElementById("query-controls")!.style.display = "block";
      document.getElementById("demo-results")!.style.display = "grid";
      document.getElementById("btn-train-start")!.removeAttribute("disabled");
      document.getElementById("btn-train-reset")!.removeAttribute("disabled");

      slider.value = "0";
      magDisplay.textContent = "0.00";

      updateDemo();
      showModelInfo();
    } catch (e) {
      alert(`Failed to create model: ${e}`);
    }
  });

  slider.addEventListener("input", () => {
    const mag = parseInt(slider.value) / 100;
    magDisplay.textContent = mag.toFixed(2);
    updateDemo();
  });
}

function showModelInfo() {
  if (!engine) return;
  const info = engine.model_info();
  const el = document.getElementById("model-info-display")!;

  const fields = [
    ["d_model", info.d_model],
    ["sublayers", info.num_layers],
    ["transformer layers", info.num_transformer_layers],
    ["blocks", info.num_blocks],
    ["block size", info.block_size],
    ["heads", info.num_heads],
    ["d_ff", info.d_ff],
    ["AttnRes ops", info.total_attnres_ops],
    ["total params", info.total_params.toLocaleString()],
    ["variant", info.is_full_attnres ? "Full" : "Block"],
  ];

  el.innerHTML = `<div class="model-info-grid">${fields
    .map(
      ([k, v]) =>
        `<span class="model-info-key">${k}</span><span class="model-info-val">${v}</span>`
    )
    .join("")}</div>`;
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

function setupTrainingControls() {
  const btnStart = document.getElementById("btn-train-start") as HTMLButtonElement;
  const btnReset = document.getElementById("btn-train-reset") as HTMLButtonElement;

  btnStart.addEventListener("click", () => {
    if (trainingInterval) {
      // Stop
      clearInterval(trainingInterval);
      trainingInterval = null;
      btnStart.textContent = "Resume Training";
      return;
    }

    if (!engine) {
      // Auto-init with default config
      document.getElementById("btn-init-model")!.click();
    }

    btnStart.textContent = "Pause";

    trainingInterval = setInterval(() => {
      if (!engine) return;
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
    }, 80);
  });

  btnReset.addEventListener("click", () => {
    if (trainingInterval) {
      clearInterval(trainingInterval);
      trainingInterval = null;
    }

    if (engine) {
      engine.reset_training();
    }

    lossHistory = [];
    normsHistory = [];

    document.getElementById("train-step")!.textContent = "0";
    document.getElementById("train-loss")!.textContent = "\u2014";
    btnStart.textContent = "Start Training";

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

// ─── Entry Point ───────────────────────────────────────────────────────

main();
