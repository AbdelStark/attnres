/**
 * Canvas-based visualizations for the AttnRes web demo.
 *
 * All drawing is done with the Canvas 2D API for universal compatibility.
 * Heatmaps use a perceptually uniform blue gradient.
 */

// ─── Color Utilities ───────────────────────────────────────────────────

/** Map a value in [0, 1] to a blue heatmap color. */
function heatColor(t: number): string {
  // Perceptually uniform: white → light blue → deep blue
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(240 - t * 210);
  const g = Math.round(244 - t * 186);
  const b = Math.round(255 - t * 160);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Get canvas context with DPI scaling. */
function getCtx(canvasId: string): {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
  canvas: HTMLCanvasElement;
} {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;

  // Handle high-DPI displays
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.scale(dpr, dpr);
  }

  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, canvas };
}

// ─── Heatmap ───────────────────────────────────────────────────────────

/**
 * Draw a depth attention weight heatmap.
 *
 * Rows = sublayers (L0 Attn, L0 MLP, L1 Attn, ...)
 * Cols = source blocks (Embed, Block 1, ..., Partial)
 *
 * Brighter = higher attention weight.
 */
export function drawHeatmap(canvasId: string, weights: number[][]): void {
  const { ctx, w, h } = getCtx(canvasId);

  if (weights.length === 0) return;

  const numRows = weights.length;
  const numCols = Math.max(...weights.map((row) => row.length));

  const labelW = 70;
  const labelH = 40;
  const padR = 60; // space for color scale
  const padB = 10;

  const cellW = (w - labelW - padR) / numCols;
  const cellH = (h - labelH - padB) / numRows;

  // Find global max for color scaling
  const maxVal = Math.max(...weights.flat(), 0.001);

  // Draw cells
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < weights[r].length; c++) {
      const val = weights[r][c];
      const norm = val / maxVal;

      ctx.fillStyle = heatColor(norm);
      ctx.fillRect(labelW + c * cellW, labelH + r * cellH, cellW - 1, cellH - 1);

      // Value text
      if (cellW > 30 && cellH > 14) {
        ctx.fillStyle = norm > 0.6 ? "#fff" : "#333";
        ctx.font = `500 ${Math.min(11, cellH * 0.5)}px "JetBrains Mono", monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          val.toFixed(2),
          labelW + c * cellW + cellW / 2,
          labelH + r * cellH + cellH / 2
        );
      }
    }
  }

  // Row labels (sublayer names)
  ctx.fillStyle = "#888";
  ctx.font = '500 10px "JetBrains Mono", monospace';
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let r = 0; r < numRows; r++) {
    const layerIdx = Math.floor(r / 2);
    const sublayer = r % 2 === 0 ? "Attn" : "MLP";
    ctx.fillText(
      `L${layerIdx} ${sublayer}`,
      labelW - 6,
      labelH + r * cellH + cellH / 2
    );
  }

  // Column labels (source blocks)
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (let c = 0; c < numCols; c++) {
    const label = c === 0 ? "Emb" : c === numCols - 1 ? "Part" : `B${c}`;
    ctx.fillText(label, labelW + c * cellW + cellW / 2, labelH - 4);
  }

  // Color scale legend
  const scaleX = w - padR + 12;
  const scaleW = 14;
  const scaleH = h - labelH - padB;

  for (let y = 0; y < scaleH; y++) {
    const t = 1 - y / scaleH;
    ctx.fillStyle = heatColor(t);
    ctx.fillRect(scaleX, labelH + y, scaleW, 1);
  }

  ctx.fillStyle = "#888";
  ctx.font = '400 9px "JetBrains Mono", monospace';
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(maxVal.toFixed(2), scaleX + scaleW + 4, labelH);
  ctx.textBaseline = "bottom";
  ctx.fillText("0.00", scaleX + scaleW + 4, labelH + scaleH);
}

// ─── Bar Chart ─────────────────────────────────────────────────────────

/**
 * Draw a bar chart of attention weights for a single sublayer.
 */
export function drawBarChart(canvasId: string, weights: number[]): void {
  const { ctx, w, h } = getCtx(canvasId);

  if (weights.length === 0) return;

  const padL = 40;
  const padR = 20;
  const padT = 20;
  const padB = 40;

  const chartW = w - padL - padR;
  const chartH = h - padT - padB;
  const barW = chartW / weights.length;
  const gap = Math.max(2, barW * 0.15);

  const maxVal = Math.max(...weights, 0.001);

  // Draw bars
  for (let i = 0; i < weights.length; i++) {
    const barH = (weights[i] / maxVal) * chartH;
    const x = padL + i * barW + gap / 2;
    const y = padT + chartH - barH;

    ctx.fillStyle = heatColor(weights[i] / maxVal);
    ctx.fillRect(x, y, barW - gap, barH);

    // Border
    ctx.strokeStyle = "#2563eb33";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, barW - gap, barH);

    // Value label
    ctx.fillStyle = "#333";
    ctx.font = '500 10px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(weights[i].toFixed(3), x + (barW - gap) / 2, y - 3);

    // X label
    ctx.fillStyle = "#888";
    ctx.textBaseline = "top";
    const label = i === 0 ? "Emb" : i === weights.length - 1 ? "Part" : `B${i}`;
    ctx.fillText(label, x + (barW - gap) / 2, padT + chartH + 6);
  }

  // Y axis
  ctx.strokeStyle = "#ddd";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + chartH);
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.stroke();

  // Y axis label
  ctx.fillStyle = "#888";
  ctx.font = '400 10px "JetBrains Mono", monospace';
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(maxVal.toFixed(2), padL - 4, padT);
  ctx.textBaseline = "bottom";
  ctx.fillText("0", padL - 4, padT + chartH);
}

// ─── Loss Curve ────────────────────────────────────────────────────────

/**
 * Draw the training loss curve.
 */
export function drawLossCurve(canvasId: string, losses: number[]): void {
  const { ctx, w, h } = getCtx(canvasId);

  if (losses.length < 2) return;

  const padL = 50;
  const padR = 20;
  const padT = 20;
  const padB = 30;

  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  const maxLoss = Math.max(...losses) * 1.05;
  const minLoss = Math.min(...losses) * 0.95;
  const range = maxLoss - minLoss || 1;

  // Grid lines
  ctx.strokeStyle = "#eee";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + chartW, y);
    ctx.stroke();
  }

  // Loss curve
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();

  for (let i = 0; i < losses.length; i++) {
    const x = padL + (i / (losses.length - 1)) * chartW;
    const y = padT + ((maxLoss - losses[i]) / range) * chartH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill under curve
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.lineTo(padL, padT + chartH);
  ctx.closePath();
  ctx.fillStyle = "rgba(37, 99, 235, 0.05)";
  ctx.fill();

  // Axes
  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + chartH);
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#888";
  ctx.font = '400 10px "JetBrains Mono", monospace';
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(maxLoss.toFixed(2), padL - 4, padT);
  ctx.textBaseline = "bottom";
  ctx.fillText(minLoss.toFixed(2), padL - 4, padT + chartH);

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Step " + losses.length, padL + chartW, padT + chartH + 6);
  ctx.textAlign = "left";
  ctx.fillText("0", padL, padT + chartH + 6);
}

// ─── Norms Chart ───────────────────────────────────────────────────────

/**
 * Draw pseudo-query norm evolution as a multi-line chart.
 * Each line represents one sublayer's ||w_l|| over training steps.
 */
export function drawNormsChart(canvasId: string, history: number[][]): void {
  const { ctx, w, h } = getCtx(canvasId);

  if (history.length < 2) return;

  const padL = 50;
  const padR = 20;
  const padT = 20;
  const padB = 30;

  const chartW = w - padL - padR;
  const chartH = h - padT - padB;

  const numSeries = history[0].length;
  const allVals = history.flat();
  const maxVal = Math.max(...allVals, 0.001) * 1.05;

  // Grid
  ctx.strokeStyle = "#eee";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + chartW, y);
    ctx.stroke();
  }

  // Draw each series with distinct colors
  const colors = generateColors(numSeries);

  for (let s = 0; s < numSeries; s++) {
    ctx.strokeStyle = colors[s];
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();

    for (let t = 0; t < history.length; t++) {
      const x = padL + (t / (history.length - 1)) * chartW;
      const y = padT + ((maxVal - history[t][s]) / maxVal) * chartH;
      if (t === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  ctx.globalAlpha = 1.0;

  // Axes
  ctx.strokeStyle = "#ccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + chartH);
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.stroke();

  // Labels
  ctx.fillStyle = "#888";
  ctx.font = '400 10px "JetBrains Mono", monospace';
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(maxVal.toFixed(2), padL - 4, padT);
  ctx.textBaseline = "bottom";
  ctx.fillText("0", padL - 4, padT + chartH);
}

function generateColors(n: number): string[] {
  const colors: string[] = [];
  for (let i = 0; i < n; i++) {
    const hue = (i / n) * 280; // Blue to purple range
    const lightness = 35 + (i % 2) * 15;
    colors.push(`hsl(${hue}, 70%, ${lightness}%)`);
  }
  return colors;
}
