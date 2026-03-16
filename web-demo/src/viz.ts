/**
 * Canvas-based visualizations for the AttnRes web demo.
 *
 * All drawing uses the Canvas 2D API. Colors are read from CSS
 * design tokens via getThemeTokens() for theme synchronization.
 */

import {
  isDarkMode,
  getThemeTokens,
  getCtx,
  heatColor,
  withAlpha,
  fillRoundedRect,
} from "./canvas-utils.js";

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
  const tokens = getThemeTokens();

  if (weights.length === 0) return;

  const numRows = weights.length;
  const numCols = Math.max(...weights.map((row) => row.length));

  const labelW = 70;
  const labelH = 40;
  const padR = 60; // space for color scale
  const padB = 10;
  const cellGap = 1.5;

  const cellW = (w - labelW - padR) / numCols;
  const cellH = (h - labelH - padB) / numRows;

  // Find global max for color scaling
  const maxVal = Math.max(...weights.flat(), 0.001);

  // Draw cells
  for (let r = 0; r < numRows; r++) {
    for (let c = 0; c < weights[r].length; c++) {
      const val = weights[r][c];
      const norm = val / maxVal;

      ctx.fillStyle = heatColor(norm, tokens);
      fillRoundedRect(
        ctx,
        labelW + c * cellW + cellGap / 2,
        labelH + r * cellH + cellGap / 2,
        cellW - cellGap,
        cellH - cellGap,
        2,
      );

      // Value text (contrast-aware against heatmap cell background)
      if (cellW > 30 && cellH > 14) {
        const dark = isDarkMode();
        ctx.fillStyle = dark
          ? norm > 0.6
            ? "#1a1a1a"
            : "#ddd"
          : norm > 0.6
            ? "#fff"
            : "#333";
        ctx.font = `500 ${Math.min(11, cellH * 0.5)}px ${tokens.fontMono}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(
          val.toFixed(2),
          labelW + c * cellW + cellW / 2,
          labelH + r * cellH + cellH / 2,
        );
      }
    }
  }

  // Row labels (sublayer names)
  ctx.fillStyle = tokens.textMuted;
  ctx.font = `500 10px ${tokens.fontMono}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let r = 0; r < numRows; r++) {
    const layerIdx = Math.floor(r / 2);
    const sublayer = r % 2 === 0 ? "Attn" : "MLP";
    ctx.fillText(
      `L${layerIdx} ${sublayer}`,
      labelW - 6,
      labelH + r * cellH + cellH / 2,
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
  const scaleW = 12;
  const scaleH = h - labelH - padB;

  for (let y = 0; y < scaleH; y++) {
    const t = 1 - y / scaleH;
    ctx.fillStyle = heatColor(t, tokens);
    ctx.fillRect(scaleX, labelH + y, scaleW, 1);
  }

  ctx.fillStyle = tokens.textMuted;
  ctx.font = `400 9px ${tokens.fontMono}`;
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
  const tokens = getThemeTokens();

  if (weights.length === 0) return;

  const padL = 40;
  const padR = 20;
  const padT = 24;
  const padB = 40;

  const chartW = w - padL - padR;
  const chartH = h - padT - padB;
  const barW = chartW / weights.length;
  const gap = Math.max(4, barW * 0.2);

  const maxVal = Math.max(...weights, 0.001);

  // Draw bars
  for (let i = 0; i < weights.length; i++) {
    const barH = (weights[i] / maxVal) * chartH;
    const x = padL + i * barW + gap / 2;
    const y = padT + chartH - barH;

    ctx.fillStyle = heatColor(weights[i] / maxVal, tokens);
    fillRoundedRect(ctx, x, y, barW - gap, barH, 3);

    // Subtle stroke
    ctx.strokeStyle = withAlpha(tokens.accent, 0.2);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, barW - gap, barH, 3);
    ctx.stroke();

    // Value label
    ctx.fillStyle = tokens.text;
    ctx.font = `500 10px ${tokens.fontMono}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(weights[i].toFixed(3), x + (barW - gap) / 2, y - 4);

    // X label
    ctx.fillStyle = tokens.textMuted;
    ctx.textBaseline = "top";
    const label = i === 0 ? "Emb" : i === weights.length - 1 ? "Part" : `B${i}`;
    ctx.fillText(label, x + (barW - gap) / 2, padT + chartH + 8);
  }

  // Y axis
  ctx.strokeStyle = tokens.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + chartH);
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.stroke();

  // Y axis labels
  ctx.fillStyle = tokens.textMuted;
  ctx.font = `400 10px ${tokens.fontMono}`;
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
  const tokens = getThemeTokens();

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
  ctx.strokeStyle = tokens.borderSubtle;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + chartW, y);
    ctx.stroke();
  }

  // Loss curve
  ctx.strokeStyle = tokens.accent;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
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
  ctx.fillStyle = withAlpha(tokens.accent, isDarkMode() ? 0.08 : 0.05);
  ctx.fill();

  // Axes
  ctx.strokeStyle = tokens.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + chartH);
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.stroke();

  // Labels
  ctx.fillStyle = tokens.textMuted;
  ctx.font = `400 10px ${tokens.fontMono}`;
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
  const tokens = getThemeTokens();

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
  ctx.strokeStyle = tokens.borderSubtle;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (chartH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + chartW, y);
    ctx.stroke();
  }

  // Draw each series with distinct colors
  const colors = generateSeriesColors(numSeries);

  for (let s = 0; s < numSeries; s++) {
    ctx.strokeStyle = colors[s];
    ctx.lineWidth = 1.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalAlpha = 0.75;
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
  ctx.strokeStyle = tokens.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + chartH);
  ctx.lineTo(padL + chartW, padT + chartH);
  ctx.stroke();

  // Labels
  ctx.fillStyle = tokens.textMuted;
  ctx.font = `400 10px ${tokens.fontMono}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(maxVal.toFixed(2), padL - 4, padT);
  ctx.textBaseline = "bottom";
  ctx.fillText("0", padL - 4, padT + chartH);
}

/** Generate distinct chart series colors in the blue-purple range. */
function generateSeriesColors(n: number): string[] {
  const dark = isDarkMode();
  const colors: string[] = [];
  for (let i = 0; i < n; i++) {
    const hue = 210 + (i / n) * 120; // Blue to purple range (210-330)
    const saturation = dark ? 65 : 70;
    const lightness = dark ? 55 + (i % 2) * 10 : 35 + (i % 2) * 15;
    colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
  }
  return colors;
}
