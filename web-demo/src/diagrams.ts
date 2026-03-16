/**
 * Static architectural diagrams for the AttnRes web demo.
 *
 * These visualize the structural difference between standard residual
 * connections and Attention Residuals. All colors read from design tokens.
 */

import {
  getThemeTokens,
  getCtx,
  heatColor,
  withAlpha,
  drawArrow,
  drawBox,
} from "./canvas-utils.js";

// ─── Standard Residual Diagram ─────────────────────────────────────────

/**
 * Visualizes a standard residual connection through 4 layers.
 * Shows the fixed +1 weight at each connection.
 */
export function drawStandardResidual(canvasId: string) {
  const { ctx, w, h } = getCtx(canvasId);
  const tokens = getThemeTokens();
  const cx = w / 2;

  const layers = ["Input x", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Output"];
  const boxW = 100;
  const boxH = 28;
  const gap = (h - 40 - layers.length * boxH) / (layers.length - 1);

  const positions = layers.map((_, i) => ({
    x: cx - boxW / 2,
    y: 20 + i * (boxH + gap),
  }));

  // Draw skip connections (right side)
  for (let i = 0; i < layers.length - 1; i++) {
    const skipX = cx + boxW / 2 + 25;
    const y1 = positions[i].y + boxH / 2;
    const y2 = positions[i + 1].y + boxH / 2;

    // Curved skip connection
    ctx.strokeStyle = tokens.border;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cx + boxW / 2, y1);
    ctx.quadraticCurveTo(skipX, (y1 + y2) / 2, cx + boxW / 2, y2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Weight label
    ctx.fillStyle = tokens.textMuted;
    ctx.font = `400 10px ${tokens.fontMono}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("+1", skipX + 8, (y1 + y2) / 2);
  }

  // Draw main connections (arrows)
  for (let i = 0; i < layers.length - 1; i++) {
    drawArrow(
      ctx,
      cx,
      positions[i].y + boxH,
      cx,
      positions[i + 1].y,
      tokens.textMuted,
    );
  }

  // Draw boxes
  for (let i = 0; i < layers.length; i++) {
    const isEndpoint = i === 0 || i === layers.length - 1;
    drawBox(
      ctx,
      positions[i].x,
      positions[i].y,
      boxW,
      boxH,
      layers[i],
      isEndpoint ? tokens.accentLight : tokens.bgAlt,
      isEndpoint ? tokens.accent : tokens.border,
      tokens.text,
      tokens.fontSans,
    );
  }
}

// ─── Comparison Diagrams ───────────────────────────────────────────────

/**
 * Compact standard residual for comparison section.
 * Shows uniform weight distribution.
 */
export function drawComparisonStandard(canvasId: string) {
  const { ctx, w, h } = getCtx(canvasId);
  const tokens = getThemeTokens();

  const numBars = 4;
  const barW = (w - 80) / numBars;
  const maxH = h - 60;
  const startX = 40;
  const baseY = h - 30;

  // Equal bars
  for (let i = 0; i < numBars; i++) {
    const barH = maxH * 0.7; // All equal
    const x = startX + i * barW + 4;

    ctx.fillStyle = tokens.borderSubtle;
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.fill();

    ctx.strokeStyle = tokens.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.stroke();

    // Label
    ctx.fillStyle = tokens.textMuted;
    ctx.font = `400 10px ${tokens.fontMono}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`L${i + 1}`, x + (barW - 8) / 2, baseY + 6);

    // Weight
    ctx.textBaseline = "bottom";
    ctx.fillText("0.25", x + (barW - 8) / 2, baseY - barH - 4);
  }

  // Title
  ctx.fillStyle = tokens.textMuted;
  ctx.font = `500 10px ${tokens.fontMono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("uniform weights", w / 2, 8);
}

/**
 * Compact AttnRes for comparison section.
 * Shows non-uniform learned weight distribution.
 */
export function drawComparisonAttnRes(canvasId: string) {
  const { ctx, w, h } = getCtx(canvasId);
  const tokens = getThemeTokens();

  const weights = [0.12, 0.18, 0.35, 0.35]; // Learned non-uniform
  const numBars = weights.length;
  const barW = (w - 80) / numBars;
  const maxH = h - 60;
  const startX = 40;
  const baseY = h - 30;
  const maxWeight = Math.max(...weights);

  for (let i = 0; i < numBars; i++) {
    const barH = (weights[i] / maxWeight) * maxH * 0.85;
    const x = startX + i * barW + 4;

    // Color from shared heatColor using design tokens
    ctx.fillStyle = heatColor(weights[i] / maxWeight, tokens);
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.fill();

    ctx.strokeStyle = withAlpha(tokens.accent, 0.2);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.stroke();

    // Label
    ctx.fillStyle = tokens.textMuted;
    ctx.font = `400 10px ${tokens.fontMono}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(i === 0 ? "Emb" : `B${i}`, x + (barW - 8) / 2, baseY + 6);

    // Weight
    ctx.textBaseline = "bottom";
    ctx.fillStyle = tokens.textSecondary;
    ctx.fillText(weights[i].toFixed(2), x + (barW - 8) / 2, baseY - barH - 4);
  }

  // Title
  ctx.fillStyle = tokens.accent;
  ctx.font = `500 10px ${tokens.fontMono}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("learned weights", w / 2, 8);
}
