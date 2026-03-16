/**
 * Static architectural diagrams for the AttnRes web demo.
 *
 * These visualize the structural difference between standard residual
 * connections and Attention Residuals.
 */

// ─── Helpers ───────────────────────────────────────────────────────────

function isDarkMode(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function getCtx(canvasId: string): {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
} {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;

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
  return { ctx, w, h };
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width: number = 1.5
) {
  const headLen = 7;
  const angle = Math.atan2(y2 - y1, x2 - x1);

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headLen * Math.cos(angle - Math.PI / 6),
    y2 - headLen * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x2 - headLen * Math.cos(angle + Math.PI / 6),
    y2 - headLen * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  fillColor: string,
  borderColor: string,
  textColor: string = isDarkMode() ? "#ededef" : "#333"
) {
  ctx.fillStyle = fillColor;
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = textColor;
  ctx.font = '500 11px "Inter", sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + w / 2, y + h / 2);
}

// ─── Standard Residual Diagram ─────────────────────────────────────────

/**
 * Visualizes a standard residual connection through 4 layers.
 * Shows the fixed +1 weight at each connection.
 */
export function drawStandardResidual(canvasId: string) {
  const { ctx, w, h } = getCtx(canvasId);
  const dark = isDarkMode();
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
    ctx.strokeStyle = dark ? "#3a3a40" : "#ddd";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(cx + boxW / 2, y1);
    ctx.quadraticCurveTo(skipX, (y1 + y2) / 2, cx + boxW / 2, y2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Weight label
    ctx.fillStyle = dark ? "#555" : "#bbb";
    ctx.font = '400 10px "JetBrains Mono", monospace';
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
      dark ? "#555" : "#999"
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
      dark
        ? (isEndpoint ? "#1e3a5f" : "#232326")
        : (isEndpoint ? "#e8edf5" : "#f5f5f5"),
      dark
        ? (isEndpoint ? "#3b82f6" : "#3a3a40")
        : (isEndpoint ? "#94a3b8" : "#ddd")
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
  const dark = isDarkMode();

  const numBars = 4;
  const barW = (w - 80) / numBars;
  const maxH = h - 60;
  const startX = 40;
  const baseY = h - 30;

  // Equal bars
  for (let i = 0; i < numBars; i++) {
    const barH = maxH * 0.7; // All equal
    const x = startX + i * barW + 4;

    ctx.fillStyle = dark ? "#2e2e32" : "#e0e0e0";
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.fill();

    ctx.strokeStyle = dark ? "#3a3a40" : "#ccc";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.stroke();

    // Label
    ctx.fillStyle = dark ? "#6e6e76" : "#999";
    ctx.font = '400 10px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`L${i + 1}`, x + (barW - 8) / 2, baseY + 6);

    // Weight
    ctx.textBaseline = "bottom";
    ctx.fillText("0.25", x + (barW - 8) / 2, baseY - barH - 4);
  }

  // Title
  ctx.fillStyle = dark ? "#6e6e76" : "#888";
  ctx.font = '500 10px "JetBrains Mono", monospace';
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
  const dark = isDarkMode();

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

    // Gradient fill matching heatmap
    const t = weights[i] / maxWeight;
    let r: number, g: number, b: number;
    if (dark) {
      r = Math.round(26 + t * 121);
      g = Math.round(29 + t * 168);
      b = Math.round(46 + t * 207);
    } else {
      r = Math.round(240 - t * 210);
      g = Math.round(244 - t * 186);
      b = Math.round(255 - t * 160);
    }
    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.fill();

    ctx.strokeStyle = dark ? "rgba(96, 165, 250, 0.2)" : "#2563eb33";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x, baseY - barH, barW - 8, barH, 3);
    ctx.stroke();

    // Label
    ctx.fillStyle = dark ? "#6e6e76" : "#888";
    ctx.font = '400 10px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(i === 0 ? "Emb" : `B${i}`, x + (barW - 8) / 2, baseY + 6);

    // Weight
    ctx.textBaseline = "bottom";
    ctx.fillStyle = dark ? "#a1a1a6" : "#555";
    ctx.fillText(weights[i].toFixed(2), x + (barW - 8) / 2, baseY - barH - 4);
  }

  // Title
  ctx.fillStyle = dark ? "#60a5fa" : "#2563eb";
  ctx.font = '500 10px "JetBrains Mono", monospace';
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("learned weights", w / 2, 8);
}
