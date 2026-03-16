/**
 * Shared canvas utilities for the AttnRes web demo.
 *
 * All colors are read from CSS custom properties at runtime,
 * ensuring canvas rendering stays synchronized with the design
 * tokens defined in style.css.
 */

// ─── Theme Detection ──────────────────────────────────────────────────

export function isDarkMode(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

// ─── CSS Token Reader ─────────────────────────────────────────────────

export interface ThemeTokens {
  text: string;
  textSecondary: string;
  textMuted: string;
  bg: string;
  bgAlt: string;
  surface: string;
  border: string;
  borderSubtle: string;
  accent: string;
  accentLight: string;
  accentDark: string;
  accentSubtle: string;
  heatmapCold: string;
  heatmapHot: string;
  fontSans: string;
  fontSerif: string;
  fontMono: string;
}

let _cachedTokens: ThemeTokens | null = null;
let _themeQuery: MediaQueryList | null = null;

function invalidateTokenCache() {
  _cachedTokens = null;
  _heatCacheKey = "";
}

/**
 * Read design tokens from CSS custom properties on :root.
 * Cached until the color scheme changes (light ↔ dark).
 */
export function getThemeTokens(): ThemeTokens {
  if (_cachedTokens) return _cachedTokens;

  const style = getComputedStyle(document.documentElement);
  const get = (name: string) => style.getPropertyValue(name).trim();

  _cachedTokens = {
    text: get("--color-text"),
    textSecondary: get("--color-text-secondary"),
    textMuted: get("--color-text-muted"),
    bg: get("--color-bg"),
    bgAlt: get("--color-bg-alt"),
    surface: get("--color-surface"),
    border: get("--color-border"),
    borderSubtle: get("--color-border-subtle"),
    accent: get("--color-accent"),
    accentLight: get("--color-accent-light"),
    accentDark: get("--color-accent-dark"),
    accentSubtle: get("--color-accent-subtle"),
    heatmapCold: get("--heatmap-cold"),
    heatmapHot: get("--heatmap-hot"),
    fontSans: get("--font-sans"),
    fontSerif: get("--font-serif"),
    fontMono: get("--font-mono"),
  };

  // Invalidate when theme changes (once)
  if (!_themeQuery) {
    _themeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    _themeQuery.addEventListener("change", invalidateTokenCache);
  }

  return _cachedTokens;
}

// ─── Color Utilities ──────────────────────────────────────────────────

function parseHex(hex: string): [number, number, number] {
  hex = hex.replace("#", "");
  return [
    parseInt(hex.slice(0, 2), 16),
    parseInt(hex.slice(2, 4), 16),
    parseInt(hex.slice(4, 6), 16),
  ];
}

function lerpRGB(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): string {
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

let _heatCold: [number, number, number] = [0, 0, 0];
let _heatHot: [number, number, number] = [0, 0, 0];
let _heatCacheKey = "";

/** Map a value in [0, 1] to a heatmap color using design tokens. */
export function heatColor(t: number, tokens: ThemeTokens): string {
  // Cache parsed RGB endpoints — only re-parse when tokens change
  const key = tokens.heatmapCold + tokens.heatmapHot;
  if (key !== _heatCacheKey) {
    _heatCold = parseHex(tokens.heatmapCold);
    _heatHot = parseHex(tokens.heatmapHot);
    _heatCacheKey = key;
  }
  t = Math.max(0, Math.min(1, t));
  return lerpRGB(_heatCold, _heatHot, t);
}

/** Create a semi-transparent version of a hex color. */
export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = parseHex(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ─── Canvas Setup ─────────────────────────────────────────────────────

/** Get canvas 2D context with DPI scaling applied. */
export function getCtx(canvasId: string): {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
  canvas: HTMLCanvasElement;
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
  return { ctx, w, h, canvas };
}

// ─── Drawing Primitives ───────────────────────────────────────────────

/** Fill a rounded rectangle path. */
export function fillRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  r = Math.min(r, w / 2, h / 2);
  if (r <= 0) {
    ctx.fillRect(x, y, w, h);
    return;
  }
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}

/** Draw an arrow between two points. */
export function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width: number = 1.5,
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
    y2 - headLen * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    x2 - headLen * Math.cos(angle + Math.PI / 6),
    y2 - headLen * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
}

/** Draw a labeled box with fill and border. */
export function drawBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  fillColor: string,
  borderColor: string,
  textColor: string,
  fontFamily: string,
) {
  ctx.fillStyle = fillColor;
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 6);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = textColor;
  ctx.font = `500 11px ${fontFamily}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + w / 2, y + h / 2);
}
