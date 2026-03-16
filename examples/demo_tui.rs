//! Interactive TUI demo of the Attention Residuals training process.
//!
//! Visualizes in real time:
//! - Training loss curve converging
//! - Depth attention weights evolving from uniform to selective
//! - Pseudo-query norm growth per sublayer
//! - Algorithm steps explained as training progresses
//!
//! Controls:
//!   Space    Start / Pause / Resume
//!   Up/Down  Adjust training speed
//!   r        Reset model and restart
//!   q / Esc  Quit
//!
//! Run with: `cargo run --example demo_tui`

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

// Explicit ratatui imports to avoid Backend name collision with burn
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph};
use ratatui::{Frame, Terminal};

use burn::backend::{Autodiff, NdArray};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::Distribution;

use attnres::{causal_mask, AttnResConfig, AttnResOp, AttnResTransformer, BlockState};

type B = NdArray;
type AB = Autodiff<B>;

// ─── Colors ───────────────────────────────────────────────────────────

const ACCENT: Color = Color::Rgb(59, 130, 246);
const ACCENT_DIM: Color = Color::Rgb(30, 64, 175);
const TEXT_DIM: Color = Color::Rgb(148, 163, 184);
const TEXT_MUTED: Color = Color::Rgb(100, 116, 139);
const GREEN: Color = Color::Rgb(34, 197, 94);
const YELLOW: Color = Color::Rgb(234, 179, 8);
const BORDER: Color = Color::Rgb(51, 65, 85);

/// Map [0, 1] to a blue heatmap color (dark navy -> bright blue).
fn heat_color(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = (15.0 + t * 44.0) as u8;
    let g = (23.0 + t * 107.0) as u8;
    let b = (42.0 + t * 204.0) as u8;
    Color::Rgb(r, g, b)
}

// ─── Algorithm Steps ──────────────────────────────────────────────────

const ALGO_STEPS: [(&str, &str); 5] = [
    (
        "Stack block representations",
        "V = [b\u{2080}; b\u{2081}; \u{2026}; b\u{2099}] \u{2208} \u{211d}^{(N+1)\u{00d7}D} \u{2014} collect all completed blocks + current partial",
    ),
    (
        "Normalize keys with RMSNorm",
        "K = RMSNorm(V) \u{2014} prevents large-magnitude blocks from dominating attention logits",
    ),
    (
        "Compute depth attention logits",
        "logits\u{1d62} = K\u{1d62} \u{00b7} w\u{2097}  \u{2200}i \u{2208} {0,\u{2026},N} \u{2014} zero-init pseudo-query scores each block",
    ),
    (
        "Softmax over depth dimension",
        "\u{03b1} = softmax(logits) \u{2014} attention over layers (depth), NOT over tokens (sequence)",
    ),
    (
        "Weighted combination",
        "h = \u{03a3} \u{03b1}\u{1d62} \u{00b7} V\u{1d62} \u{2014} learned convex combination replaces fixed residual",
    ),
];

// ─── App State ────────────────────────────────────────────────────────

struct App {
    // Config
    d_model: usize,
    num_layers: usize,
    num_transformer_layers: usize,
    num_blocks: usize,
    num_heads: usize,
    vocab_size: usize,
    d_ff: usize,

    // Training
    step: usize,
    max_steps: usize,
    loss_history: Vec<f64>,
    depth_weights: Vec<Vec<f32>>,
    pseudo_query_norms: Vec<f32>,

    // UI
    paused: bool,
    speed: usize,
    algo_phase: usize,
    completed: bool,
}

impl App {
    fn new(config: &AttnResConfig) -> Self {
        Self {
            d_model: config.d_model,
            num_layers: config.num_layers,
            num_transformer_layers: config.num_transformer_layers(),
            num_blocks: config.num_blocks,
            num_heads: config.num_heads,
            vocab_size: config.vocab_size,
            d_ff: config.effective_d_ff(),
            step: 0,
            max_steps: 200,
            loss_history: Vec::with_capacity(200),
            depth_weights: Vec::new(),
            pseudo_query_norms: Vec::new(),
            paused: true,
            speed: 3,
            algo_phase: 0,
            completed: false,
        }
    }

    fn reset(&mut self) {
        self.step = 0;
        self.loss_history.clear();
        self.depth_weights.clear();
        self.pseudo_query_norms.clear();
        self.paused = true;
        self.algo_phase = 0;
        self.completed = false;
    }

    fn record_step(&mut self, loss: f32, weights: Vec<Vec<f32>>, norms: Vec<f32>) {
        self.loss_history.push(loss as f64);
        self.depth_weights = weights;
        self.pseudo_query_norms = norms;
        self.step += 1;
        self.algo_phase = (self.step / 8) % ALGO_STEPS.len();
        if self.step >= self.max_steps {
            self.completed = true;
            self.paused = true;
        }
    }
}

// ─── Training ─────────────────────────────────────────────────────────

fn train_step(
    model: AttnResTransformer<AB>,
    optim: &mut impl Optimizer<AttnResTransformer<AB>, AB>,
    mask: &Tensor<AB, 3>,
    config: &AttnResConfig,
    device: &<AB as burn::prelude::Backend>::Device,
) -> (AttnResTransformer<AB>, f32) {
    let batch_size = 2;
    let seq_len = 16;

    let input = Tensor::<AB, 2, Int>::random(
        [batch_size, seq_len],
        Distribution::Uniform(0.0, config.vocab_size as f64),
        device,
    );
    let targets = Tensor::<AB, 2, Int>::random(
        [batch_size, seq_len],
        Distribution::Uniform(0.0, config.vocab_size as f64),
        device,
    );

    let logits = model.forward(input, Some(mask));
    let [b, t, v] = logits.dims();

    let loss_fn = CrossEntropyLossConfig::new()
        .with_logits(true)
        .init(device);
    let loss = loss_fn
        .forward(logits.reshape([b * t, v]), targets.reshape([b * t]))
        .mean();

    let loss_val: f32 = loss.clone().into_scalar();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optim.step(0.001, model, grads);

    (model, loss_val)
}

// ─── Diagnostics ──────────────────────────────────────────────────────

/// Compute attention alpha weights for a single AttnRes operation.
fn compute_alpha(
    op: &AttnResOp<AB>,
    blocks: &[Tensor<AB, 3>],
    partial: &Tensor<AB, 3>,
) -> Vec<f32> {
    let mut sources: Vec<Tensor<AB, 3>> = blocks.to_vec();
    sources.push(partial.clone());
    let n = sources.len();

    let v = Tensor::stack(sources, 0); // [N+1, B, T, D]
    let k = op.norm.forward_4d(v); // [N+1, B, T, D]

    let w = op
        .pseudo_query
        .val()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0); // [1, 1, 1, D]
    let logits = (k * w).sum_dim(3).squeeze_dim::<3>(3); // [N+1, B, T]
    let alpha = softmax(logits, 0); // [N+1, B, T]

    // Average over batch and sequence -> [N+1]
    alpha
        .mean_dim(2)
        .squeeze_dim::<2>(2)
        .mean_dim(1)
        .squeeze_dim::<1>(1)
        .reshape([n])
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// Extract depth attention weights and pseudo-query norms from the model.
fn extract_diagnostics(
    model: &AttnResTransformer<AB>,
    device: &<AB as burn::prelude::Backend>::Device,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    // Small probe input for computing actual depth weights
    let probe = Tensor::<AB, 2, Int>::zeros([1, 4], device);
    let hidden = model.embed_tokens(probe);
    let mut state = BlockState::new(hidden);

    let mut depth_weights = Vec::new();
    let mut norms = Vec::new();

    for layer in model.layers() {
        let (attn_res, mlp_res) = layer.attn_res_ops();

        // Pseudo-query norms (attn + mlp sublayers)
        let norm_a: f32 = attn_res
            .pseudo_query
            .val()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .into_scalar();
        let norm_m: f32 = mlp_res
            .pseudo_query
            .val()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .into_scalar();
        norms.push(norm_a);
        norms.push(norm_m);

        // Replicate boundary handling from layer.forward() to get
        // the correct block state for alpha computation.
        let current_partial = state
            .partial_block
            .clone()
            .unwrap_or_else(|| Tensor::zeros_like(state.blocks.last().unwrap()));

        let at_boundary = layer.is_at_boundary();
        let mut blocks_snap = state.blocks.clone();
        if at_boundary {
            blocks_snap.push(current_partial.clone());
        }
        let partial_snap = if at_boundary {
            Tensor::zeros_like(blocks_snap.last().unwrap())
        } else {
            current_partial
        };

        // Compute actual attention weights for the attn sublayer
        let alpha = compute_alpha(attn_res, &blocks_snap, &partial_snap);
        depth_weights.push(alpha);

        // Run the real forward to advance block state
        state = layer.forward(state, None);
    }

    (depth_weights, norms)
}

// ─── Entry Point ──────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run(&mut terminal);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(ref e) = result {
        eprintln!("Error: {e}");
    }
    result
}

fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();
    let config = AttnResConfig::new(32, 8, 2)
        .with_num_heads(4)
        .with_vocab_size(256)
        .with_d_ff(64);

    let mut model: AttnResTransformer<AB> = config.init_model(&device);
    let mut optim = AdamConfig::new().init();
    let mask = causal_mask::<AB>(2, 16, &device);
    let mut app = App::new(&config);

    // Initial diagnostics (shows uniform weights at step 0)
    let (weights, norms) = extract_diagnostics(&model, &device);
    app.depth_weights = weights;
    app.pseudo_query_norms = norms;

    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        // Non-blocking event poll (~60fps)
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char(' ') => {
                            if app.completed {
                                model = config.init_model(&device);
                                optim = AdamConfig::new().init();
                                app.reset();
                                let (w, n) = extract_diagnostics(&model, &device);
                                app.depth_weights = w;
                                app.pseudo_query_norms = n;
                            }
                            app.paused = !app.paused;
                        }
                        KeyCode::Up => app.speed = (app.speed + 1).min(5),
                        KeyCode::Down => app.speed = app.speed.saturating_sub(1).max(1),
                        KeyCode::Char('r') => {
                            model = config.init_model(&device);
                            optim = AdamConfig::new().init();
                            app.reset();
                            let (w, n) = extract_diagnostics(&model, &device);
                            app.depth_weights = w;
                            app.pseudo_query_norms = n;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Training tick
        if !app.paused && app.step < app.max_steps {
            let tick_ms: u64 = match app.speed {
                1 => 250,
                2 => 150,
                3 => 80,
                4 => 40,
                _ => 15,
            };
            if last_tick.elapsed() >= Duration::from_millis(tick_ms) {
                let (new_model, loss) =
                    train_step(model, &mut optim, &mask, &config, &device);
                model = new_model;

                let (weights, norms) = extract_diagnostics(&model, &device);
                app.record_step(loss, weights, norms);

                last_tick = Instant::now();
            }
        }
    }
}

// ─── UI ───────────────────────────────────────────────────────────────

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // Title
            Constraint::Length(8),  // Model + Training
            Constraint::Length(10), // Loss curve
            Constraint::Min(8),    // Weights + Norms
            Constraint::Length(4),  // Algorithm
            Constraint::Length(1),  // Footer
        ])
        .split(f.area());

    draw_title(f, chunks[0]);
    draw_top_panels(f, chunks[1], app);
    draw_loss_chart(f, chunks[2], app);
    draw_bottom_panels(f, chunks[3], app);
    draw_algorithm(f, chunks[4], app);
    draw_footer(f, chunks[5], app);
}

fn draw_title(f: &mut Frame, area: Rect) {
    let title = Line::from(vec![
        Span::styled(" \u{03b1} ", Style::default().fg(Color::White).bg(ACCENT).bold()),
        Span::raw("  "),
        Span::styled("AttnRes", Style::default().fg(Color::White).bold()),
        Span::styled(
            " \u{2014} Interactive Training Demo",
            Style::default().fg(TEXT_DIM),
        ),
    ]);
    f.render_widget(Paragraph::new(title), area);
}

fn draw_top_panels(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);

    // Model info
    let model_text = vec![
        Line::from(vec![
            Span::styled("  d_model     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(format!("{}", app.d_model), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  layers      ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} ({} transformer)",
                    app.num_layers, app.num_transformer_layers
                ),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  blocks      ", Style::default().fg(TEXT_MUTED)),
            Span::styled(format!("{}", app.num_blocks), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  heads       ", Style::default().fg(TEXT_MUTED)),
            Span::styled(format!("{}", app.num_heads), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  d_ff        ", Style::default().fg(TEXT_MUTED)),
            Span::styled(format!("{}", app.d_ff), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  vocab       ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{}", app.vocab_size),
                Style::default().fg(Color::White),
            ),
        ]),
    ];
    let model_block = Paragraph::new(model_text).block(
        Block::default()
            .title(Span::styled(
                " Model ",
                Style::default().fg(ACCENT).bold(),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER)),
    );
    f.render_widget(model_block, chunks[0]);

    // Training info
    let loss_str = app
        .loss_history
        .last()
        .map_or("\u{2014}".to_string(), |v| format!("{v:.4}"));

    let (status_color, status_text) = if app.completed {
        (GREEN, "Complete")
    } else if app.paused {
        (YELLOW, "Paused")
    } else {
        (ACCENT, "Training")
    };

    let speed_dots: String = (1..=5)
        .map(|i| if i <= app.speed { '\u{25cf}' } else { '\u{25cb}' })
        .collect();

    let pct = if app.max_steps > 0 {
        (app.step * 100) / app.max_steps
    } else {
        0
    };
    let bar_w = (chunks[1].width as usize).saturating_sub(14);
    let filled = (bar_w * app.step) / app.max_steps.max(1);
    let bar = format!(
        "  {}{}  {}%",
        "\u{2588}".repeat(filled),
        "\u{2591}".repeat(bar_w.saturating_sub(filled)),
        pct,
    );

    let train_text = vec![
        Line::from(vec![
            Span::styled("  Status  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(status_text, Style::default().fg(status_color).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Step    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{}", app.step),
                Style::default().fg(Color::White).bold(),
            ),
            Span::styled(
                format!(" / {}", app.max_steps),
                Style::default().fg(TEXT_MUTED),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Loss    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(loss_str, Style::default().fg(Color::White).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Speed   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(speed_dots, Style::default().fg(ACCENT)),
            Span::styled(format!(" {}x", app.speed), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(""),
        Line::from(Span::styled(bar, Style::default().fg(ACCENT_DIM))),
    ];

    let train_block = Paragraph::new(train_text).block(
        Block::default()
            .title(Span::styled(
                " Training ",
                Style::default().fg(ACCENT).bold(),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER)),
    );
    f.render_widget(train_block, chunks[1]);
}

fn draw_loss_chart(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            " Loss Curve ",
            Style::default().fg(ACCENT).bold(),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER));

    if app.loss_history.is_empty() {
        let empty = Paragraph::new(Span::styled(
            "  Press Space to start training...",
            Style::default().fg(TEXT_MUTED).italic(),
        ))
        .block(block);
        f.render_widget(empty, area);
        return;
    }

    let loss_data: Vec<(f64, f64)> = app
        .loss_history
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();

    let min_loss = app
        .loss_history
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        * 0.95;
    let max_loss = app
        .loss_history
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        * 1.05;

    let dataset = Dataset::default()
        .name("loss")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(ACCENT))
        .data(&loss_data);

    let chart = Chart::new(vec![dataset])
        .block(block)
        .x_axis(
            Axis::default()
                .title(Span::styled("step", Style::default().fg(TEXT_MUTED)))
                .style(Style::default().fg(TEXT_MUTED))
                .bounds([0.0, app.max_steps as f64])
                .labels::<Vec<Line>>(vec![
                    Line::from("0"),
                    Line::from(format!("{}", app.max_steps / 2)),
                    Line::from(format!("{}", app.max_steps)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(Span::styled("loss", Style::default().fg(TEXT_MUTED)))
                .style(Style::default().fg(TEXT_MUTED))
                .bounds([min_loss, max_loss])
                .labels::<Vec<Line>>(vec![
                    Line::from(format!("{:.2}", min_loss)),
                    Line::from(format!("{:.2}", max_loss)),
                ]),
        );

    f.render_widget(chart, area);
}

fn draw_bottom_panels(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    draw_depth_weights(f, chunks[0], app);
    draw_norms(f, chunks[1], app);
}

fn draw_depth_weights(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            " Depth Attention Weights ",
            Style::default().fg(ACCENT).bold(),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER));

    if app.depth_weights.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  Initializing...",
                Style::default().fg(TEXT_MUTED),
            ))
            .block(block),
            area,
        );
        return;
    }

    let num_sources = app.depth_weights.first().map_or(0, |w| w.len());

    // Header row
    let mut header_spans = vec![Span::styled("          ", Style::default())];
    for c in 0..num_sources {
        let label = if c == 0 {
            "Emb".to_string()
        } else if c == num_sources - 1 {
            "Part".to_string()
        } else {
            format!("B{c}")
        };
        header_spans.push(Span::styled(
            format!("{:>8}", label),
            Style::default().fg(TEXT_DIM),
        ));
    }

    let mut lines = vec![Line::from(header_spans)];

    // One row per transformer layer (attn sublayer alpha)
    for (i, weights) in app.depth_weights.iter().enumerate() {
        let label = format!("  L{i} Attn");
        let mut spans = vec![Span::styled(
            format!("{:<10}", label),
            Style::default().fg(TEXT_MUTED),
        )];

        let max_w = weights.iter().cloned().fold(0.0f32, f32::max);
        for &w in weights {
            let t = if max_w > 0.001 { w / max_w } else { 0.5 };
            // Colored cell: weight value on heatmap background
            let fg = if t > 0.7 {
                Color::White
            } else {
                Color::Rgb(200, 210, 230)
            };
            spans.push(Span::styled(
                format!("  {:.3} ", w),
                Style::default().fg(fg).bg(heat_color(t)),
            ));
        }
        lines.push(Line::from(spans));
    }

    // Insight annotation
    if app.step > 0 {
        lines.push(Line::from(""));
        let max_norm = app
            .pseudo_query_norms
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        let insight = if max_norm < 0.05 {
            "\u{25b6} Near-uniform: pseudo-queries still close to zero init"
        } else if max_norm < 0.15 {
            "\u{25b6} Emerging: deeper layers starting to differentiate"
        } else {
            "\u{25b6} Selective: clear preferences for specific blocks"
        };
        lines.push(Line::from(Span::styled(
            format!("  {insight}"),
            Style::default().fg(TEXT_DIM).italic(),
        )));
    }

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_norms(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(Span::styled(
            " \u{2016}w\u{2097}\u{2016} Norms ",
            Style::default().fg(ACCENT).bold(),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER));

    if app.pseudo_query_norms.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  Initializing...",
                Style::default().fg(TEXT_MUTED),
            ))
            .block(block),
            area,
        );
        return;
    }

    let max_norm = app
        .pseudo_query_norms
        .iter()
        .cloned()
        .fold(0.001f32, f32::max);
    let bar_width = (area.width as usize).saturating_sub(20);

    let mut lines = Vec::new();
    for (i, &norm) in app.pseudo_query_norms.iter().enumerate() {
        let layer = i / 2;
        let sublayer = if i % 2 == 0 { "Attn" } else { "MLP " };
        let label = format!("  L{layer} {sublayer} ");

        let fill = ((norm / max_norm) * bar_width as f32).round() as usize;
        let bar: String = "\u{2588}".repeat(fill.min(bar_width));

        lines.push(Line::from(vec![
            Span::styled(label, Style::default().fg(TEXT_MUTED)),
            Span::styled(bar, Style::default().fg(ACCENT)),
            Span::styled(
                format!(" {:.3}", norm),
                Style::default().fg(TEXT_DIM),
            ),
        ]));
    }

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_algorithm(f: &mut Frame, area: Rect, app: &App) {
    let (title, desc) = ALGO_STEPS[app.algo_phase];

    let lines = vec![
        Line::from(vec![
            Span::styled("  \u{25b6} Step ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{}: ", app.algo_phase + 1),
                Style::default().fg(ACCENT).bold(),
            ),
            Span::styled(title, Style::default().fg(Color::White).bold()),
        ]),
        Line::from(Span::styled(
            format!("    {desc}"),
            Style::default().fg(TEXT_DIM),
        )),
    ];

    let block = Block::default()
        .title(Span::styled(
            " Algorithm ",
            Style::default().fg(ACCENT).bold(),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER));

    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_footer(f: &mut Frame, area: Rect, _app: &App) {
    let footer = Line::from(vec![
        Span::styled(" Space", Style::default().fg(Color::White).bold()),
        Span::styled(" Start/Pause  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("\u{2191}\u{2193}", Style::default().fg(Color::White).bold()),
        Span::styled(" Speed  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("r", Style::default().fg(Color::White).bold()),
        Span::styled(" Reset  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("q", Style::default().fg(Color::White).bold()),
        Span::styled(" Quit", Style::default().fg(TEXT_MUTED)),
    ]);
    f.render_widget(Paragraph::new(footer), area);
}
