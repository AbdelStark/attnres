//! Interactive TUI demo of Attention Residuals.
//!
//! The demo trains a tiny model in real time and exposes the AttnRes mechanics:
//! - Live training telemetry and loss convergence
//! - Per-sublayer depth routing heatmaps
//! - Block boundaries and active block flow
//! - Step-by-step AttnRes pipeline inspection
//! - Two-phase inference parity and merge behavior
//!
//! Controls:
//!   Space        Start / Pause / Resume
//!   Up / Down    Adjust training speed
//!   Left / Right Inspect previous / next sublayer
//!   Tab          Cycle views
//!   1 / 2 / 3    Overview / Pipeline / Inference
//!   ?            Toggle help
//!   r            Reset model
//!   q / Esc      Quit
//!
//! Run with: `cargo run --example demo_tui --release`

use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Chart, Clear, Dataset, Gauge, Paragraph, Tabs, Wrap,
};
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

const BG: Color = Color::Rgb(7, 12, 20);
const PANEL_BG: Color = Color::Rgb(12, 18, 31);
const PANEL_ALT: Color = Color::Rgb(16, 25, 41);
const BORDER: Color = Color::Rgb(58, 76, 106);
const TEXT: Color = Color::Rgb(231, 238, 251);
const TEXT_DIM: Color = Color::Rgb(170, 184, 210);
const TEXT_MUTED: Color = Color::Rgb(110, 127, 158);
const CYAN: Color = Color::Rgb(86, 196, 255);
const BLUE: Color = Color::Rgb(84, 140, 255);
const MINT: Color = Color::Rgb(94, 234, 173);
const AMBER: Color = Color::Rgb(255, 196, 88);
const CORAL: Color = Color::Rgb(255, 124, 108);
const MAGENTA: Color = Color::Rgb(223, 126, 255);
const SLATE: Color = Color::Rgb(42, 55, 82);

const LOSS_WINDOW: usize = 240;
const EVENT_CAPACITY: usize = 8;
const MIN_WIDTH: u16 = 80;
const MIN_HEIGHT: u16 = 24;
const COMFORT_WIDTH: u16 = 100;
const COMFORT_HEIGHT: u16 = 32;

const ALGO_STEPS: [(&str, &str); 5] = [
    (
        "Stack block representations",
        "Collect completed blocks plus the current partial into the depth stack V.",
    ),
    (
        "RMS-normalize the sources",
        "Normalize each source before scoring so magnitude alone cannot dominate routing.",
    ),
    (
        "Score depth with w_l",
        "Project normalized sources onto the learned pseudo-query to get depth logits.",
    ),
    (
        "Softmax over depth",
        "Turn logits into routing mass over the depth axis, not the token axis.",
    ),
    (
        "Route and accumulate",
        "Blend sources with alpha, run the sublayer, and fold the result into the active block.",
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ViewMode {
    Overview,
    Pipeline,
    Inference,
}

impl ViewMode {
    fn title(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Pipeline => "Pipeline",
            Self::Inference => "Inference",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Overview => 0,
            Self::Pipeline => 1,
            Self::Inference => 2,
        }
    }

    fn from_index(index: usize) -> Self {
        match index % 3 {
            0 => Self::Overview,
            1 => Self::Pipeline,
            _ => Self::Inference,
        }
    }

    fn next(self) -> Self {
        Self::from_index(self.index() + 1)
    }
}

#[derive(Clone, Copy, Debug)]
enum SublayerKind {
    Attn,
    Mlp,
}

impl SublayerKind {
    fn short(self) -> &'static str {
        match self {
            Self::Attn => "Attn",
            Self::Mlp => "MLP",
        }
    }

    fn code(self) -> &'static str {
        match self {
            Self::Attn => "A",
            Self::Mlp => "M",
        }
    }

    fn accent(self) -> Color {
        match self {
            Self::Attn => CYAN,
            Self::Mlp => MAGENTA,
        }
    }
}

#[derive(Clone)]
struct AlphaDetails {
    logits: Vec<f32>,
    weights: Vec<f32>,
}

#[derive(Clone)]
struct SublayerSnapshot {
    index: usize,
    layer_idx: usize,
    kind: SublayerKind,
    target_block: usize,
    slot_in_block: usize,
    boundary_before: bool,
    has_partial: bool,
    source_labels: Vec<String>,
    logits: Vec<f32>,
    weights: Vec<f32>,
    query_norm: f32,
    attn_res_rms: f32,
    partial_rms_before: f32,
    partial_rms_after: f32,
    sublayer_out_rms: f32,
    selectivity: f32,
    entropy: f32,
    dominant_source_idx: usize,
    dominant_source_label: String,
    dominant_weight: f32,
    inter_mass: f32,
    intra_mass: f32,
}

impl SublayerSnapshot {
    fn label(&self) -> String {
        format!("L{} {}", self.layer_idx, self.kind.short())
    }

    fn chip(&self) -> String {
        format!("{}{}", self.kind.code(), self.layer_idx)
    }

    fn source_count(&self) -> usize {
        self.weights.len()
    }

    fn route_mode(&self) -> &'static str {
        if !self.has_partial {
            "inter-only"
        } else if self.intra_mass >= self.inter_mass {
            "partial-dominant"
        } else {
            "cache-dominant"
        }
    }
}

#[derive(Clone)]
struct BlockSummary {
    block_idx: usize,
    avg_selectivity: f32,
    avg_query_norm: f32,
    sublayers: usize,
}

#[derive(Clone)]
struct Diagnostics {
    sublayers: Vec<SublayerSnapshot>,
    block_summaries: Vec<BlockSummary>,
    avg_selectivity: f32,
    avg_query_norm: f32,
    max_query_norm: f32,
    max_sources: usize,
    active_block: usize,
    current_block_fill: usize,
    completed_residual_blocks: usize,
    partial_rms: f32,
    hidden_rms: f32,
    two_phase_diff: f32,
}

impl Diagnostics {
    fn empty(num_sublayers: usize) -> Self {
        Self {
            sublayers: Vec::with_capacity(num_sublayers),
            block_summaries: Vec::new(),
            avg_selectivity: 0.0,
            avg_query_norm: 0.0,
            max_query_norm: 0.0,
            max_sources: 1,
            active_block: 1,
            current_block_fill: 0,
            completed_residual_blocks: 0,
            partial_rms: 0.0,
            hidden_rms: 0.0,
            two_phase_diff: 0.0,
        }
    }
}

struct App {
    d_model: usize,
    num_layers: usize,
    num_transformer_layers: usize,
    num_blocks: usize,
    block_size: usize,
    num_heads: usize,
    vocab_size: usize,
    d_ff: usize,

    step: usize,
    max_steps: usize,
    loss_history: Vec<f64>,
    last_loss: Option<f64>,
    loss_delta: Option<f64>,
    loss_ema: Option<f64>,

    paused: bool,
    speed: usize,
    completed: bool,
    show_help: bool,
    view: ViewMode,
    selected_sublayer: usize,
    algo_phase: usize,

    last_train_ms: f64,
    last_diag_ms: f64,
    avg_loop_ms: f64,

    diagnostics: Diagnostics,
    events: VecDeque<String>,
}

impl App {
    fn new(config: &AttnResConfig, diagnostics: Diagnostics) -> Self {
        Self {
            d_model: config.d_model,
            num_layers: config.num_layers,
            num_transformer_layers: config.num_transformer_layers(),
            num_blocks: config.num_blocks,
            block_size: config.block_size(),
            num_heads: config.num_heads,
            vocab_size: config.vocab_size,
            d_ff: config.effective_d_ff(),
            step: 0,
            max_steps: 320,
            loss_history: Vec::with_capacity(LOSS_WINDOW),
            last_loss: None,
            loss_delta: None,
            loss_ema: None,
            paused: true,
            speed: 3,
            completed: false,
            show_help: false,
            view: ViewMode::Overview,
            selected_sublayer: 0,
            algo_phase: 0,
            last_train_ms: 0.0,
            last_diag_ms: 0.0,
            avg_loop_ms: 0.0,
            diagnostics,
            events: VecDeque::with_capacity(EVENT_CAPACITY),
        }
    }

    fn reset(&mut self, diagnostics: Diagnostics) {
        self.step = 0;
        self.loss_history.clear();
        self.last_loss = None;
        self.loss_delta = None;
        self.loss_ema = None;
        self.paused = true;
        self.completed = false;
        self.algo_phase = 0;
        self.last_train_ms = 0.0;
        self.last_diag_ms = 0.0;
        self.avg_loop_ms = 0.0;
        self.diagnostics = diagnostics;
        self.selected_sublayer = self
            .selected_sublayer
            .min(self.diagnostics.sublayers.len().saturating_sub(1));
        self.events.clear();
        self.bootstrap_events();
    }

    fn bootstrap_events(&mut self) {
        self.push_event("dashboard armed; press Space to start training");
        self.push_event("inspect sublayers with Left/Right and switch views with Tab or 1/2/3");
        self.push_event(format!(
            "block size {} over {} residual blocks",
            self.block_size, self.num_blocks
        ));
    }

    fn push_event(&mut self, message: impl Into<String>) {
        if self.events.len() == EVENT_CAPACITY {
            self.events.pop_back();
        }
        self.events
            .push_front(format!("s{:03}  {}", self.step, message.into()));
    }

    fn selected_snapshot(&self) -> &SublayerSnapshot {
        &self.diagnostics.sublayers[self.selected_sublayer]
    }

    fn progress_ratio(&self) -> f64 {
        self.step as f64 / self.max_steps.max(1) as f64
    }

    fn steps_per_second(&self) -> f64 {
        if self.avg_loop_ms <= 0.0 {
            0.0
        } else {
            1000.0 / self.avg_loop_ms
        }
    }

    fn eta_seconds(&self) -> f64 {
        let remaining = self.max_steps.saturating_sub(self.step) as f64;
        let sps = self.steps_per_second();
        if sps <= 0.0 {
            0.0
        } else {
            remaining / sps
        }
    }

    fn set_view(&mut self, view: ViewMode) {
        if self.view != view {
            self.view = view;
            self.push_event(format!("view -> {}", self.view.title()));
        }
    }

    fn move_selection(&mut self, delta: isize) {
        let len = self.diagnostics.sublayers.len();
        if len == 0 {
            return;
        }
        let next = (self.selected_sublayer as isize + delta).rem_euclid(len as isize) as usize;
        self.selected_sublayer = next;
    }

    fn set_speed(&mut self, next: usize) {
        let next = next.clamp(1, 5);
        if self.speed != next {
            self.speed = next;
            self.push_event(format!("speed -> {}x", self.speed));
        }
    }

    fn record_step(&mut self, loss: f32, diagnostics: Diagnostics, train_ms: f64, diag_ms: f64) {
        let previous_band = routing_band(self.diagnostics.avg_selectivity);
        let previous_dominant = self
            .diagnostics
            .sublayers
            .get(self.selected_sublayer)
            .map(|snapshot| snapshot.dominant_source_label.clone());

        let loss = f64::from(loss);
        self.loss_delta = self.last_loss.map(|prev| loss - prev);
        self.loss_ema = Some(match self.loss_ema {
            Some(ema) => ema * 0.85 + loss * 0.15,
            None => loss,
        });
        self.last_loss = Some(loss);
        self.loss_history.push(loss);
        if self.loss_history.len() > LOSS_WINDOW {
            self.loss_history.remove(0);
        }

        self.last_train_ms = train_ms;
        self.last_diag_ms = diag_ms;
        let loop_ms = train_ms + diag_ms;
        self.avg_loop_ms = if self.avg_loop_ms <= 0.0 {
            loop_ms
        } else {
            self.avg_loop_ms * 0.82 + loop_ms * 0.18
        };

        self.diagnostics = diagnostics;
        self.selected_sublayer = self
            .selected_sublayer
            .min(self.diagnostics.sublayers.len().saturating_sub(1));

        self.step += 1;
        self.algo_phase = (self.step / 4) % ALGO_STEPS.len();

        let new_band = routing_band(self.diagnostics.avg_selectivity);
        if self.step == 1 {
            self.push_event("loss stream live; zero-init routing starts near-uniform");
        }
        if previous_band != new_band {
            self.push_event(format!("routing regime -> {}", band_description(new_band)));
        }
        if let Some(previous_dominant) = previous_dominant {
            let selected = self.selected_snapshot();
            if previous_dominant != selected.dominant_source_label {
                self.push_event(format!(
                    "{} now favors {} ({:.0}%)",
                    selected.label(),
                    selected.dominant_source_label,
                    selected.dominant_weight * 100.0
                ));
            }
        }
        if self.step % 25 == 0 {
            self.push_event(format!(
                "two-phase parity max diff {:.2e}",
                self.diagnostics.two_phase_diff
            ));
        }

        if self.step >= self.max_steps {
            self.completed = true;
            self.paused = true;
            self.push_event("run complete; press Space to restart or r to reset");
        }
    }
}

fn band_description(band: &'static str) -> &'static str {
    match band {
        "uniform" => "uniform averaging",
        "forming" => "emerging preferences",
        _ => "selective routing",
    }
}

fn routing_band(selectivity: f32) -> &'static str {
    if selectivity < 0.18 {
        "uniform"
    } else if selectivity < 0.36 {
        "forming"
    } else {
        "selective"
    }
}

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

    if let Err(ref error) = result {
        eprintln!("Error: {error}");
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

    let train_mask = causal_mask::<AB>(2, 16, &device);
    let probe_tokens = Tensor::<AB, 2, Int>::zeros([1, 8], &device);
    let probe_mask = causal_mask::<AB>(1, 8, &device);

    let mut model: AttnResTransformer<AB> = config.init_model(&device);
    let mut optim = AdamConfig::new().init();

    let initial_diagnostics = extract_diagnostics(&model, &config, &probe_tokens, &probe_mask);
    let mut app = App::new(&config, initial_diagnostics);
    app.bootstrap_events();

    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|frame| ui(frame, &app))?;

        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Esc if app.show_help => app.show_help = false,
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Char('?') => app.show_help = !app.show_help,
                        KeyCode::Tab => app.set_view(app.view.next()),
                        KeyCode::Char('1') => app.set_view(ViewMode::Overview),
                        KeyCode::Char('2') => app.set_view(ViewMode::Pipeline),
                        KeyCode::Char('3') => app.set_view(ViewMode::Inference),
                        KeyCode::Left | KeyCode::Char('h') => app.move_selection(-1),
                        KeyCode::Right | KeyCode::Char('l') => app.move_selection(1),
                        KeyCode::Up => app.set_speed(app.speed + 1),
                        KeyCode::Down => app.set_speed(app.speed.saturating_sub(1)),
                        KeyCode::Char('r') => {
                            model = config.init_model(&device);
                            optim = AdamConfig::new().init();
                            let diagnostics =
                                extract_diagnostics(&model, &config, &probe_tokens, &probe_mask);
                            app.reset(diagnostics);
                            last_tick = Instant::now();
                        }
                        KeyCode::Char(' ') => {
                            if app.completed {
                                model = config.init_model(&device);
                                optim = AdamConfig::new().init();
                                let diagnostics = extract_diagnostics(
                                    &model,
                                    &config,
                                    &probe_tokens,
                                    &probe_mask,
                                );
                                app.reset(diagnostics);
                                app.paused = false;
                                app.push_event("new training run started");
                                last_tick = Instant::now();
                            } else {
                                app.paused = !app.paused;
                                if app.paused {
                                    app.push_event("training paused");
                                } else {
                                    app.push_event("training resumed");
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if !app.paused && app.step < app.max_steps {
            let tick_ms: u64 = match app.speed {
                1 => 260,
                2 => 150,
                3 => 85,
                4 => 45,
                _ => 20,
            };

            if last_tick.elapsed() >= Duration::from_millis(tick_ms) {
                let train_started = Instant::now();
                let (new_model, loss) =
                    train_step(model, &mut optim, &train_mask, &config, &device);
                let train_ms = train_started.elapsed().as_secs_f64() * 1000.0;
                model = new_model;

                let diagnostics_started = Instant::now();
                let diagnostics = extract_diagnostics(&model, &config, &probe_tokens, &probe_mask);
                let diag_ms = diagnostics_started.elapsed().as_secs_f64() * 1000.0;

                app.record_step(loss, diagnostics, train_ms, diag_ms);
                last_tick = Instant::now();
            }
        }
    }
}

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
    let [batch, tokens, vocab] = logits.dims();

    let loss_fn = CrossEntropyLossConfig::new().with_logits(true).init(device);
    let loss = loss_fn
        .forward(
            logits.reshape([batch * tokens, vocab]),
            targets.reshape([batch * tokens]),
        )
        .mean();

    let loss_val: f32 = loss.clone().into_scalar();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optim.step(0.001, model, grads);

    (model, loss_val)
}

fn extract_diagnostics(
    model: &AttnResTransformer<AB>,
    config: &AttnResConfig,
    probe_tokens: &Tensor<AB, 2, Int>,
    probe_mask: &Tensor<AB, 3>,
) -> Diagnostics {
    let mut diagnostics = Diagnostics::empty(config.num_layers);
    let hidden = model.embed_tokens(probe_tokens.clone());
    let mut state = BlockState::new(hidden);
    let block_size = config.block_size();

    for layer in model.layers() {
        let layer_idx = layer.layer_idx();
        let attn_idx = layer_idx * 2;
        let mlp_idx = attn_idx + 1;
        let (attn_res, mlp_res) = layer.attn_res_ops();

        let current_partial = state.partial_block.take();
        let attn_details = compute_alpha_details(attn_res, &state.blocks, current_partial.as_ref());
        let attn_h = attn_res.forward_optional_partial(&state.blocks, current_partial.as_ref());
        let has_partial = current_partial.is_some();
        let partial_rms_before = current_partial.as_ref().map(tensor_rms::<3>).unwrap_or(0.0);

        let boundary_before_attn = attn_idx > 0 && attn_idx.is_multiple_of(block_size);
        let target_block_attn = if boundary_before_attn {
            state.blocks.len() + 1
        } else {
            state.blocks.len()
        };

        let mut partial_for_attn =
            current_partial.unwrap_or_else(|| Tensor::zeros_like(state.blocks.last().unwrap()));
        if boundary_before_attn {
            state.blocks.push(partial_for_attn.clone());
            partial_for_attn = Tensor::zeros_like(state.blocks.last().unwrap());
        }

        let attn_out = layer.forward_attn_sublayer(attn_h.clone(), Some(probe_mask));
        let partial_after_attn = partial_for_attn + attn_out.clone();

        diagnostics.sublayers.push(build_sublayer_snapshot(
            attn_idx,
            layer_idx,
            SublayerKind::Attn,
            target_block_attn,
            attn_idx % block_size,
            boundary_before_attn,
            has_partial,
            attn_details,
            attn_res,
            partial_rms_before,
            &attn_h,
            &attn_out,
            &partial_after_attn,
        ));

        let mlp_details = compute_alpha_details(mlp_res, &state.blocks, Some(&partial_after_attn));
        let mlp_h = mlp_res.forward_optional_partial(&state.blocks, Some(&partial_after_attn));
        let partial_rms_before = tensor_rms::<3>(&partial_after_attn);

        let boundary_before_mlp = mlp_idx > 0 && mlp_idx.is_multiple_of(block_size);
        let target_block_mlp = if boundary_before_mlp {
            state.blocks.len() + 1
        } else {
            state.blocks.len()
        };

        let mut partial_for_mlp = partial_after_attn.clone();
        if boundary_before_mlp {
            state.blocks.push(partial_for_mlp.clone());
            partial_for_mlp = Tensor::zeros_like(state.blocks.last().unwrap());
        }

        let mlp_out = layer.forward_mlp_sublayer(mlp_h.clone());
        let partial_after_mlp = partial_for_mlp + mlp_out.clone();

        diagnostics.sublayers.push(build_sublayer_snapshot(
            mlp_idx,
            layer_idx,
            SublayerKind::Mlp,
            target_block_mlp,
            mlp_idx % block_size,
            boundary_before_mlp,
            true,
            mlp_details,
            mlp_res,
            partial_rms_before,
            &mlp_h,
            &mlp_out,
            &partial_after_mlp,
        ));

        state.partial_block = Some(partial_after_mlp);
    }

    diagnostics.max_sources = diagnostics
        .sublayers
        .iter()
        .map(SublayerSnapshot::source_count)
        .max()
        .unwrap_or(1);

    if !diagnostics.sublayers.is_empty() {
        diagnostics.avg_selectivity = diagnostics
            .sublayers
            .iter()
            .map(|snapshot| snapshot.selectivity)
            .sum::<f32>()
            / diagnostics.sublayers.len() as f32;
        diagnostics.avg_query_norm = diagnostics
            .sublayers
            .iter()
            .map(|snapshot| snapshot.query_norm)
            .sum::<f32>()
            / diagnostics.sublayers.len() as f32;
        diagnostics.max_query_norm = diagnostics
            .sublayers
            .iter()
            .map(|snapshot| snapshot.query_norm)
            .fold(0.0f32, f32::max);
    }

    diagnostics.active_block = state.blocks.len();
    diagnostics.current_block_fill = diagnostics
        .sublayers
        .last()
        .map(|snapshot| snapshot.slot_in_block + 1)
        .unwrap_or(0);
    diagnostics.completed_residual_blocks = state.blocks.len().saturating_sub(1);
    diagnostics.partial_rms = state
        .partial_block
        .as_ref()
        .map(tensor_rms::<3>)
        .unwrap_or(0.0);

    diagnostics.block_summaries = build_block_summaries(&diagnostics.sublayers, config.num_blocks);

    let hidden = model.forward_hidden(probe_tokens.clone(), Some(probe_mask));
    diagnostics.hidden_rms = tensor_rms::<3>(&hidden);

    let standard = model.forward(probe_tokens.clone(), Some(probe_mask));
    let two_phase = model.forward_two_phase(probe_tokens.clone(), Some(probe_mask));
    diagnostics.two_phase_diff = tensor_max_abs::<3>(&(standard - two_phase));

    diagnostics
}

fn build_block_summaries(snapshots: &[SublayerSnapshot], num_blocks: usize) -> Vec<BlockSummary> {
    let mut summaries = Vec::with_capacity(num_blocks);
    for block_idx in 1..=num_blocks {
        let rows: Vec<&SublayerSnapshot> = snapshots
            .iter()
            .filter(|snapshot| snapshot.target_block == block_idx)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_selectivity = rows
            .iter()
            .map(|snapshot| snapshot.selectivity)
            .sum::<f32>()
            / rows.len() as f32;
        let avg_query_norm =
            rows.iter().map(|snapshot| snapshot.query_norm).sum::<f32>() / rows.len() as f32;
        summaries.push(BlockSummary {
            block_idx,
            avg_selectivity,
            avg_query_norm,
            sublayers: rows.len(),
        });
    }
    summaries
}

fn compute_alpha_details(
    op: &AttnResOp<AB>,
    blocks: &[Tensor<AB, 3>],
    partial: Option<&Tensor<AB, 3>>,
) -> AlphaDetails {
    let mut sources: Vec<Tensor<AB, 3>> = blocks.to_vec();
    if let Some(partial) = partial {
        sources.push(partial.clone());
    }

    let count = sources.len();
    let values = Tensor::stack(sources, 0);
    let keys = op.norm.forward_4d(values);
    let w = op
        .pseudo_query
        .val()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0);
    let logits = (keys * w).sum_dim(3).squeeze_dim::<3>(3);
    let alpha = softmax(logits.clone(), 0);

    AlphaDetails {
        logits: average_depth_metric(logits, count),
        weights: average_depth_metric(alpha, count),
    }
}

fn average_depth_metric(metric: Tensor<AB, 3>, count: usize) -> Vec<f32> {
    metric
        .mean_dim(2)
        .squeeze_dim::<2>(2)
        .mean_dim(1)
        .squeeze_dim::<1>(1)
        .reshape([count])
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

fn build_sublayer_snapshot(
    index: usize,
    layer_idx: usize,
    kind: SublayerKind,
    target_block: usize,
    slot_in_block: usize,
    boundary_before: bool,
    has_partial: bool,
    details: AlphaDetails,
    op: &AttnResOp<AB>,
    partial_rms_before: f32,
    attn_res_output: &Tensor<AB, 3>,
    sublayer_output: &Tensor<AB, 3>,
    partial_rms_after: &Tensor<AB, 3>,
) -> SublayerSnapshot {
    let source_labels = source_labels(details.weights.len(), has_partial);
    let dominant_source_idx = argmax(&details.weights);
    let dominant_weight = details.weights[dominant_source_idx];
    let dominant_source_label = source_labels[dominant_source_idx].clone();
    let entropy = normalized_entropy(&details.weights);

    let (inter_mass, intra_mass) = if has_partial {
        let intra = *details.weights.last().unwrap_or(&0.0);
        (1.0 - intra, intra)
    } else {
        (1.0, 0.0)
    };

    SublayerSnapshot {
        index,
        layer_idx,
        kind,
        target_block,
        slot_in_block,
        boundary_before,
        has_partial,
        source_labels,
        logits: details.logits,
        weights: details.weights,
        query_norm: tensor_rms::<1>(&op.pseudo_query.val()),
        attn_res_rms: tensor_rms::<3>(attn_res_output),
        partial_rms_before,
        partial_rms_after: tensor_rms::<3>(partial_rms_after),
        sublayer_out_rms: tensor_rms::<3>(sublayer_output),
        selectivity: 1.0 - entropy,
        entropy,
        dominant_source_idx,
        dominant_source_label,
        dominant_weight,
        inter_mass,
        intra_mass,
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn normalized_entropy(weights: &[f32]) -> f32 {
    if weights.len() <= 1 {
        return 0.0;
    }
    let entropy = weights
        .iter()
        .copied()
        .filter(|weight| *weight > 0.0)
        .map(|weight| -weight * weight.ln())
        .sum::<f32>();
    let max_entropy = (weights.len() as f32).ln();
    if max_entropy <= 0.0 {
        0.0
    } else {
        (entropy / max_entropy).clamp(0.0, 1.0)
    }
}

fn source_labels(count: usize, has_partial: bool) -> Vec<String> {
    let block_count = if has_partial {
        count.saturating_sub(1)
    } else {
        count
    };
    let mut labels = (0..block_count).map(block_label).collect::<Vec<_>>();
    if has_partial {
        labels.push("Part".to_string());
    }
    labels
}

fn block_label(index: usize) -> String {
    if index == 0 {
        "Emb".to_string()
    } else {
        format!("B{index}")
    }
}

fn tensor_rms<const D: usize>(tensor: &Tensor<AB, D>) -> f32 {
    tensor.clone().powf_scalar(2.0).mean().sqrt().into_scalar()
}

fn tensor_max_abs<const D: usize>(tensor: &Tensor<AB, D>) -> f32 {
    tensor.clone().abs().max().into_scalar()
}

fn ui(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(Block::new().style(Style::default().bg(BG).fg(TEXT)), area);

    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        draw_min_size(frame, area, app);
        if app.show_help {
            draw_help_overlay(frame, area);
        }
        return;
    }

    if area.width < COMFORT_WIDTH || area.height < COMFORT_HEIGHT {
        draw_small_ui(frame, area, app);
        if app.show_help {
            draw_help_overlay(frame, area);
        }
        return;
    }

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(20),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, layout[0], app);

    let compact = area.width < 140 || area.height < 44;
    match app.view {
        ViewMode::Overview => draw_overview(frame, layout[1], app, compact),
        ViewMode::Pipeline => draw_pipeline(frame, layout[1], app, compact),
        ViewMode::Inference => draw_inference(frame, layout[1], app, compact),
    }

    draw_footer(frame, layout[2], app);

    if app.show_help {
        draw_help_overlay(frame, area);
    }
}

fn draw_small_ui(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(6),
            Constraint::Length(1),
        ])
        .split(area);

    let selected = app.selected_snapshot();
    let header = vec![
        Line::from(vec![
            pill("α", BG, CYAN),
            Span::raw(" "),
            Span::styled("AttnRes TUI", Style::default().fg(TEXT).bold()),
            Span::styled(
                format!("  {}  •  {}", app.view.title(), selected.label()),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("status  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                if app.completed {
                    "complete"
                } else if app.paused {
                    "paused"
                } else {
                    "training"
                },
                Style::default()
                    .fg(if app.completed {
                        MINT
                    } else if app.paused {
                        AMBER
                    } else {
                        CYAN
                    })
                    .bold(),
            ),
            Span::styled("  •  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "step {}/{}  •  speed {}x",
                    app.step, app.max_steps, app.speed
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("routing ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                band_description(routing_band(app.diagnostics.avg_selectivity)),
                Style::default().fg(TEXT_DIM),
            ),
            Span::styled("  •  parity ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.2e}", app.diagnostics.two_phase_diff),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(header), layout[0]);

    draw_small_metrics(frame, layout[1], app);
    draw_small_selected(frame, layout[2], app);
    draw_small_events(frame, layout[3], app);
    draw_footer(frame, layout[4], app);
}

fn draw_small_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let loss = app
        .last_loss
        .map(|loss| format!("{loss:.4}"))
        .unwrap_or_else(|| "—".to_string());
    let selected = app.selected_snapshot();
    let lines = vec![
        Line::from(vec![
            Span::styled("loss      ", Style::default().fg(TEXT_MUTED)),
            Span::styled(loss, Style::default().fg(TEXT)),
            Span::styled("  •  ema ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                app.loss_ema
                    .map(|loss| format!("{loss:.4}"))
                    .unwrap_or_else(|| "—".to_string()),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("selective ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.0}%", app.diagnostics.avg_selectivity * 100.0),
                Style::default().fg(TEXT_DIM),
            ),
            Span::styled("  •  ||w|| ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3} avg", app.diagnostics.avg_query_norm),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("blocks    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "active B{} slot {}/{}",
                    app.diagnostics.active_block,
                    app.diagnostics.current_block_fill,
                    app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("selected  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} -> {} {:.0}%",
                    selected.label(),
                    selected.dominant_source_label,
                    selected.dominant_weight * 100.0
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(Span::styled(
            format!(
                "spark     {}",
                sparkline(&app.loss_history, area.width.saturating_sub(14) as usize)
            ),
            Style::default().fg(BLUE),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Snapshot", CYAN).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_small_selected(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let width = area.width.saturating_sub(20) as usize;
    let lines: Vec<Line> = selected
        .source_labels
        .iter()
        .zip(selected.weights.iter())
        .take(4)
        .map(|(label, weight)| {
            let filled = ((width as f32) * *weight).round() as usize;
            Line::from(vec![
                Span::styled(format!("{label:<6} "), Style::default().fg(TEXT_MUTED)),
                Span::styled(
                    "█".repeat(filled.min(width)),
                    Style::default().fg(heat_color(*weight)),
                ),
                Span::styled(
                    format!(" {:>4.0}%", weight * 100.0),
                    Style::default().fg(TEXT_DIM),
                ),
            ])
        })
        .collect();

    let mut all_lines = vec![
        Line::from(vec![
            Span::styled("mode   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(selected.route_mode(), Style::default().fg(TEXT_DIM)),
            Span::styled("  •  block ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "B{} slot {}/{}",
                    selected.target_block,
                    selected.slot_in_block + 1,
                    app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("write  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "partial {:.3} -> {:.3}",
                    selected.partial_rms_before, selected.partial_rms_after
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    all_lines.extend(lines);

    frame.render_widget(
        Paragraph::new(all_lines)
            .block(
                panel("Selected Route", selected.kind.accent())
                    .style(Style::default().bg(PANEL_ALT)),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_small_events(frame: &mut Frame, area: Rect, app: &App) {
    let events: Vec<Line> = app
        .events
        .iter()
        .take(4)
        .map(|event| {
            Line::from(vec![
                Span::styled("• ", Style::default().fg(CYAN)),
                Span::styled(event.clone(), Style::default().fg(TEXT_DIM)),
            ])
        })
        .collect();

    frame.render_widget(
        Paragraph::new(events)
            .block(panel("Feed", MINT).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_min_size(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel("Need a Bigger Terminal", CORAL);
    let lines = vec![
        Line::from(Span::styled(
            format!(
                "Current size: {}x{}  •  Minimum: {}x{}  •  Comfortable: {}x{}",
                area.width, area.height, MIN_WIDTH, MIN_HEIGHT, COMFORT_WIDTH, COMFORT_HEIGHT
            ),
            Style::default().fg(TEXT),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "This dashboard is responsive, but the visual routing views need more room.",
            Style::default().fg(TEXT_DIM),
        )),
        Line::from(Span::styled(
            format!(
                "View: {}  •  Step: {} / {}  •  Selected: {}",
                app.view.title(),
                app.step,
                app.max_steps,
                app.selected_snapshot().label()
            ),
            Style::default().fg(TEXT_DIM),
        )),
        Line::from(""),
        Line::from(hint_line()),
    ];

    frame.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        centered_rect(area, 72, 40),
    );
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(2),
        ])
        .split(area);

    let header = Line::from(vec![
        pill("α depth", BG, CYAN),
        Span::raw(" "),
        Span::styled("AttnRes TUI Lab", Style::default().fg(TEXT).bold()),
        Span::styled(
            "  terminal observability for depth routing",
            Style::default().fg(TEXT_DIM),
        ),
    ]);
    frame.render_widget(Paragraph::new(header), rows[0]);

    let status = if app.completed {
        pill("complete", BG, MINT)
    } else if app.paused {
        pill("paused", BG, AMBER)
    } else {
        pill("training", BG, BLUE)
    };

    let selected = app.selected_snapshot();
    let badges = Line::from(vec![
        status,
        Span::raw(" "),
        pill(
            format!("step {:03}/{:03}", app.step, app.max_steps),
            TEXT,
            SLATE,
        ),
        Span::raw(" "),
        pill(format!("view {}", app.view.title()), TEXT, SLATE),
        Span::raw(" "),
        pill(
            format!("inspect {}", selected.label()),
            TEXT,
            selected.kind.accent(),
        ),
    ]);
    frame.render_widget(Paragraph::new(badges), rows[1]);

    let row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(rows[2]);

    let tabs = Tabs::new(vec!["Overview", "Pipeline", "Inference"])
        .select(app.view.index())
        .divider(Span::styled("  ", Style::default()))
        .style(Style::default().fg(TEXT_MUTED))
        .highlight_style(
            Style::default()
                .fg(BG)
                .bg(CYAN)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(tabs, row[0]);

    let narrative = Line::from(vec![
        Span::styled("Routing  ", Style::default().fg(TEXT_MUTED)),
        Span::styled(
            band_description(routing_band(app.diagnostics.avg_selectivity)),
            Style::default().fg(TEXT),
        ),
        Span::styled("  •  ", Style::default().fg(TEXT_MUTED)),
        Span::styled(
            format!(
                "{} leads {} at {:.0}%",
                selected.label(),
                selected.dominant_source_label,
                selected.dominant_weight * 100.0
            ),
            Style::default().fg(TEXT_DIM),
        ),
    ]);
    frame.render_widget(Paragraph::new(narrative), row[1]);
}

fn draw_overview(frame: &mut Frame, area: Rect, app: &App, compact: bool) {
    if compact {
        draw_overview_compact(frame, area, app);
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(13),
            Constraint::Min(16),
        ])
        .split(area);

    draw_overview_metrics(frame, rows[0], app);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(rows[1]);
    draw_loss_chart(frame, middle[0], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(6)])
        .split(middle[1]);
    draw_architecture_card(frame, right[0], app);
    draw_runtime_card(frame, right[1], app);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(rows[2]);
    draw_attention_heatmap(frame, bottom[0], app, false);

    let side = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(7)])
        .split(bottom[1]);
    draw_selected_detail(frame, side[0], app);
    draw_event_feed(frame, side[1], app);
}

fn draw_overview_compact(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Min(14),
        ])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Length(6)])
        .split(rows[0]);
    draw_overview_metrics(frame, top[0], app);
    draw_architecture_card(frame, top[1], app);
    draw_loss_chart(frame, rows[1], app);

    let bottom = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(10),
            Constraint::Min(6),
        ])
        .split(rows[2]);
    draw_attention_heatmap(frame, bottom[0], app, true);
    draw_selected_detail(frame, bottom[1], app);
    draw_event_feed(frame, bottom[2], app);
}

fn draw_pipeline(frame: &mut Frame, area: Rect, app: &App, compact: bool) {
    if compact {
        draw_pipeline_compact(frame, area, app);
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Min(12),
        ])
        .split(area);

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
        .split(rows[0]);
    draw_algorithm_card(frame, top[0], app);
    draw_pipeline_identity(frame, top[1], app);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);
    draw_metric_bars(frame, middle[0], app, true);
    draw_metric_bars(frame, middle[1], app, false);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
        .split(rows[2]);
    draw_route_story(frame, bottom[0], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(6)])
        .split(bottom[1]);
    draw_block_schedule(frame, right[0], app);
    draw_event_feed(frame, right[1], app);
}

fn draw_pipeline_compact(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Length(11),
            Constraint::Length(11),
            Constraint::Min(10),
        ])
        .split(area);

    draw_algorithm_card(frame, rows[0], app);
    draw_pipeline_identity(frame, rows[1], app);

    let bars = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Length(10)])
        .split(rows[2]);
    draw_metric_bars(frame, bars[0], app, true);
    draw_metric_bars(frame, bars[1], app, false);
    draw_route_story(frame, rows[3], app);
}

fn draw_inference(frame: &mut Frame, area: Rect, app: &App, compact: bool) {
    if compact {
        draw_inference_compact(frame, area, app);
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Min(12),
        ])
        .split(area);

    draw_inference_metrics(frame, rows[0], app);

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
        .split(rows[1]);
    draw_two_phase_schedule(frame, middle[0], app);
    draw_merge_card(frame, middle[1], app);

    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(rows[2]);
    draw_block_cache_card(frame, bottom[0], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(6)])
        .split(bottom[1]);
    draw_selected_detail(frame, right[0], app);
    draw_event_feed(frame, right[1], app);
}

fn draw_inference_compact(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Min(10),
        ])
        .split(area);

    draw_inference_metrics(frame, rows[0], app);
    draw_two_phase_schedule(frame, rows[1], app);
    draw_merge_card(frame, rows[2], app);
    draw_block_cache_card(frame, rows[3], app);
}

fn draw_overview_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    let status_lines = vec![
        Line::from(vec![
            Span::styled("status  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                if app.completed {
                    "Complete"
                } else if app.paused {
                    "Paused"
                } else {
                    "Training"
                },
                Style::default()
                    .fg(if app.completed {
                        MINT
                    } else if app.paused {
                        AMBER
                    } else {
                        CYAN
                    })
                    .bold(),
            ),
        ]),
        Line::from(vec![
            Span::styled("speed   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(speed_dots(app.speed), Style::default().fg(BLUE)),
            Span::styled(format!("  {}x", app.speed), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("eta     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format_duration(app.eta_seconds()),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    draw_card(
        frame,
        cards[0],
        "Training",
        CYAN,
        format!("{:03}/{}", app.step, app.max_steps),
        status_lines,
        Some(app.progress_ratio()),
    );

    let delta = app
        .loss_delta
        .map(|change| {
            if change < 0.0 {
                format!("↓ {:.4}", change.abs())
            } else {
                format!("↑ {:.4}", change)
            }
        })
        .unwrap_or_else(|| "—".to_string());
    let loss_lines = vec![
        Line::from(vec![
            Span::styled("delta   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(delta, Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("ema     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                app.loss_ema
                    .map(|loss| format!("{loss:.4}"))
                    .unwrap_or_else(|| "—".to_string()),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(Span::styled(
            format!("spark   {}", sparkline(&app.loss_history, 18)),
            Style::default().fg(BLUE),
        )),
    ];
    draw_card(
        frame,
        cards[1],
        "Loss",
        BLUE,
        app.last_loss
            .map(|loss| format!("{loss:.4}"))
            .unwrap_or_else(|| "—".to_string()),
        loss_lines,
        None,
    );

    let band = routing_band(app.diagnostics.avg_selectivity);
    let routing_lines = vec![
        Line::from(vec![
            Span::styled("band    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(band_description(band), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("||w||   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{:.3} avg / {:.3} max",
                    app.diagnostics.avg_query_norm, app.diagnostics.max_query_norm
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("top src ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                app.selected_snapshot().dominant_source_label.clone(),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    draw_card(
        frame,
        cards[2],
        "Routing",
        MINT,
        format!("{:.0}%", app.diagnostics.avg_selectivity * 100.0),
        routing_lines,
        Some(app.diagnostics.avg_selectivity as f64),
    );

    let parity_lines = vec![
        Line::from(vec![
            Span::styled("throughput  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.1} steps/s", app.steps_per_second()),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("train ms    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.1}", app.last_train_ms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("diag ms     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.1}", app.last_diag_ms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    draw_card(
        frame,
        cards[3],
        "Parity",
        CORAL,
        format!("{:.2e}", app.diagnostics.two_phase_diff),
        parity_lines,
        Some((1.0 - (app.diagnostics.two_phase_diff.min(1e-2) / 1e-2)) as f64),
    );
}

fn draw_inference_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let cards = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(area);

    let parity_lines = vec![
        Line::from(vec![
            Span::styled("probe hidden  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3}", app.diagnostics.hidden_rms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("avg loop ms   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.1}", app.avg_loop_ms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("status        ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                if app.diagnostics.two_phase_diff < 1e-4 {
                    "parity locked"
                } else {
                    "monitoring drift"
                },
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    draw_card(
        frame,
        cards[0],
        "Two-Phase Parity",
        CORAL,
        format!("{:.2e}", app.diagnostics.two_phase_diff),
        parity_lines,
        Some((1.0 - (app.diagnostics.two_phase_diff.min(1e-2) / 1e-2)) as f64),
    );

    let selected = app.selected_snapshot();
    let merge_lines = vec![
        Line::from(vec![
            Span::styled("selected   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(selected.label(), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("mode       ", Style::default().fg(TEXT_MUTED)),
            Span::styled(selected.route_mode(), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("block slot ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "B{} · {}/{}",
                    selected.target_block,
                    selected.slot_in_block + 1,
                    app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    draw_card(
        frame,
        cards[1],
        "Merge Split",
        AMBER,
        format!(
            "{:.0}% / {:.0}%",
            selected.inter_mass * 100.0,
            selected.intra_mass * 100.0
        ),
        merge_lines,
        Some(selected.intra_mass as f64),
    );

    let cache_lines = vec![
        Line::from(vec![
            Span::styled("completed  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{}", app.diagnostics.completed_residual_blocks),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("active     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "B{}  slot {}/{}",
                    app.diagnostics.active_block,
                    app.diagnostics.current_block_fill,
                    app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("partial    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3} rms", app.diagnostics.partial_rms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    draw_card(
        frame,
        cards[2],
        "Block Cache",
        MINT,
        format!("{}/{}", app.num_blocks, app.num_blocks),
        cache_lines,
        Some(app.diagnostics.completed_residual_blocks as f64 / app.num_blocks.max(1) as f64),
    );
}

fn draw_card(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    accent: Color,
    value: String,
    lines: Vec<Line<'static>>,
    gauge: Option<f64>,
) {
    let block = panel(title, accent);
    let mut constraints = vec![Constraint::Length(2)];
    constraints.extend(std::iter::repeat_n(Constraint::Length(1), lines.len()));
    if gauge.is_some() {
        constraints.push(Constraint::Length(2));
    }
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(block.inner(area));

    frame.render_widget(block.style(Style::default().bg(PANEL_BG)), area);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            value,
            Style::default().fg(TEXT).bold(),
        ))),
        inner[0],
    );
    for (offset, line) in lines.into_iter().enumerate() {
        frame.render_widget(Paragraph::new(line), inner[offset + 1]);
    }
    if let Some(ratio) = gauge {
        let gauge_index = inner.len() - 1;
        let gauge = Gauge::default()
            .ratio(ratio.clamp(0.0, 1.0))
            .use_unicode(true)
            .gauge_style(Style::default().fg(accent).bg(SLATE))
            .style(Style::default().bg(PANEL_BG))
            .label(Span::styled("", Style::default()));
        frame.render_widget(gauge, inner[gauge_index]);
    }
}

fn draw_loss_chart(frame: &mut Frame, area: Rect, app: &App) {
    let block = panel("Loss Stream", BLUE);
    if app.loss_history.is_empty() {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "Press Space to start training.",
                    Style::default().fg(TEXT_DIM).italic(),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "The chart will track raw loss in cyan and a smooth EMA in mint.",
                    Style::default().fg(TEXT_MUTED),
                )),
            ])
            .block(block)
            .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    let raw: Vec<(f64, f64)> = app
        .loss_history
        .iter()
        .enumerate()
        .map(|(index, value)| (index as f64, *value))
        .collect();

    let ema = ema_points(&app.loss_history, 0.18);

    let min_loss = app
        .loss_history
        .iter()
        .fold(f64::INFINITY, |min, value| min.min(*value))
        * 0.97;
    let max_loss = app
        .loss_history
        .iter()
        .fold(f64::NEG_INFINITY, |max, value| max.max(*value))
        * 1.03;

    let datasets = vec![
        Dataset::default()
            .name("loss")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(CYAN))
            .data(&raw),
        Dataset::default()
            .name("ema")
            .marker(symbols::Marker::HalfBlock)
            .style(Style::default().fg(MINT))
            .data(&ema),
    ];

    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(
            Axis::default()
                .title(Span::styled("step", Style::default().fg(TEXT_MUTED)))
                .style(Style::default().fg(TEXT_MUTED))
                .bounds([0.0, app.max_steps as f64])
                .labels(vec![
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
                .labels(vec![
                    Line::from(format!("{min_loss:.2}")),
                    Line::from(format!("{max_loss:.2}")),
                ]),
        );

    frame.render_widget(chart, area);
}

fn ema_points(values: &[f64], alpha: f64) -> Vec<(f64, f64)> {
    let mut ema = None;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let next = match ema {
                Some(previous) => previous * (1.0 - alpha) + value * alpha,
                None => *value,
            };
            ema = Some(next);
            (index as f64, next)
        })
        .collect()
}

fn draw_architecture_card(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let mut lines = vec![
        Line::from(vec![
            Span::styled("model     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "d={}  heads={}  d_ff={}  vocab={}",
                    app.d_model, app.num_heads, app.d_ff, app.vocab_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("layers    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} sublayers  •  {} transformer layers",
                    app.num_layers, app.num_transformer_layers
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("blocks    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} blocks  •  {} sublayers per block",
                    app.num_blocks, app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("cursor    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{}  ->  B{} slot {}/{}",
                    selected.label(),
                    selected.target_block,
                    selected.slot_in_block + 1,
                    app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(""),
    ];
    lines.extend(block_rail_lines(app, area.width.saturating_sub(4) as usize));

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Block Topology", MINT).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn block_rail_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let selected = app.selected_snapshot().index;
    let mut current = Vec::new();
    let mut line_width = 0usize;

    for summary in &app.diagnostics.block_summaries {
        let mut spans = vec![
            Span::styled(
                format!(" B{} ", summary.block_idx),
                Style::default()
                    .fg(BG)
                    .bg(heat_color(summary.avg_selectivity)),
            ),
            Span::raw(" "),
        ];

        for snapshot in app
            .diagnostics
            .sublayers
            .iter()
            .filter(|snapshot| snapshot.target_block == summary.block_idx)
        {
            let style = if snapshot.index == selected {
                Style::default()
                    .fg(BG)
                    .bg(snapshot.kind.accent())
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(snapshot.kind.accent())
            };
            spans.push(Span::styled(format!("{} ", snapshot.chip()), style));
        }

        let item_width: usize = spans.iter().map(|span| span.content.len()).sum();
        if line_width + item_width > width && !current.is_empty() {
            lines.push(Line::from(current));
            current = Vec::new();
            line_width = 0;
        }

        current.extend(spans);
        current.push(Span::styled("│ ", Style::default().fg(TEXT_MUTED)));
        line_width += item_width + 2;
    }

    if !current.is_empty() {
        lines.push(Line::from(current));
    }

    lines
}

fn draw_runtime_card(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let lines = vec![
        Line::from(vec![
            Span::styled("selected mode  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(selected.route_mode(), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("source count   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{}", selected.source_count()),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("attnres rms    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3}", selected.attn_res_rms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("partial rms    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{:.3} -> {:.3}",
                    selected.partial_rms_before, selected.partial_rms_after
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("probe hidden   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3}", app.diagnostics.hidden_rms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Runtime", AMBER).style(Style::default().bg(PANEL_ALT)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_attention_heatmap(frame: &mut Frame, area: Rect, app: &App, compact: bool) {
    let block = panel("Depth Routing Heatmap", CYAN).style(Style::default().bg(PANEL_BG));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let max_rows = inner.height.saturating_sub(2) as usize;
    let rows = visible_window(
        app.diagnostics.sublayers.len(),
        max_rows.saturating_sub(1),
        app.selected_sublayer,
    );
    let cell_width = if compact { 3 } else { 6 };
    let mut lines = Vec::new();

    let mut header = vec![Span::styled(
        if compact { "      " } else { "          " },
        Style::default(),
    )];
    for col in 0..app.diagnostics.max_sources {
        let label = if col == 0 {
            "Emb".to_string()
        } else if col == app.diagnostics.max_sources - 1 {
            "Part".to_string()
        } else {
            format!("B{col}")
        };
        header.push(Span::styled(
            format!("{label:>width$}", width = cell_width),
            Style::default().fg(TEXT_MUTED),
        ));
    }
    lines.push(Line::from(header));

    for snapshot in &app.diagnostics.sublayers[rows.clone()] {
        let label = if snapshot.index == app.selected_sublayer {
            format!("▸ {:<6}", snapshot.chip())
        } else {
            format!("  {:<6}", snapshot.chip())
        };
        let mut spans = vec![Span::styled(
            label,
            Style::default()
                .fg(if snapshot.index == app.selected_sublayer {
                    snapshot.kind.accent()
                } else {
                    TEXT_DIM
                })
                .add_modifier(if snapshot.index == app.selected_sublayer {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        )];

        for col in 0..app.diagnostics.max_sources {
            if col < snapshot.weights.len() {
                let weight = snapshot.weights[col];
                let content = if compact {
                    "██".to_string()
                } else {
                    format!("{:>4.0}%", weight * 100.0)
                };
                spans.push(Span::styled(
                    format!("{content:>width$}", width = cell_width),
                    Style::default()
                        .fg(if weight > 0.55 { BG } else { TEXT })
                        .bg(heat_color(weight)),
                ));
            } else {
                spans.push(Span::styled(
                    format!("{:>width$}", "·", width = cell_width),
                    Style::default().fg(TEXT_MUTED),
                ));
            }
        }
        lines.push(Line::from(spans));
    }

    if rows.start > 0 || rows.end < app.diagnostics.sublayers.len() {
        lines.push(Line::from(Span::styled(
            format!(
                "showing rows {}-{} of {}",
                rows.start + 1,
                rows.end,
                app.diagnostics.sublayers.len()
            ),
            Style::default().fg(TEXT_MUTED).italic(),
        )));
    }

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn draw_selected_detail(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let block =
        panel("Selected Sublayer", selected.kind.accent()).style(Style::default().bg(PANEL_ALT));
    let lines = vec![
        Line::from(vec![
            Span::styled("identity   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{}  •  B{} slot {}/{}",
                    selected.label(),
                    selected.target_block,
                    selected.slot_in_block + 1,
                    app.block_size
                ),
                Style::default().fg(TEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("dominant   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} at {:.0}%",
                    selected.dominant_source_label,
                    selected.dominant_weight * 100.0
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("selective  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{:.0}%  entropy {:.2}",
                    selected.selectivity * 100.0,
                    selected.entropy
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("merge      ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                if selected.has_partial {
                    format!(
                        "cache {:.0}%  /  partial {:.0}%",
                        selected.inter_mass * 100.0,
                        selected.intra_mass * 100.0
                    )
                } else {
                    "cache only".to_string()
                },
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("query norm ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3}", selected.query_norm),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("boundary   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                if selected.boundary_before {
                    "yes: this sublayer opens a new residual block"
                } else {
                    "no: this sublayer writes into the current block"
                },
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_event_feed(frame: &mut Frame, area: Rect, app: &App) {
    let lines = if app.events.is_empty() {
        vec![Line::from(Span::styled(
            "No events yet.",
            Style::default().fg(TEXT_MUTED),
        ))]
    } else {
        app.events
            .iter()
            .map(|event| {
                Line::from(vec![
                    Span::styled("• ", Style::default().fg(CYAN)),
                    Span::styled(event.clone(), Style::default().fg(TEXT_DIM)),
                ])
            })
            .collect()
    };

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Event Feed", CYAN).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_algorithm_card(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let mut lines = Vec::new();
    for (index, (title, detail)) in ALGO_STEPS.iter().enumerate() {
        let active = index == app.algo_phase;
        lines.push(Line::from(vec![
            Span::styled(
                if active { "▶ " } else { "  " },
                Style::default().fg(if active {
                    selected.kind.accent()
                } else {
                    TEXT_MUTED
                }),
            ),
            Span::styled(
                format!("{}. {}", index + 1, title),
                Style::default()
                    .fg(if active { TEXT } else { TEXT_DIM })
                    .add_modifier(if active {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ]));
        lines.push(Line::from(Span::styled(
            format!("   {detail}"),
            Style::default().fg(TEXT_MUTED),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .block(
                panel("Algorithm Filmstrip", selected.kind.accent())
                    .style(Style::default().bg(PANEL_BG)),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_pipeline_identity(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let score_span = if selected.logits.is_empty() {
        0.0
    } else {
        let min = selected
            .logits
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max = selected
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        max - min
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("cursor    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(selected.label(), Style::default().fg(TEXT).bold()),
        ]),
        Line::from(vec![
            Span::styled("sources   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} -> {}",
                    selected.source_labels.join(" + "),
                    selected.route_mode()
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("logit span", Style::default().fg(TEXT_MUTED)),
            Span::styled(format!("{score_span:.4}"), Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("output    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "attnres {:.3}  •  sublayer {:.3}",
                    selected.attn_res_rms, selected.sublayer_out_rms
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("block     ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "writes into B{}  •  slot {}/{}",
                    selected.target_block,
                    selected.slot_in_block + 1,
                    app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Inspector", AMBER).style(Style::default().bg(PANEL_ALT)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_metric_bars(frame: &mut Frame, area: Rect, app: &App, logits: bool) {
    let selected = app.selected_snapshot();
    let title = if logits {
        "Pre-Softmax Depth Scores"
    } else {
        "Softmax Routing Mass"
    };
    let accent = if logits { CORAL } else { CYAN };
    let block = panel(title, accent).style(Style::default().bg(PANEL_BG));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let values = if logits {
        &selected.logits
    } else {
        &selected.weights
    };
    let max_abs = if logits {
        values
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.001f32, f32::max)
    } else {
        values.iter().copied().fold(0.001f32, f32::max)
    };
    let width = inner.width.saturating_sub(22) as usize;

    let lines: Vec<Line> = selected
        .source_labels
        .iter()
        .zip(values.iter())
        .map(|(label, value)| {
            let magnitude = if logits {
                value.abs() / max_abs
            } else {
                *value / max_abs
            };
            let filled = ((width as f32) * magnitude).round() as usize;
            let bar = "█".repeat(filled.min(width));
            let color = if logits {
                if *value >= 0.0 {
                    CORAL
                } else {
                    BLUE
                }
            } else {
                heat_color(*value)
            };
            Line::from(vec![
                Span::styled(format!("{label:<6} "), Style::default().fg(TEXT_MUTED)),
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(format!(" {:>7.4}", value), Style::default().fg(TEXT_DIM)),
            ])
        })
        .collect();

    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn draw_route_story(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let lines = vec![
        Line::from(vec![
            Span::styled("1  stack   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} sources => {}",
                    selected.source_count(),
                    selected.source_labels.join(", ")
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("2  norm    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "RMS-normalized before scoring; ||w_l|| = {:.3}",
                    selected.query_norm
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("3  score   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "dominant logit source is {}",
                    selected.source_labels[selected.dominant_source_idx]
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("4  soften  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "softmax over depth yields {:.0}% to {}",
                    selected.dominant_weight * 100.0,
                    selected.dominant_source_label
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("5  write   ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "partial RMS {:.3} -> {:.3}; sublayer contributes {:.3}",
                    selected.partial_rms_before,
                    selected.partial_rms_after,
                    selected.sublayer_out_rms
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            if selected.boundary_before {
                "This sublayer starts a new block, so its output writes into a fresh residual accumulator."
            } else {
                "This sublayer keeps accumulating into the current residual block."
            },
            Style::default().fg(TEXT),
        )),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("What Happened Here", MINT).style(Style::default().bg(PANEL_ALT)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_block_schedule(frame: &mut Frame, area: Rect, app: &App) {
    let selected_block = app.selected_snapshot().target_block;
    let lines: Vec<Line> = app
        .diagnostics
        .block_summaries
        .iter()
        .map(|summary| {
            let label_style = if summary.block_idx == selected_block {
                Style::default().fg(BG).bg(CYAN).bold()
            } else {
                Style::default().fg(TEXT).bg(SLATE)
            };
            let chips: Vec<Span> = app
                .diagnostics
                .sublayers
                .iter()
                .filter(|snapshot| snapshot.target_block == summary.block_idx)
                .map(|snapshot| {
                    if snapshot.index == app.selected_sublayer {
                        Span::styled(
                            format!(" {} ", snapshot.chip()),
                            Style::default().fg(BG).bg(snapshot.kind.accent()).bold(),
                        )
                    } else {
                        Span::styled(
                            format!(" {} ", snapshot.chip()),
                            Style::default().fg(snapshot.kind.accent()),
                        )
                    }
                })
                .collect();

            let mut spans = vec![
                Span::styled(format!(" B{} ", summary.block_idx), label_style),
                Span::raw(" "),
            ];
            spans.extend(chips);
            spans.push(Span::styled(
                format!(
                    "  {} sl  sel {:.0}%  ||w|| {:.3}",
                    summary.sublayers,
                    summary.avg_selectivity * 100.0,
                    summary.avg_query_norm
                ),
                Style::default().fg(TEXT_MUTED),
            ));
            Line::from(spans)
        })
        .collect();

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Block Schedule", CYAN).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_two_phase_schedule(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    let snapshots: Vec<&SublayerSnapshot> = app
        .diagnostics
        .sublayers
        .iter()
        .filter(|snapshot| snapshot.target_block == selected.target_block)
        .collect();

    let phase1 = snapshots
        .iter()
        .map(|snapshot| {
            format!(
                "{}({})",
                snapshot.chip(),
                snapshot
                    .source_count()
                    .saturating_sub(snapshot.has_partial as usize)
            )
        })
        .collect::<Vec<_>>()
        .join("  ");
    let phase2 = snapshots
        .iter()
        .map(|snapshot| snapshot.chip())
        .collect::<Vec<_>>()
        .join(" -> ");

    let lines = vec![
        Line::from(vec![
            Span::styled("phase 1  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "parallel over cached blocks for B{}: {}",
                    selected.target_block, phase1
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("phase 2  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("sequential intra-block merge: {phase2}"),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Numbers in parentheses show how many cached sources each query can see before the partial is merged.",
            Style::default().fg(TEXT_MUTED),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("selected ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{}  •  cache {:.0}% / partial {:.0}%",
                    selected.label(),
                    selected.inter_mass * 100.0,
                    selected.intra_mass * 100.0
                ),
                Style::default().fg(TEXT),
            ),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Two-Phase Schedule", CORAL).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_merge_card(frame: &mut Frame, area: Rect, app: &App) {
    let selected = app.selected_snapshot();
    frame.render_widget(
        panel("Online Merge", AMBER).style(Style::default().bg(PANEL_ALT)),
        area,
    );
    let inner = panel("Online Merge", AMBER).inner(area);

    let gauge_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(3),
        ])
        .split(inner);

    let cache = Gauge::default()
        .ratio(selected.inter_mass as f64)
        .label(Span::styled(
            format!("cache {:.0}%", selected.inter_mass * 100.0),
            Style::default().fg(TEXT).bold(),
        ))
        .use_unicode(true)
        .gauge_style(Style::default().fg(CYAN).bg(SLATE))
        .style(Style::default().bg(PANEL_ALT));
    frame.render_widget(cache, gauge_rows[0]);

    let partial = Gauge::default()
        .ratio(selected.intra_mass as f64)
        .label(Span::styled(
            format!("partial {:.0}%", selected.intra_mass * 100.0),
            Style::default().fg(TEXT).bold(),
        ))
        .use_unicode(true)
        .gauge_style(Style::default().fg(AMBER).bg(SLATE))
        .style(Style::default().bg(PANEL_ALT));
    frame.render_widget(partial, gauge_rows[1]);

    let lines = vec![
        Line::from(vec![
            Span::styled("parity  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{:.2e} max diff vs standard forward",
                    app.diagnostics.two_phase_diff
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("note    ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                if selected.has_partial {
                    "The partial competes with cached blocks through the same normalized query."
                } else {
                    "No partial exists here yet, so only cached blocks contribute."
                },
                Style::default().fg(TEXT_DIM),
            ),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }),
        gauge_rows[2],
    );
}

fn draw_block_cache_card(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("cache state  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{} completed residual blocks + active B{}",
                    app.diagnostics.completed_residual_blocks, app.diagnostics.active_block
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("active fill  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!(
                    "{}/{} sublayers",
                    app.diagnostics.current_block_fill, app.block_size
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled("partial rms  ", Style::default().fg(TEXT_MUTED)),
            Span::styled(
                format!("{:.3}", app.diagnostics.partial_rms),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(""),
    ];

    for summary in &app.diagnostics.block_summaries {
        let bar_width = area.width.saturating_sub(24) as usize;
        let filled = (summary.avg_selectivity * bar_width as f32).round() as usize;
        let bar = "█".repeat(filled.min(bar_width));
        lines.push(Line::from(vec![
            Span::styled(
                format!("B{} ", summary.block_idx),
                Style::default().fg(TEXT_MUTED),
            ),
            Span::styled(
                bar,
                Style::default().fg(heat_color(summary.avg_selectivity)),
            ),
            Span::styled(
                format!(
                    " {:.0}%  ||w|| {:.3}",
                    summary.avg_selectivity * 100.0,
                    summary.avg_query_norm
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .block(panel("Block Cache Health", MINT).style(Style::default().bg(PANEL_BG)))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_footer(frame: &mut Frame, area: Rect, _app: &App) {
    frame.render_widget(Paragraph::new(hint_line()), area);
}

fn hint_line() -> Line<'static> {
    Line::from(vec![
        Span::styled("Space", Style::default().fg(TEXT).bold()),
        Span::styled(" train/pause  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("←/→", Style::default().fg(TEXT).bold()),
        Span::styled(" inspect  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("Tab", Style::default().fg(TEXT).bold()),
        Span::styled(" views  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("1/2/3", Style::default().fg(TEXT).bold()),
        Span::styled(" jump  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("?", Style::default().fg(TEXT).bold()),
        Span::styled(" help  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("r", Style::default().fg(TEXT).bold()),
        Span::styled(" reset  ", Style::default().fg(TEXT_MUTED)),
        Span::styled("q", Style::default().fg(TEXT).bold()),
        Span::styled(" quit", Style::default().fg(TEXT_MUTED)),
    ])
}

fn draw_help_overlay(frame: &mut Frame, area: Rect) {
    let popup = centered_rect(area, 74, 68);
    frame.render_widget(Clear, popup);
    let block = panel("Controls", CYAN).style(Style::default().bg(PANEL_BG));
    let lines = vec![
        Line::from(vec![
            Span::styled("Space", Style::default().fg(TEXT).bold()),
            Span::styled("  Start, pause, resume, or restart after completion", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("Up / Down", Style::default().fg(TEXT).bold()),
            Span::styled("  Increase or decrease training speed", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("Left / Right", Style::default().fg(TEXT).bold()),
            Span::styled("  Move the inspection cursor across Attn/MLP sublayers", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("Tab or 1/2/3", Style::default().fg(TEXT).bold()),
            Span::styled("  Switch between overview, pipeline, and inference views", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("r", Style::default().fg(TEXT).bold()),
            Span::styled("  Reinitialize the model and clear telemetry", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("?", Style::default().fg(TEXT).bold()),
            Span::styled("  Toggle this help overlay", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(vec![
            Span::styled("q / Esc", Style::default().fg(TEXT).bold()),
            Span::styled("  Quit the demo", Style::default().fg(TEXT_DIM)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Overview focuses on training health, heatmaps, and events. Pipeline explains the selected sublayer step by step. Inference shows how two-phase execution matches the standard forward pass.",
            Style::default().fg(TEXT),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        popup,
    );
}

fn panel<'a>(title: &'a str, accent: Color) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(BG).bg(accent).bold(),
        ))
}

fn centered_rect(area: Rect, width_percent: u16, height_percent: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_percent) / 2),
            Constraint::Percentage(height_percent),
            Constraint::Percentage((100 - height_percent) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(vertical[1])[1]
}

fn speed_dots(speed: usize) -> String {
    (1..=5)
        .map(|index| if index <= speed { '●' } else { '○' })
        .collect()
}

fn format_duration(seconds: f64) -> String {
    if seconds <= 0.0 {
        return "—".to_string();
    }
    let total = seconds.round() as u64;
    let mins = total / 60;
    let secs = total % 60;
    if mins > 0 {
        format!("{mins}m {secs:02}s")
    } else {
        format!("{secs}s")
    }
}

fn visible_window(total: usize, max_visible: usize, selected: usize) -> std::ops::Range<usize> {
    if total <= max_visible || max_visible == 0 {
        return 0..total;
    }
    let half = max_visible / 2;
    let mut start = selected.saturating_sub(half);
    let mut end = start + max_visible;
    if end > total {
        end = total;
        start = total.saturating_sub(max_visible);
    }
    start..end
}

fn sparkline(values: &[f64], width: usize) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() || width == 0 {
        return String::new();
    }

    let stride = (values.len().max(width) + width - 1) / width;
    let samples: Vec<f64> = values
        .chunks(stride.max(1))
        .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
        .collect();
    let min = samples
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value));
    let max = samples
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let range = (max - min).max(1e-6);

    samples
        .iter()
        .take(width)
        .map(|value| {
            let normalized = ((*value - min) / range).clamp(0.0, 1.0);
            let index = (normalized * (BLOCKS.len() - 1) as f64).round() as usize;
            BLOCKS[index]
        })
        .collect()
}

fn pill<T: Into<String>>(text: T, fg: Color, bg: Color) -> Span<'static> {
    Span::styled(
        format!(" {} ", text.into()),
        Style::default().fg(fg).bg(bg).bold(),
    )
}

fn heat_color(weight: f32) -> Color {
    let weight = weight.clamp(0.0, 1.0);
    let red = (26.0 + weight * 95.0) as u8;
    let green = (48.0 + weight * 170.0) as u8;
    let blue = (70.0 + weight * 185.0) as u8;
    Color::Rgb(red, green, blue)
}
