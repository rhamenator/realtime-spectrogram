#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, StreamConfig};
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use eframe::egui;

struct FftHelper {
    planner: FftPlanner<f32>,
    fft: Arc<dyn Fft<f32>>,
    buffer: Vec<Complex32>,
}

impl FftHelper {
    fn new(size: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(size);
        let buffer = vec![Complex32::new(0.0, 0.0); size];
        Self {
            planner,
            fft,
            buffer,
        }
    }

    fn ensure_size(&mut self, size: usize) {
        if self.buffer.len() != size {
            self.fft = self.planner.plan_fft_forward(size);
            self.buffer.resize(size, Complex32::new(0.0, 0.0));
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "spectrogram-rs", about = "Realtime audio spectrogram")]
struct Args {
    #[arg(short, long, default_value = "1024")]
    chunk: usize,
    #[arg(short, long, default_value = "44100")]
    sample_rate: u32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let sample_rate = args.sample_rate;
    let chunk = args.chunk;

    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Spectrogram",
        options,
        Box::new(move |cc| Box::new(SpectrogramApp::new(cc, sample_rate, chunk))),
    );

    Ok(())
}

fn audio_thread(
    sample_rate: u32,
    chunk: usize,
    device_name: String,
    tx: mpsc::Sender<(Vec<f32>, Vec<f32>)>,
    running: Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .output_devices()?
        .find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
        .or_else(|| host.default_output_device())
        .ok_or_else(|| anyhow::anyhow!("No output device"))?;
    let config = device.default_output_config()?;
    let sample_format = config.sample_format();

    let mut stream_config: StreamConfig = config.clone().into();
    if sample_rate != stream_config.sample_rate.0 {
        eprintln!(
            "Requested sample rate {sample_rate} not supported; using {}",
            stream_config.sample_rate.0
        );
    }
    stream_config.buffer_size = BufferSize::Fixed(chunk as u32);

    let buffer_l = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buffer_r = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let buf_l = buffer_l.clone();
    let buf_r = buffer_r.clone();
    let fft = Arc::new(Mutex::new(FftHelper::new(chunk)));

    let err_fn = |err| eprintln!("Stream error: {}", err);

    let channels = stream_config.channels as usize;

    let stream = match sample_format {
        SampleFormat::F32 => {
            let fft = fft.clone();
            let buf_l = buf_l.clone();
            let buf_r = buf_r.clone();
            let tx = tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    handle_input(data, channels, &buf_l, &buf_r, chunk, &tx, &fft);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let fft = fft.clone();
            let buf_l = buf_l.clone();
            let buf_r = buf_r.clone();
            let tx = tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    let data_f32: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    handle_input(&data_f32, channels, &buf_l, &buf_r, chunk, &tx, &fft);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let fft = fft.clone();
            let buf_l = buf_l.clone();
            let buf_r = buf_r.clone();
            let tx = tx.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    let data_f32: Vec<f32> = data
                        .iter()
                        .map(|&s| s as f32 / u16::MAX as f32 - 0.5)
                        .collect();

                    handle_input(&data_f32, channels, &buf_l, &buf_r, chunk, &tx, &fft);
                },
                err_fn,
                None,
            )?
        }
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;
    while running.load(std::sync::atomic::Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(100));
    }

    Ok(())
}

fn handle_input(
    input: &[f32],
    channels: usize,
    buf_l: &Arc<Mutex<VecDeque<f32>>>,
    buf_r: &Arc<Mutex<VecDeque<f32>>>,
    chunk: usize,
    tx: &mpsc::Sender<(Vec<f32>, Vec<f32>)>,
    fft: &Arc<Mutex<FftHelper>>,
) {
    let mut left = buf_l.lock().unwrap();
    let mut right = buf_r.lock().unwrap();

    for frame in input.chunks(channels) {
        if let Some(&l) = frame.first() {
            left.push_back(l);
            if channels > 1 {
                right.push_back(frame[1]);
            } else {
                right.push_back(l);
            }
        }
    }

    while left.len() >= chunk && right.len() >= chunk {
        let db_l;
        let db_r;
        {
            let slice_l = left.make_contiguous();
            let slice_r = right.make_contiguous();
            let mut fft = fft.lock().unwrap();
            db_l = compute_fft_db(&slice_l[..chunk], &mut fft);
            db_r = compute_fft_db(&slice_r[..chunk], &mut fft);
        }
        let _ = left.drain(..chunk);
        let _ = right.drain(..chunk);
        if tx.send((db_l, db_r)).is_err() {
            return;
        }
    }
}
fn compute_fft_db(samples: &[f32], fft: &mut FftHelper) -> Vec<f32> {
    fft.ensure_size(samples.len());
    for (b, &s) in fft.buffer.iter_mut().zip(samples.iter()) {
        b.re = s;
        b.im = 0.0;
    }
    fft.fft.process(&mut fft.buffer);
    fft.buffer
        .iter()
        .take(samples.len() / 2 + 1)
        .map(|c| 20.0 * c.norm().max(1e-6).log10())
        .collect()
}

#[derive(Copy, Clone, PartialEq)]
enum ColorMap {
    BlueRed,
    Grayscale,
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Cividis,
    Turbo,
}

impl ColorMap {
    fn as_str(&self) -> &'static str {
        match self {
            ColorMap::BlueRed => "Blue/Red",
            ColorMap::Grayscale => "Grayscale",
            ColorMap::Viridis => "viridis",
            ColorMap::Plasma => "plasma",
            ColorMap::Inferno => "inferno",
            ColorMap::Magma => "magma",
            ColorMap::Cividis => "cividis",
            ColorMap::Turbo => "turbo",
        }
    }

    fn color(&self, t: f32) -> egui::Color32 {
        match self {
            ColorMap::BlueRed => {
                egui::Color32::from_rgb((t * 255.0) as u8, 0, ((1.0 - t) * 255.0) as u8)
            }

            ColorMap::Grayscale => {
                let v = (t * 255.0) as u8;
                egui::Color32::from_gray(v)
            }
            ColorMap::Viridis => {
                let [r, g, b, _] = colorgrad::viridis().at(t as f64).to_rgba8();
                egui::Color32::from_rgb(r, g, b)
            }
            ColorMap::Plasma => {
                let [r, g, b, _] = colorgrad::plasma().at(t as f64).to_rgba8();
                egui::Color32::from_rgb(r, g, b)
            }
            ColorMap::Inferno => {
                let [r, g, b, _] = colorgrad::inferno().at(t as f64).to_rgba8();
                egui::Color32::from_rgb(r, g, b)
            }
            ColorMap::Magma => {
                let [r, g, b, _] = colorgrad::magma().at(t as f64).to_rgba8();
                egui::Color32::from_rgb(r, g, b)
            }
            ColorMap::Cividis => {
                let [r, g, b, _] = colorgrad::cividis().at(t as f64).to_rgba8();
                egui::Color32::from_rgb(r, g, b)
            }
            ColorMap::Turbo => {
                let [r, g, b, _] = colorgrad::turbo().at(t as f64).to_rgba8();
                egui::Color32::from_rgb(r, g, b)
            }
        }
    }
}

impl Default for ColorMap {
    fn default() -> Self {
        Self::Viridis
    }
}

struct SpectrogramApp {
    rx: mpsc::Receiver<(Vec<f32>, Vec<f32>)>,
    sample_rate: u32,
    chunk: usize,
    running_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    handle: Option<std::thread::JoinHandle<()>>,
    history_l: VecDeque<Vec<f32>>,
    history_r: VecDeque<Vec<f32>>,
    /// Maximum number of columns to keep; matches display width
    max_columns: usize,
    freq_bins: usize,
    tex_l: Option<egui::TextureHandle>,
    tex_r: Option<egui::TextureHandle>,
    /// Stored spectrogram pixel columns for incremental updates
    pixels_l: std::collections::VecDeque<Vec<egui::Color32>>,
    pixels_r: std::collections::VecDeque<Vec<egui::Color32>>,
    min_db: f32,
    max_db: f32,
    colormap: ColorMap,
    interpolate: bool,
    show_config: bool,
    freq_min: f32,
    freq_max: f32,
    log_freq: bool,
    devices: Vec<String>,
    selected_device: String,
}

impl SpectrogramApp {
    fn new(_cc: &eframe::CreationContext<'_>, sample_rate: u32, chunk: usize) -> Self {
        let host = cpal::default_host();
        let devices: Vec<String> = host
            .output_devices()
            .map(|ds| ds.filter_map(|d| d.name().ok()).collect())
            .unwrap_or_default();
        let default_device = host
            .default_output_device()
            .and_then(|d| d.name().ok())
            .unwrap_or_else(|| devices.get(0).cloned().unwrap_or_default());

        let (tx, rx) = mpsc::channel();
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let run_clone = running.clone();
        let device_name = default_device.clone();
        let handle = thread::spawn(move || {
            let _ = audio_thread(sample_rate, chunk, device_name, tx, run_clone);
        });
        Self {
            rx,
            sample_rate,
            chunk,
            running_flag: Some(running),
            handle: Some(handle),
            history_l: VecDeque::new(),
            history_r: VecDeque::new(),
            max_columns: 0,
            freq_bins: chunk / 2 + 1,
            tex_l: None,
            tex_r: None,
            pixels_l: std::collections::VecDeque::new(),
            pixels_r: std::collections::VecDeque::new(),
            min_db: -90.0,
            max_db: 0.0,
            colormap: ColorMap::default(),
            interpolate: true,
            show_config: false,
            freq_min: 20.0,
            freq_max: sample_rate as f32 / 2.0,
            log_freq: false,
            devices,
            selected_device: default_device,
        }
    }

    /// Convert a single FFT frame into a column of colors (top frequency first)
    fn frame_to_column(&self, frame: &[f32]) -> Vec<egui::Color32> {
        let display_bins = self.freq_bins;
        let freq_min = self.freq_min.max(0.0);
        let freq_max = self.freq_max.min(self.sample_rate as f32 / 2.0);
        let mut column = Vec::with_capacity(display_bins);
        for y in (0..display_bins).rev() {
            let frac = y as f32 / (display_bins - 1) as f32;
            let freq = if self.log_freq {
                if freq_min <= 0.0 {
                    0.0
                } else {
                    freq_min * (freq_max / freq_min).powf(frac)
                }
            } else {
                freq_min + frac * (freq_max - freq_min)
            };
            let bin = (((freq / self.sample_rate as f32) * self.chunk as f32).round() as usize)
                .min(self.freq_bins - 1);
            let v = frame.get(bin).copied().unwrap_or(self.min_db);
            let t = ((v - self.min_db) / (self.max_db - self.min_db)).clamp(0.0, 1.0);
            let color = self.colormap.color(t);
            column.push(color);
        }
        column
    }

    fn start_audio(&mut self) {
        if self.running_flag.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let run_clone = running.clone();
        let sample_rate = self.sample_rate;
        let chunk = self.chunk;
        let device_name = self.selected_device.clone();
        let handle = thread::spawn(move || {
            let _ = audio_thread(sample_rate, chunk, device_name, tx, run_clone);
        });
        self.rx = rx;
        self.running_flag = Some(running);
        self.handle = Some(handle);
        self.freq_bins = chunk / 2 + 1;
        self.max_columns = 0;
        self.history_l.clear();
        self.history_r.clear();
        self.pixels_l.clear();
        self.pixels_r.clear();
        self.tex_l = None;
        self.tex_r = None;
    }

    fn stop_audio(&mut self) {
        if let Some(flag) = self.running_flag.take() {
            flag.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SpectrogramApp {
    fn drop(&mut self) {
        self.stop_audio();
    }
}

impl eframe::App for SpectrogramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(self.running_flag.is_none(), egui::Button::new("Start"))
                    .clicked()
                {
                    self.start_audio();
                }
                if ui
                    .add_enabled(self.running_flag.is_some(), egui::Button::new("Stop"))
                    .clicked()
                {
                    self.stop_audio();
                }

                ui.separator();
                if ui.button("Settings").clicked() {
                    self.show_config = true;
                }
            });
        });

        // Determine maximum history width based on available pixels
        let available_rect = ctx.available_rect();
        let target_width = available_rect.width().max(1.0).round() as usize;
        if target_width != self.max_columns {
            self.max_columns = target_width;
        }

        while self.pixels_l.len() > self.max_columns {
            self.pixels_l.pop_front();
            self.pixels_r.pop_front();
            self.history_l.pop_front();
            self.history_r.pop_front();
        }
        if self.show_config {
            let mut open = self.show_config;
            egui::Window::new("Settings")
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.add_enabled_ui(self.running_flag.is_none(), |ui| {
                        ui.label("Device:");
                        egui::ComboBox::from_id_source("device")
                            .selected_text(&self.selected_device)
                            .show_ui(ui, |ui| {
                                for d in &self.devices {
                                    ui.selectable_value(&mut self.selected_device, d.clone(), d);
                                }
                            });
                        ui.label("Sample Rate:");
                        egui::ComboBox::from_id_source("sample_rate")
                            .selected_text(self.sample_rate.to_string())
                            .show_ui(ui, |ui| {
                                for &sr in
                                    [8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000].iter()
                                {
                                    ui.selectable_value(&mut self.sample_rate, sr, sr.to_string());
                                }
                            });
                        ui.label("Chunk:");
                        egui::ComboBox::from_id_source("chunk")
                            .selected_text(self.chunk.to_string())
                            .show_ui(ui, |ui| {
                                for &c in [256, 512, 1024, 2048, 4096, 8192].iter() {
                                    ui.selectable_value(&mut self.chunk, c, c.to_string());
                                }
                            });
                    });
                    ui.separator();
                    ui.label("Min dB:");
                    ui.add(egui::DragValue::new(&mut self.min_db));
                    ui.label("Max dB:");
                    ui.add(egui::DragValue::new(&mut self.max_db));
                    ui.separator();
                    ui.label("Freq min (Hz):");
                    ui.add(
                        egui::DragValue::new(&mut self.freq_min)
                            .clamp_range(0.0..=self.sample_rate as f64 / 2.0),
                    );
                    ui.label("Freq max (Hz):");
                    ui.add(
                        egui::DragValue::new(&mut self.freq_max)
                            .clamp_range(0.0..=self.sample_rate as f64 / 2.0),
                    );
                    ui.checkbox(&mut self.log_freq, "Log frequency scale");
                    ui.separator();
                    egui::ComboBox::from_id_source("colormap")
                        .selected_text(self.colormap.as_str())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::BlueRed,
                                ColorMap::BlueRed.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Grayscale,
                                ColorMap::Grayscale.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Viridis,
                                ColorMap::Viridis.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Plasma,
                                ColorMap::Plasma.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Inferno,
                                ColorMap::Inferno.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Magma,
                                ColorMap::Magma.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Cividis,
                                ColorMap::Cividis.as_str(),
                            );
                            ui.selectable_value(
                                &mut self.colormap,
                                ColorMap::Turbo,
                                ColorMap::Turbo.as_str(),
                            );
                        });
                    ui.checkbox(&mut self.interpolate, "Interpolate");
                    if ui.button("Apply").clicked() {
                        if self.running_flag.is_some() {
                            self.stop_audio();
                            self.start_audio();
                        }
                    }
                });
            self.show_config = open;
        }

        let mut updated = false;
        while let Ok((left, right)) = self.rx.try_recv() {
            let col_l = self.frame_to_column(&left);
            let col_r = self.frame_to_column(&right);
            self.pixels_l.push_back(col_l);
            self.pixels_r.push_back(col_r);
            self.history_l.push_back(left);
            self.history_r.push_back(right);
            if self.pixels_l.len() > self.max_columns {
                self.pixels_l.pop_front();
                self.pixels_r.pop_front();
                self.history_l.pop_front();
                self.history_r.pop_front();
            }
            updated = true;
        }

        let width = self.pixels_l.len();
        if updated && width > 0 {
            let display_bins = self.freq_bins;
            let mut flat_l: Vec<egui::Color32> = Vec::with_capacity(width * display_bins);
            let mut flat_r: Vec<egui::Color32> = Vec::with_capacity(width * display_bins);
            for y in 0..display_bins {
                for col in &self.pixels_l {
                    flat_l.push(col[y]);
                }
                for col in &self.pixels_r {
                    flat_r.push(col[y]);
                }
            }
            let tex_options = if self.interpolate {
                egui::TextureOptions::LINEAR
            } else {
                egui::TextureOptions::NEAREST
            };

            let image_l = egui::ColorImage {
                size: [width, display_bins],
                pixels: flat_l,
            };
            if let Some(tex) = &mut self.tex_l {
                tex.set(image_l, tex_options);
            } else {
                self.tex_l = Some(ctx.load_texture("spec_l", image_l, tex_options));
            }
            let image_r = egui::ColorImage {
                size: [width, display_bins],
                pixels: flat_r,
            };
            if let Some(tex) = &mut self.tex_r {
                tex.set(image_r, tex_options);
            } else {
                self.tex_r = Some(ctx.load_texture("spec_r", image_r, tex_options));
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let half_h = available.y / 2.0;
            if let (Some(l), Some(r)) = (&self.tex_l, &self.tex_r) {
                ui.label("Left Channel");
                ui.add(
                    egui::Image::from_texture(l).fit_to_exact_size(egui::vec2(available.x, half_h)),
                );
                ui.separator();
                ui.label("Right Channel");
                ui.add(
                    egui::Image::from_texture(r).fit_to_exact_size(egui::vec2(available.x, half_h)),
                );
            } else if self.running_flag.is_some() {
                ui.label("Waiting for audio...");
            } else {
                ui.label("Audio stopped");
            }
        });

        if updated {
            ctx.request_repaint();
        }
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}
