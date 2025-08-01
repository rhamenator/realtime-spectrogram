#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig, BufferSize};
use num_complex::Complex32;
use rustfft::FftPlanner;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use eframe::egui;

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

fn audio_thread(sample_rate: u32, chunk: usize, tx: mpsc::Sender<(Vec<f32>, Vec<f32>)>, running: Arc<std::sync::atomic::AtomicBool>) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
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

    let buffer_l = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buffer_r = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buf_l = buffer_l.clone();
    let buf_r = buffer_r.clone();

    let err_fn = |err| eprintln!("Stream error: {}", err);

    let channels = stream_config.channels as usize;

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                handle_input(data, channels, &buf_l, &buf_r, chunk, &tx);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                handle_input(&data_f32, channels, &buf_l, &buf_r, chunk, &tx);
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / u16::MAX as f32 - 0.5).collect();
                handle_input(&data_f32, channels, &buf_l, &buf_r, chunk, &tx);
            },
            err_fn,
            None,
        )?,
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
    buf_l: &Arc<Mutex<Vec<f32>>>,
    buf_r: &Arc<Mutex<Vec<f32>>>,
    chunk: usize,
    tx: &mpsc::Sender<(Vec<f32>, Vec<f32>)>,
) {
    let mut left = buf_l.lock().unwrap();
    let mut right = buf_r.lock().unwrap();

    for frame in input.chunks(channels) {
        if let Some(&l) = frame.get(0) {
            left.push(l);
            if channels > 1 {
                right.push(frame[1]);
            } else {
                right.push(l);
            }
        }
    }

    while left.len() >= chunk && right.len() >= chunk {
        let frame_l: Vec<f32> = left.drain(..chunk).collect();
        let frame_r: Vec<f32> = right.drain(..chunk).collect();
        let db_l = compute_fft_db(&frame_l);
        let db_r = compute_fft_db(&frame_r);
        if tx.send((db_l, db_r)).is_err() {
            return;
        }
    }
}

fn compute_fft_db(samples: &[f32]) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(samples.len());
    let mut buffer: Vec<Complex32> = samples.iter().map(|&s| Complex32 { re: s, im: 0.0 }).collect();
    fft.process(&mut buffer);
    buffer.iter()
        .take(samples.len()/2 + 1)
        .map(|c| 20.0 * c.norm().max(1e-6).log10())
        .collect()
}

#[derive(Copy, Clone, PartialEq)]
enum ColorMap {
    BlueRed,
    Grayscale,
}

impl ColorMap {
    fn as_str(&self) -> &'static str {
        match self {
            ColorMap::BlueRed => "Blue/Red",
            ColorMap::Grayscale => "Grayscale",
        }
    }

    fn color(&self, t: f32) -> egui::Color32 {
        match self {
            ColorMap::BlueRed => egui::Color32::from_rgb((t * 255.0) as u8, 0, ((1.0 - t) * 255.0) as u8),
            ColorMap::Grayscale => {
                let v = (t * 255.0) as u8;
                egui::Color32::from_gray(v)
            }
        }
    }
}

impl Default for ColorMap {
    fn default() -> Self { Self::BlueRed }
}

struct SpectrogramApp {
    rx: mpsc::Receiver<(Vec<f32>, Vec<f32>)>,
    sample_rate: u32,
    chunk: usize,
    running_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    handle: Option<std::thread::JoinHandle<()>>,
    history_l: Vec<Vec<f32>>,
    history_r: Vec<Vec<f32>>,
    max_frames: usize,
    freq_bins: usize,
    tex_l: Option<egui::TextureHandle>,
    tex_r: Option<egui::TextureHandle>,
    min_db: f32,
    max_db: f32,
    colormap: ColorMap,
}

impl SpectrogramApp {
    fn new(_cc: &eframe::CreationContext<'_>, sample_rate: u32, chunk: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let run_clone = running.clone();
        let handle = thread::spawn(move || {
            let _ = audio_thread(sample_rate, chunk, tx, run_clone);
        });
        Self {
            rx,
            sample_rate,
            chunk,
            running_flag: Some(running),
            handle: Some(handle),
            history_l: Vec::new(),
            history_r: Vec::new(),
            max_frames: 200,
            freq_bins: chunk / 2 + 1,
            tex_l: None,
            tex_r: None,
            min_db: -90.0,
            max_db: 0.0,
            colormap: ColorMap::default(),
        }
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
        let handle = thread::spawn(move || {
            let _ = audio_thread(sample_rate, chunk, tx, run_clone);
        });
        self.rx = rx;
        self.running_flag = Some(running);
        self.handle = Some(handle);
        self.freq_bins = chunk / 2 + 1;
        self.history_l.clear();
        self.history_r.clear();
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

    fn start_audio(&mut self) {
        if self.running_flag.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let run_clone = running.clone();
        let sample_rate = self.sample_rate;
        let chunk = self.chunk;
        let handle = thread::spawn(move || {
            let _ = audio_thread(sample_rate, chunk, tx, run_clone);
        });
        self.rx = rx;
        self.running_flag = Some(running);
        self.handle = Some(handle);
        self.freq_bins = chunk / 2 + 1;
        self.history.clear();
        self.texture = None;
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

impl Drop for SpectrogramApp {
    fn drop(&mut self) {
        self.stop_audio();
    }
}

impl eframe::App for SpectrogramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.add_enabled(self.running_flag.is_none(), egui::Button::new("Start")).clicked() {
                    self.start_audio();
                }
                if ui.add_enabled(self.running_flag.is_some(), egui::Button::new("Stop")).clicked() {
                    self.stop_audio();
                }

                ui.separator();

                ui.add_enabled_ui(self.running_flag.is_none(), |ui| {
                    ui.label("Sample Rate:");
                    ui.add(egui::DragValue::new(&mut self.sample_rate).clamp_range(8000..=96000));
                    ui.label("Chunk:");
                    ui.add(egui::DragValue::new(&mut self.chunk).clamp_range(256..=8192));
                });

                ui.separator();
                ui.label("Min dB:");
                ui.add(egui::DragValue::new(&mut self.min_db));
                ui.label("Max dB:");
                ui.add(egui::DragValue::new(&mut self.max_db));

                egui::ComboBox::from_id_source("colormap")
                    .selected_text(self.colormap.as_str())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.colormap, ColorMap::BlueRed, ColorMap::BlueRed.as_str());
                        ui.selectable_value(&mut self.colormap, ColorMap::Grayscale, ColorMap::Grayscale.as_str());
                    });
            });
        });

        while let Ok((left, right)) = self.rx.try_recv() {
            if self.history_l.len() >= self.max_frames {
                self.history_l.remove(0);
                self.history_r.remove(0);
            }
            self.history_l.push(left);
            self.history_r.push(right);
        }

        let mut pixels_l: Vec<u8> = Vec::new();
        let mut pixels_r: Vec<u8> = Vec::new();
        for y in (0..self.freq_bins).rev() {
            for frame in &self.history_l {
                let v = frame.get(y).copied().unwrap_or(self.min_db);
                let t = ((v - self.min_db) / (self.max_db - self.min_db)).clamp(0.0, 1.0);
                let color = self.colormap.color(t);
                pixels_l.extend_from_slice(&[color.r(), color.g(), color.b()]);
            }
            for frame in &self.history_r {
                let v = frame.get(y).copied().unwrap_or(self.min_db);
                let t = ((v - self.min_db) / (self.max_db - self.min_db)).clamp(0.0, 1.0);
                let color = self.colormap.color(t);
                pixels_r.extend_from_slice(&[color.r(), color.g(), color.b()]);
            }
        }

        if !pixels_l.is_empty() {
            let size = [self.history_l.len(), self.freq_bins];
            let image = egui::ColorImage::from_rgb(size, &pixels_l);
            let tex = self.tex_l.get_or_insert_with(|| ctx.load_texture("spec_l", image.clone(), Default::default()));
            tex.set(image, Default::default());
        }
        if !pixels_r.is_empty() {
            let size = [self.history_r.len(), self.freq_bins];
            let image = egui::ColorImage::from_rgb(size, &pixels_r);
            let tex = self.tex_r.get_or_insert_with(|| ctx.load_texture("spec_r", image.clone(), Default::default()));
            tex.set(image, Default::default());
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let half_h = available.y / 2.0;
            if let (Some(l), Some(r)) = (&self.tex_l, &self.tex_r) {
                ui.label("Left Channel");
                ui.add(egui::Image::from_texture(l).fit_to_exact_size(egui::vec2(available.x, half_h)));
                ui.separator();
                ui.label("Right Channel");
                ui.add(egui::Image::from_texture(r).fit_to_exact_size(egui::vec2(available.x, half_h)));
            } else {
                if self.running_flag.is_some() {
                    ui.label("Waiting for audio...");
                } else {
                    ui.label("Audio stopped");
                }
            }
        });

        ctx.request_repaint();
    }
}
