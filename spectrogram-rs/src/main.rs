use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig, SampleRate, BufferSize};
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

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let sample_rate = args.sample_rate;
    let chunk = args.chunk;
    let freq_bins = chunk / 2 + 1;

    thread::spawn(move || {
        if let Err(e) = audio_thread(sample_rate, chunk, tx) {
            eprintln!("Audio thread error: {e}");
        }
    });

    let options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Spectrogram",
        options,
        Box::new(move |cc| Box::new(SpectrogramApp::new(cc, rx, freq_bins))),
    );

    Ok(())
}

fn audio_thread(sample_rate: u32, chunk: usize, tx: mpsc::Sender<Vec<f32>>) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device"))?;
    let config = device.default_input_config()?;
    let sample_format = config.sample_format();

    let mut stream_config: StreamConfig = config.clone().into();
    stream_config.channels = 1;
    stream_config.sample_rate = SampleRate(sample_rate);
    stream_config.buffer_size = BufferSize::Fixed(chunk as u32);

    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let buf_clone = buffer.clone();

    let err_fn = |err| eprintln!("Stream error: {}", err);

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                handle_input(data, &buf_clone, chunk, &tx);
            },
            err_fn,
            None,
        )?,
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                handle_input(&data_f32, &buf_clone, chunk, &tx);
            },
            err_fn,
            None,
        )?,
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / u16::MAX as f32 - 0.5).collect();
                handle_input(&data_f32, &buf_clone, chunk, &tx);
            },
            err_fn,
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    stream.play()?;
    loop {
        thread::sleep(Duration::from_millis(100));
    }
}

fn handle_input(input: &[f32], buffer: &Arc<Mutex<Vec<f32>>>, chunk: usize, tx: &mpsc::Sender<Vec<f32>>) {
    let mut buf = buffer.lock().unwrap();
    buf.extend_from_slice(input);
    while buf.len() >= chunk {
        let frame: Vec<f32> = buf.drain(..chunk).collect();
        let db = compute_fft_db(&frame);
        if tx.send(db).is_err() {
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

struct SpectrogramApp {
    rx: mpsc::Receiver<Vec<f32>>,
    history: Vec<Vec<f32>>,
    max_frames: usize,
    freq_bins: usize,
    texture: Option<egui::TextureHandle>,
}

impl SpectrogramApp {
    fn new(_cc: &eframe::CreationContext<'_>, rx: mpsc::Receiver<Vec<f32>>, freq_bins: usize) -> Self {
        Self {
            rx,
            history: Vec::new(),
            max_frames: 200,
            freq_bins,
            texture: None,
        }
    }
}

impl eframe::App for SpectrogramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(frame) = self.rx.try_recv() {
            if self.history.len() >= self.max_frames {
                self.history.remove(0);
            }
            self.history.push(frame);
        }

        let mut pixels: Vec<u8> = Vec::new();
        for y in 0..self.freq_bins {
            for frame in &self.history {
                let v = frame.get(y).copied().unwrap_or(-90.0);
                let t = ((v + 90.0) / 90.0).clamp(0.0, 1.0);
                let color = egui::Color32::from_rgb((t * 255.0) as u8, 0, ((1.0 - t) * 255.0) as u8);
                pixels.extend_from_slice(&[color.r(), color.g(), color.b()]);
            }
        }
        if !pixels.is_empty() {
            let size = [self.history.len() as usize, self.freq_bins];
            let image = egui::ColorImage::from_rgb(size, &pixels);
            let tex = self.texture.get_or_insert_with(|| ctx.load_texture("spec", image.clone(), Default::default()));
            tex.set(image, Default::default());
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.texture {
                let available = ui.available_size();
                ui.add(egui::Image::from_texture(tex).fit_to_exact_size(available));
            } else {
                ui.label("Waiting for audio...");
            }
        });

        ctx.request_repaint();
    }
}
