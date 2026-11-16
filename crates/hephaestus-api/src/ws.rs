//! WebSocket transport layer for streaming audio transcription.
//!
//! Handles connection lifecycle, query parameter validation, audio
//! buffering with windowed chunking, and PCM encoding conversion.
//! The actual ASR inference is wired in Plan 11-03; this module
//! provides the serving-layer foundation.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::response::IntoResponse;
use hephaestus_core::{InferenceInput, PipelineOutput};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

/// Query parameters for the `/ws/transcribe` WebSocket endpoint (D-05).
///
/// Extracted from the URL query string at connection upgrade time.
/// Invalid values are rejected with 400 Bad Request before allocating
/// WebSocket resources (T-11-06).
#[derive(Debug, Deserialize)]
pub struct TranscribeParams {
    /// Audio sample rate in Hz. Only 16000 is supported (D-08).
    pub sample_rate: u32,

    /// Channel label for the transcript (display string only, T-11-03).
    pub channel: String,

    /// PCM encoding format: `"f32"` or `"i16"` (D-06).
    pub encoding: String,
}

/// Supported PCM audio encodings (D-06).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoding {
    /// 32-bit IEEE 754 floating point, little-endian.
    F32,
    /// 16-bit signed integer, little-endian.
    I16,
}

impl AudioEncoding {
    /// Parse an encoding string into an [`AudioEncoding`] variant.
    ///
    /// Accepts `"f32"` and `"i16"` (case-sensitive). Returns an error
    /// message for anything else.
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f32" => Ok(Self::F32),
            "i16" => Ok(Self::I16),
            other => Err(format!(
                "unsupported encoding '{other}': valid encodings are 'f32' and 'i16'"
            )),
        }
    }
}

/// JSON transcript message sent back to clients (D-07).
///
/// Each windowed audio chunk produces one transcript message.
/// The `text` field is empty until Plan 11-03 wires inference.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptMessage {
    /// Channel label from the connection parameters.
    pub channel: String,

    /// Monotonically increasing chunk index within this connection.
    pub chunk_index: u64,

    /// Transcript text for this audio window (empty until inference is wired).
    pub text: String,
}

/// Convert little-endian i16 PCM bytes to f32 samples in [-1.0, 1.0].
///
/// Interprets every 2 bytes as a little-endian `i16` and normalizes
/// by dividing by 32768.0. Trailing bytes that do not form a complete
/// sample are silently ignored (T-11-02).
pub fn i16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Convert little-endian f32 PCM bytes to f32 samples.
///
/// Interprets every 4 bytes as a little-endian `f32`. Trailing bytes
/// that do not form a complete sample are silently ignored (T-11-02).
pub fn f32_bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if sample.is_finite() { sample } else { 0.0 }
        })
        .collect()
}

/// Accumulates PCM samples and emits fixed-size windows with overlap.
///
/// Audio frames arrive in variable-size chunks over the WebSocket.
/// The buffer accumulates samples and emits complete windows of
/// `window_samples` length, advancing by `window_samples - overlap_samples`
/// each time. This provides the windowed chunking needed for streaming
/// ASR (D-10).
///
/// A maximum buffer size of `2 * window_samples` prevents memory
/// exhaustion from fast senders or slow consumers (T-11-04).
pub struct AudioBuffer {
    /// Accumulated PCM samples (f32, normalized).
    samples: Vec<f32>,

    /// Number of samples per output window.
    window_samples: usize,

    /// Number of samples to overlap between consecutive windows.
    overlap_samples: usize,

    /// Monotonically increasing chunk counter.
    chunk_index: u64,
}

impl AudioBuffer {
    /// Create a new audio buffer with the given window and overlap durations.
    ///
    /// # Arguments
    ///
    /// * `window_secs` -- Duration of each output window in seconds.
    /// * `overlap_secs` -- Overlap between consecutive windows in seconds.
    /// * `sample_rate` -- Audio sample rate in Hz.
    pub fn new(window_secs: f32, overlap_secs: f32, sample_rate: u32) -> Self {
        let window_samples = (window_secs * sample_rate as f32) as usize;
        let overlap_samples = (overlap_secs * sample_rate as f32) as usize;

        // Guard against zero window (would cause infinite loop in push()).
        assert!(window_samples > 0, "window_samples must be > 0 (window_secs too small for sample rate)");
        assert!(
            window_samples > overlap_samples,
            "window_samples ({window_samples}) must exceed overlap_samples ({overlap_samples})"
        );

        Self {
            samples: Vec::with_capacity(window_samples),
            window_samples,
            overlap_samples,
            chunk_index: 0,
        }
    }

    /// Push new samples into the buffer and extract complete windows.
    ///
    /// Returns a vec of `(window_samples, chunk_index)` tuples for each
    /// complete window extracted. The vec may be empty if insufficient
    /// samples have accumulated.
    pub fn push(&mut self, new_samples: &[f32]) -> Vec<(Vec<f32>, u64)> {
        self.samples.extend_from_slice(new_samples);

        // Cap buffer at 2x window_samples to prevent memory exhaustion (T-11-04).
        let max_size = self.window_samples * 2;
        if self.samples.len() > max_size {
            let excess = self.samples.len() - max_size;
            tracing::warn!(
                excess_samples = excess,
                "audio buffer exceeded max size, draining oldest samples"
            );
            self.samples.drain(..excess);
        }

        let mut windows = Vec::new();
        let step = self.window_samples - self.overlap_samples;

        while self.samples.len() >= self.window_samples {
            let window = self.samples[..self.window_samples].to_vec();
            let idx = self.chunk_index;
            self.chunk_index += 1;
            self.samples.drain(..step);
            windows.push((window, idx));
        }

        windows
    }

    /// Flush remaining samples as a final short window.
    ///
    /// Returns `None` if the buffer is empty. Otherwise returns the
    /// remaining samples with the current chunk index and increments
    /// the counter.
    pub fn flush(&mut self) -> Option<(Vec<f32>, u64)> {
        if self.samples.is_empty() {
            return None;
        }

        let remaining = std::mem::take(&mut self.samples);
        let idx = self.chunk_index;
        self.chunk_index += 1;
        Some((remaining, idx))
    }
}

/// WebSocket upgrade handler for `/ws/transcribe`.
///
/// Validates query parameters (sample rate and encoding) before
/// upgrading the connection. Invalid parameters are rejected with
/// 400 Bad Request, avoiding resource allocation for bad requests
/// (T-11-06).
pub async fn ws_transcribe(
    ws: WebSocketUpgrade,
    Query(params): Query<TranscribeParams>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ApiError> {
    // Gate on readiness (consistent with HTTP and gRPC handlers).
    if !state.is_ready() {
        return Err(ApiError::NotReady);
    }

    // Validate channel label length.
    if params.channel.len() > 256 {
        return Err(ApiError::BadRequest(
            "channel label must be 256 characters or fewer".into(),
        ));
    }

    // Validate sample rate (D-08).
    if params.sample_rate != 16000 {
        return Err(ApiError::BadRequest(format!(
            "unsupported sample_rate {}: only 16000 Hz is supported",
            params.sample_rate
        )));
    }

    // Validate encoding (D-06).
    AudioEncoding::from_str(&params.encoding).map_err(ApiError::BadRequest)?;

    Ok(ws.on_upgrade(move |socket| handle_transcribe_socket(socket, params, state)))
}

/// Handle an established WebSocket connection for audio transcription.
///
/// Receives binary audio frames, converts PCM bytes to f32 samples,
/// buffers with windowed chunking, runs ASR inference via the pipeline,
/// and sends back JSON transcript messages with real transcription text.
///
/// Enforces a 30-second idle timeout to prevent connection slot
/// exhaustion from idle clients (T-11-05).
async fn handle_transcribe_socket(
    socket: WebSocket,
    params: TranscribeParams,
    state: Arc<AppState>,
) {
    use futures_util::SinkExt;

    let encoding = match AudioEncoding::from_str(&params.encoding) {
        Ok(e) => e,
        Err(msg) => {
            tracing::error!(error = %msg, "invalid encoding after validation");
            return;
        }
    };

    let mut buffer = AudioBuffer::new(
        state.window_size_secs(),
        state.overlap_secs(),
        params.sample_rate,
    );
    let idle_timeout = std::time::Duration::from_secs(30);

    let (mut sender, mut receiver) = {
        use futures_util::StreamExt;
        socket.split()
    };

    loop {
        let msg = match tokio::time::timeout(idle_timeout, {
            use futures_util::StreamExt;
            receiver.next()
        })
        .await
        {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(e))) => {
                tracing::warn!(error = %e, "websocket receive error");
                break;
            }
            Ok(None) => {
                // Stream ended (client closed).
                break;
            }
            Err(_) => {
                tracing::info!("websocket idle timeout, closing connection");
                break;
            }
        };

        match msg {
            Message::Binary(data) => {
                let samples = match encoding {
                    AudioEncoding::I16 => i16_bytes_to_f32(&data),
                    AudioEncoding::F32 => f32_bytes_to_samples(&data),
                };

                let windows = buffer.push(&samples);
                for (window, chunk_index) in windows {
                    let text = match run_asr_inference(&state, window).await {
                        Ok(t) => t,
                        Err(e) => {
                            tracing::error!(
                                model_id = %state.model_id(),
                                error = %e,
                                chunk_index,
                                "ASR inference failed"
                            );
                            // Send sanitized error to client (no internal details).
                            let error_msg = serde_json::json!({
                                "error": "inference failed",
                                "chunk_index": chunk_index,
                            });
                            if let Ok(json) = serde_json::to_string(&error_msg) {
                                if let Err(e) = sender.send(Message::Text(json.into())).await {
                                    tracing::warn!(error = %e, "failed to send error message");
                                    return;
                                }
                            }
                            continue;
                        }
                    };

                    let transcript = TranscriptMessage {
                        channel: params.channel.clone(),
                        chunk_index,
                        text,
                    };

                    tracing::debug!(
                        chunk_index,
                        text_len = transcript.text.len(),
                        "emitting transcript for audio window"
                    );

                    if let Ok(json) = serde_json::to_string(&transcript) {
                        if let Err(e) = sender.send(Message::Text(json.into())).await {
                            tracing::warn!(error = %e, "failed to send transcript message");
                            return;
                        }
                    }
                }
            }
            Message::Close(_) => {
                break;
            }
            _ => {
                // Ignore ping/pong/text frames.
            }
        }
    }

    // Flush remaining samples on close.
    if let Some((remaining, chunk_index)) = buffer.flush() {
        match run_asr_inference(&state, remaining).await {
            Ok(text) => {
                let transcript = TranscriptMessage {
                    channel: params.channel.clone(),
                    chunk_index,
                    text,
                };
                if let Ok(json) = serde_json::to_string(&transcript) {
                    if let Err(e) = sender.send(Message::Text(json.into())).await {
                        tracing::debug!(error = %e, "failed to send final transcript on close");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    model_id = %state.model_id(),
                    error = %e,
                    "ASR inference failed on flush"
                );
            }
        }
    }
}

/// Run ASR inference on a window of audio samples.
///
/// Takes a place in the inference queue, prepares the window (feature
/// extraction), then runs on the inference worker thread.
/// Extracts transcript text from the pipeline output.
async fn run_asr_inference(
    state: &AppState,
    samples: Vec<f32>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // A window has no deadline; a full queue fails it at once.
    let ticket = state.admit(None)?;

    // Feature extraction.
    let prepared = state.prepare(InferenceInput::Audio(samples))?;

    // The ONNX session runs on the inference worker thread.
    let output = ticket.run(prepared, None).await?.output;

    match output {
        PipelineOutput::Asr(text) => Ok(text),
        other => {
            tracing::warn!(
                output_type = ?std::mem::discriminant(&other),
                "expected ASR output from pipeline, got different variant"
            );
            // Convert non-ASR output to text representation as fallback.
            Ok(other.to_json()["text"]
                .as_str()
                .unwrap_or("")
                .to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_buffer_computes_correct_window_and_overlap() {
        let buf = AudioBuffer::new(30.0, 1.0, 16000);
        assert_eq!(buf.window_samples, 480_000);
        assert_eq!(buf.overlap_samples, 16_000);
    }

    #[test]
    fn audio_buffer_push_returns_no_windows_when_insufficient() {
        let mut buf = AudioBuffer::new(1.0, 0.0, 16000);
        // Push fewer samples than one window (16000).
        let samples = vec![0.0_f32; 8000];
        let windows = buf.push(&samples);
        assert!(windows.is_empty());
    }

    #[test]
    fn audio_buffer_push_returns_one_window_at_exact_size() {
        let mut buf = AudioBuffer::new(1.0, 0.0, 16000);
        let samples = vec![0.5_f32; 16000];
        let windows = buf.push(&samples);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].0.len(), 16000);
        assert_eq!(windows[0].1, 0);
    }

    #[test]
    fn audio_buffer_push_returns_multiple_windows() {
        let mut buf = AudioBuffer::new(1.0, 0.0, 16000);
        let samples = vec![0.1_f32; 32000];
        let windows = buf.push(&samples);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].1, 0);
        assert_eq!(windows[1].1, 1);
    }

    #[test]
    fn audio_buffer_push_drains_correctly_with_overlap() {
        // 1-second window, 0.5-second overlap at 100 Hz for small test.
        let mut buf = AudioBuffer::new(1.0, 0.5, 100);
        assert_eq!(buf.window_samples, 100);
        assert_eq!(buf.overlap_samples, 50);

        // Push exactly 100 samples -- should yield 1 window.
        let samples = vec![1.0_f32; 100];
        let windows = buf.push(&samples);
        assert_eq!(windows.len(), 1);

        // After draining: step = 100 - 50 = 50, so 50 samples remain.
        // Push 50 more to reach 100 total again.
        let more_samples = vec![2.0_f32; 50];
        let windows = buf.push(&more_samples);
        assert_eq!(windows.len(), 1);

        // The first 50 samples of this window should be the overlap (1.0).
        assert_eq!(windows[0].0[0], 1.0);
        // The last 50 should be the new data (2.0).
        assert_eq!(windows[0].0[50], 2.0);
    }

    #[test]
    fn audio_buffer_flush_returns_none_on_empty() {
        let mut buf = AudioBuffer::new(1.0, 0.0, 16000);
        assert!(buf.flush().is_none());
    }

    #[test]
    fn audio_buffer_flush_returns_remaining_samples() {
        let mut buf = AudioBuffer::new(1.0, 0.0, 16000);
        let samples = vec![0.3_f32; 5000];
        buf.push(&samples);
        let flushed = buf.flush();
        assert!(flushed.is_some());
        let (remaining, idx) = flushed.unwrap();
        assert_eq!(remaining.len(), 5000);
        assert_eq!(idx, 0);
    }

    #[test]
    fn audio_buffer_flush_increments_chunk_index() {
        let mut buf = AudioBuffer::new(1.0, 0.0, 16000);
        // Push enough for one window + leftover.
        let samples = vec![0.0_f32; 20000];
        let windows = buf.push(&samples);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].1, 0);

        // Flush the remaining 4000 samples.
        let flushed = buf.flush().unwrap();
        assert_eq!(flushed.0.len(), 4000);
        assert_eq!(flushed.1, 1); // chunk_index incremented.
    }

    #[test]
    fn audio_buffer_caps_at_max_size() {
        // 1-second window, no overlap, 100 Hz.
        let mut buf = AudioBuffer::new(1.0, 0.0, 100);
        // Max size = 2 * 100 = 200.
        // Push 50 samples (partial window, no extraction).
        buf.push(&vec![0.0_f32; 50]);
        // Push 200 more -- total 250, exceeds cap of 200.
        // After cap enforcement: 200 samples remain. Then extraction: 2 windows.
        let windows = buf.push(&vec![0.0_f32; 200]);
        assert_eq!(windows.len(), 2);
    }

    #[test]
    fn i16_bytes_to_f32_known_value() {
        // i16 value 16384 in little-endian: 0x00, 0x40
        let bytes: &[u8] = &[0x00, 0x40];
        let samples = i16_bytes_to_f32(bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn i16_bytes_to_f32_empty_input() {
        let samples = i16_bytes_to_f32(&[]);
        assert!(samples.is_empty());
    }

    #[test]
    fn i16_bytes_to_f32_odd_length_ignores_trailing() {
        // 3 bytes: one complete i16 sample + 1 trailing byte.
        let bytes: &[u8] = &[0x00, 0x40, 0xFF];
        let samples = i16_bytes_to_f32(bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn f32_bytes_to_samples_roundtrip() {
        let original: f32 = 0.75;
        let bytes = original.to_le_bytes();
        let samples = f32_bytes_to_samples(&bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn f32_bytes_to_samples_empty() {
        let samples = f32_bytes_to_samples(&[]);
        assert!(samples.is_empty());
    }

    #[test]
    fn f32_bytes_to_samples_trailing_bytes_ignored() {
        // 4 bytes for one f32 + 2 trailing bytes.
        let original: f32 = 1.0;
        let mut bytes = original.to_le_bytes().to_vec();
        bytes.push(0xAB);
        bytes.push(0xCD);
        let samples = f32_bytes_to_samples(&bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn audio_encoding_from_str_valid() {
        assert_eq!(AudioEncoding::from_str("f32").unwrap(), AudioEncoding::F32);
        assert_eq!(AudioEncoding::from_str("i16").unwrap(), AudioEncoding::I16);
    }

    #[test]
    fn audio_encoding_from_str_invalid() {
        assert!(AudioEncoding::from_str("pcm").is_err());
        assert!(AudioEncoding::from_str("").is_err());
    }

    #[test]
    fn i16_bytes_to_f32_negative_value() {
        // i16 value -16384 in little-endian: 0x00, 0xC0
        let bytes: &[u8] = &[0x00, 0xC0];
        let samples = i16_bytes_to_f32(bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn i16_bytes_to_f32_zero() {
        let bytes: &[u8] = &[0x00, 0x00];
        let samples = i16_bytes_to_f32(bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0]).abs() < 1e-6);
    }
}
