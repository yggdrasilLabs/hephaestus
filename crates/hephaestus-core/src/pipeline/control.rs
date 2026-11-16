//! Per-request limits on a generation, and the token counts it reports.

use std::sync::Arc;
use std::time::Instant;

use crate::error::CoreError;

use super::PipelineOutput;

/// Limits on one generation, checked between token steps.
///
/// The token loops of the text generation and encoder-decoder seq2seq
/// pipelines ask before every step whether to go on. Single-pass profiles
/// run one forward pass and ignore the control.
///
/// The default control limits nothing.
///
/// ```
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicBool, Ordering};
/// use std::time::{Duration, Instant};
///
/// use hephaestus_core::GenerationControl;
///
/// let cancelled = Arc::new(AtomicBool::new(false));
/// let flag = Arc::clone(&cancelled);
/// let control = GenerationControl::new()
///     .with_deadline(Instant::now() + Duration::from_secs(30))
///     .with_cancel_check(move || flag.load(Ordering::Relaxed));
/// # let _ = control;
/// ```
#[derive(Clone, Default)]
#[must_use = "a control does nothing until passed to an execute_with method"]
pub struct GenerationControl {
    deadline: Option<Instant>,
    cancelled: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl GenerationControl {
    /// A control that limits nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stop the generation with [`CoreError::Timeout`] once `deadline` has
    /// passed.
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Stop the generation with [`CoreError::Cancelled`] once `cancelled`
    /// returns `true`. It is called once per token step, on the thread
    /// that runs the model, so it must be cheap.
    pub fn with_cancel_check(
        mut self,
        cancelled: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        self.cancelled = Some(Arc::new(cancelled));
        self
    }

    /// This control, with its deadline no later than `deadline`.
    pub(super) fn limited_to(&self, deadline: Option<Instant>) -> Self {
        let deadline = match (self.deadline, deadline) {
            (Some(own), Some(other)) => Some(own.min(other)),
            (own, other) => own.or(other),
        };
        Self {
            deadline,
            cancelled: self.cancelled.clone(),
        }
    }

    /// Fail if the generation `what` must stop before its next step,
    /// `generated` tokens in.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Cancelled`] if the caller cancelled, else
    /// [`CoreError::Timeout`] if the deadline has passed.
    pub(super) fn check(&self, what: &str, generated: usize) -> Result<(), CoreError> {
        if self.cancelled.as_ref().is_some_and(|cancelled| cancelled()) {
            return Err(CoreError::Cancelled(format!(
                "{what} cancelled after {generated} tokens",
            )));
        }
        match self.deadline {
            Some(deadline) if Instant::now() >= deadline => Err(CoreError::Timeout(format!(
                "{what} exceeded its time budget after {generated} tokens",
            ))),
            _ => Ok(()),
        }
    }
}

/// Generated tokens per second of wall time, for the log line of a
/// finished generation. Zero when no time was measured.
pub(super) fn tokens_per_second(generated: usize, elapsed: std::time::Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        generated as f64 / seconds
    } else {
        0.0
    }
}

/// Token counts of one generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsage {
    /// Tokens the model read: the whole prompt for text generation
    /// (template and system message included), the encoder input for
    /// seq2seq.
    pub prompt_tokens: usize,
    /// Tokens the model generated, the end token included when it
    /// produced one.
    pub completion_tokens: usize,
}

/// The result of [`PipelineKind::execute_with`](super::PipelineKind::execute_with).
pub struct Execution {
    /// The model's output.
    pub output: PipelineOutput,
    /// Token counts; `Some` for text generation and encoder-decoder
    /// seq2seq, `None` for every single-pass profile.
    pub usage: Option<TokenUsage>,
}

impl Execution {
    /// The JSON of the output ([`PipelineOutput::to_json`]), with
    /// `prompt_tokens` and `completion_tokens` added when the run counted
    /// tokens. Without counts it is the output's JSON, unchanged.
    pub fn to_json(&self) -> serde_json::Value {
        let mut json = self.output.to_json();
        if let (Some(usage), Some(object)) = (self.usage, json.as_object_mut()) {
            object.insert("prompt_tokens".to_string(), usage.prompt_tokens.into());
            object.insert(
                "completion_tokens".to_string(),
                usage.completion_tokens.into(),
            );
        }
        json
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::pipeline::Entity;

    #[test]
    fn default_control_never_stops_a_generation() {
        assert!(GenerationControl::new().check("generation", 0).is_ok());
    }

    #[test]
    fn passed_deadline_is_a_timeout_naming_the_progress() {
        // Arrange
        let control = GenerationControl::new().with_deadline(Instant::now());

        // Act
        let err = control
            .check("text generation", 7)
            .expect_err("deadline passed");

        // Assert
        assert!(matches!(err, CoreError::Timeout(_)), "{err:?}");
        assert!(
            err.to_string()
                .contains("text generation exceeded its time budget after 7 tokens")
        );
    }

    #[test]
    fn cancel_check_is_asked_every_time_and_wins_over_the_deadline() {
        // Arrange
        let flag = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&flag);
        let control = GenerationControl::new()
            .with_deadline(Instant::now() + Duration::from_secs(60))
            .with_cancel_check(move || seen.load(Ordering::SeqCst));

        // Act
        let before = control.check("generation", 1);
        flag.store(true, Ordering::SeqCst);
        let after = control
            .limited_to(Some(Instant::now()))
            .check("generation", 2);

        // Assert
        assert!(before.is_ok());
        let err = after.expect_err("cancelled");
        assert!(matches!(err, CoreError::Cancelled(_)), "{err:?}");
        assert!(err.to_string().contains("after 2 tokens"));
    }

    #[test]
    fn limiting_keeps_the_earlier_deadline() {
        let now = Instant::now();
        let later = now + Duration::from_secs(60);

        assert!(
            GenerationControl::new()
                .limited_to(None)
                .check("g", 0)
                .is_ok()
        );
        assert!(
            GenerationControl::new()
                .limited_to(Some(now))
                .check("g", 0)
                .is_err()
        );
        assert!(
            GenerationControl::new()
                .with_deadline(now)
                .limited_to(Some(later))
                .check("g", 0)
                .is_err()
        );
        assert!(
            GenerationControl::new()
                .with_deadline(later)
                .limited_to(Some(now))
                .check("g", 0)
                .is_err()
        );
        assert!(
            GenerationControl::new()
                .with_deadline(later)
                .limited_to(None)
                .check("g", 0)
                .is_ok()
        );
    }

    #[test]
    fn execution_without_usage_serializes_exactly_like_its_output() {
        // Every single-pass profile: the response must not change at all.
        let outputs = || {
            vec![
                PipelineOutput::Classifier {
                    label: "POSITIVE".to_string(),
                    score: 0.75,
                },
                PipelineOutput::Embeddings(vec![0.5, -0.25]),
                PipelineOutput::Seq2Seq("fused".to_string()),
                PipelineOutput::TokenClassifier(vec![Entity {
                    word: "Ada".to_string(),
                    entity: "PER".to_string(),
                    score: 0.5,
                    start: 0,
                    end: 3,
                }]),
                PipelineOutput::Asr("hello".to_string()),
            ]
        };
        for (output, expected) in outputs().into_iter().zip(outputs()) {
            // Act
            let execution = Execution {
                output,
                usage: None,
            };

            // Assert -- same value and same bytes.
            let expected = expected.to_json();
            assert_eq!(execution.to_json(), expected);
            assert_eq!(execution.to_json().to_string(), expected.to_string());
        }
    }

    #[test]
    fn execution_with_usage_adds_the_two_token_counts() {
        // Arrange
        let execution = Execution {
            output: PipelineOutput::TextGeneration("hi".to_string()),
            usage: Some(TokenUsage {
                prompt_tokens: 12,
                completion_tokens: 3,
            }),
        };

        // Act
        let json = execution.to_json();

        // Assert
        assert_eq!(
            json,
            serde_json::json!({ "generated_text": "hi", "prompt_tokens": 12, "completion_tokens": 3 })
        );
    }
}
