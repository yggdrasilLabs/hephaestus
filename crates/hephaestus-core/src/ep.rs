//! Execution provider selection for ONNX Runtime sessions.
//!
//! Maps the `EXECUTION_PROVIDER` configuration value to the concrete
//! [`ort::ep`] types that register hardware-accelerated backends on a
//! session builder.

use std::fmt;
use std::str::FromStr;
use std::sync::RwLock;

use ort::ep::ExecutionProviderDispatch;

use crate::error::CoreError;

/// How the CUDA arena grows when it needs more device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaArenaExtendStrategy {
    /// Grow in chunks that double in size (ONNX Runtime's default): fewer
    /// device allocations, more memory held than is in use.
    #[default]
    NextPowerOfTwo,
    /// Grow by exactly what is asked for: the arena holds little more
    /// than its peak use.
    SameAsRequested,
}

impl fmt::Display for CudaArenaExtendStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NextPowerOfTwo => write!(f, "next_power_of_two"),
            Self::SameAsRequested => write!(f, "same_as_requested"),
        }
    }
}

impl FromStr for CudaArenaExtendStrategy {
    type Err = CoreError;

    /// Parse `next_power_of_two` or `same_as_requested` (case-insensitive).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Config`] for any other value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "next_power_of_two" => Ok(Self::NextPowerOfTwo),
            "same_as_requested" => Ok(Self::SameAsRequested),
            other => Err(CoreError::Config(format!(
                "unknown CUDA arena extend strategy '{other}'; accepted values: \
                 next_power_of_two, same_as_requested",
            ))),
        }
    }
}

/// Settings of the CUDA execution provider, for every session of the
/// process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaSettings {
    /// Index of the GPU to run on.
    pub device_id: i32,
    /// Cap on the device memory arena of one session, in bytes. `None`
    /// leaves it to ONNX Runtime (no cap). Memory CUDA and cuDNN allocate
    /// outside the arena is not counted.
    pub memory_limit_bytes: Option<usize>,
    /// How the arena grows.
    pub arena_extend_strategy: CudaArenaExtendStrategy,
}

static CUDA_SETTINGS: RwLock<CudaSettings> = RwLock::new(CudaSettings {
    device_id: 0,
    memory_limit_bytes: None,
    arena_extend_strategy: CudaArenaExtendStrategy::NextPowerOfTwo,
});

/// Set the CUDA settings used by every session created from now on.
///
/// Call once at startup, before the first pipeline is built.
pub fn configure_cuda(settings: CudaSettings) {
    *CUDA_SETTINGS
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = settings;
}

/// The CUDA settings in force.
pub fn cuda_settings() -> CudaSettings {
    *CUDA_SETTINGS
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// ONNX Runtime execution provider selection.
///
/// Each variant corresponds to a hardware backend. GPU variants
/// require the matching Cargo feature to be compiled in. A provider that
/// was asked for and cannot be used is an error at startup, never a
/// silent fallback to the CPU: without the feature
/// [`to_ort_providers`](Self::to_ort_providers) fails, and with it the
/// provider is registered so that a failure to set it up (no driver, no
/// CUDA libraries, no device) fails the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU backend (always available, the default).
    Cpu,
    /// NVIDIA CUDA backend (requires `cuda` feature).
    Cuda,
    /// NVIDIA TensorRT backend (requires `tensorrt` feature).
    TensorRt,
    /// Apple CoreML backend (requires `coreml` feature).
    CoreMl,
}

impl fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda => write!(f, "cuda"),
            Self::TensorRt => write!(f, "tensorrt"),
            Self::CoreMl => write!(f, "coreml"),
        }
    }
}

impl FromStr for ExecutionProvider {
    type Err = CoreError;

    /// Parse an execution provider name (case-insensitive).
    ///
    /// Accepted values: `cpu`, `cuda`, `tensorrt`, `coreml`.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Config`] for unrecognised values, listing
    /// the accepted set.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" => Ok(Self::TensorRt),
            "coreml" => Ok(Self::CoreMl),
            other => Err(CoreError::Config(format!(
                "unknown execution provider '{other}'; accepted values: cpu, cuda, tensorrt, coreml",
            ))),
        }
    }
}

impl ExecutionProvider {
    /// Convert to the [`ort`] execution provider dispatches to register
    /// on a session builder.
    ///
    /// - `Cpu` returns an empty vec (ort uses CPU by default when no
    ///   EPs are registered).
    /// - GPU variants return a single-element vec with the configured
    ///   provider, gated behind the corresponding Cargo feature. The
    ///   provider is marked so that the session builder fails if it
    ///   cannot be registered, instead of logging and going on without
    ///   it. CUDA takes its device, memory limit and arena strategy from
    ///   [`cuda_settings`].
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Config`] if the required Cargo feature was
    /// not compiled in (T-EP-01 strict allowlist).
    pub fn to_ort_providers(&self) -> Result<Vec<ExecutionProviderDispatch>, CoreError> {
        self.ort_providers(false)
    }

    /// As [`to_ort_providers`](Self::to_ort_providers), for the small
    /// helper sessions that sit next to a model session: on CUDA their
    /// arena grows by exactly what is asked for, so it holds no more
    /// than the tensors they return.
    pub(crate) fn helper_ort_providers(
        &self,
    ) -> Result<Vec<ExecutionProviderDispatch>, CoreError> {
        self.ort_providers(true)
    }

    /// ONNX Runtime's name for the provider, as it appears in its logs.
    pub fn ort_name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPUExecutionProvider",
            Self::Cuda => "CUDAExecutionProvider",
            Self::TensorRt => "TensorrtExecutionProvider",
            Self::CoreMl => "CoreMLExecutionProvider",
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn ort_providers(&self, helper: bool) -> Result<Vec<ExecutionProviderDispatch>, CoreError> {
        match self {
            Self::Cpu => Ok(Vec::new()),
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    use ort::ep::ArenaExtendStrategy;

                    let settings = cuda_settings();
                    let strategy = match settings.arena_extend_strategy {
                        _ if helper => ArenaExtendStrategy::SameAsRequested,
                        CudaArenaExtendStrategy::NextPowerOfTwo => {
                            ArenaExtendStrategy::NextPowerOfTwo
                        }
                        CudaArenaExtendStrategy::SameAsRequested => {
                            ArenaExtendStrategy::SameAsRequested
                        }
                    };
                    let mut cuda = ort::ep::CUDA::default()
                        .with_device_id(settings.device_id)
                        .with_arena_extend_strategy(strategy);
                    if let Some(limit) = settings.memory_limit_bytes {
                        cuda = cuda.with_memory_limit(limit);
                    }
                    Ok(vec![cuda.build().error_on_failure()])
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(CoreError::Config(
                        "EXECUTION_PROVIDER=cuda requires the 'cuda' cargo feature, which this \
                         binary was built without; rebuild with: cargo build --features cuda \
                         (there is no fallback to the CPU: set EXECUTION_PROVIDER=cpu to run \
                         on it)"
                            .to_string(),
                    ))
                }
            }
            Self::TensorRt => {
                #[cfg(feature = "tensorrt")]
                {
                    Ok(vec![
                        ort::ep::TensorRT::default().build().error_on_failure(),
                    ])
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    Err(CoreError::Config(
                        "EXECUTION_PROVIDER=tensorrt requires the 'tensorrt' cargo feature; \
                         rebuild with: cargo build --features tensorrt"
                            .to_string(),
                    ))
                }
            }
            Self::CoreMl => {
                #[cfg(feature = "coreml")]
                {
                    Ok(vec![ort::ep::CoreML::default().build().error_on_failure()])
                }
                #[cfg(not(feature = "coreml"))]
                {
                    Err(CoreError::Config(
                        "EXECUTION_PROVIDER=coreml requires the 'coreml' cargo feature; \
                         rebuild with: cargo build --features coreml"
                            .to_string(),
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all_valid_variants() {
        assert_eq!("cpu".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::Cpu);
        assert_eq!("cuda".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::Cuda);
        assert_eq!("tensorrt".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::TensorRt);
        assert_eq!("coreml".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::CoreMl);
    }

    #[test]
    fn parse_case_insensitive() {
        assert_eq!("CPU".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::Cpu);
        assert_eq!("CUDA".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::Cuda);
        assert_eq!("TensorRT".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::TensorRt);
        assert_eq!("CoreML".parse::<ExecutionProvider>().unwrap(), ExecutionProvider::CoreMl);
    }

    #[test]
    fn parse_rejects_unknown() {
        let err = "vulkan".parse::<ExecutionProvider>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown execution provider"), "got: {msg}");
        assert!(msg.contains("vulkan"), "should echo the bad value: {msg}");
        assert!(msg.contains("cpu, cuda, tensorrt, coreml"), "should list accepted: {msg}");
    }

    #[test]
    fn display_matches_parse_input() {
        for name in &["cpu", "cuda", "tensorrt", "coreml"] {
            let ep: ExecutionProvider = name.parse().unwrap();
            assert_eq!(&ep.to_string(), *name);
        }
    }

    #[test]
    fn arena_extend_strategy_parses_and_displays() {
        for name in ["next_power_of_two", "same_as_requested"] {
            let strategy: CudaArenaExtendStrategy = name.parse().unwrap();
            assert_eq!(strategy.to_string(), name);
        }
        assert_eq!(
            "Same_As_Requested"
                .parse::<CudaArenaExtendStrategy>()
                .unwrap(),
            CudaArenaExtendStrategy::SameAsRequested
        );
        let msg = "double"
            .parse::<CudaArenaExtendStrategy>()
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("next_power_of_two, same_as_requested"),
            "{msg}"
        );
    }

    #[test]
    fn cpu_returns_empty_providers() {
        let providers = ExecutionProvider::Cpu.to_ort_providers().unwrap();
        assert!(providers.is_empty());
    }

    // GPU variants without features compiled in should return Config error.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_without_feature_returns_error() {
        let err = ExecutionProvider::Cuda.to_ort_providers().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cuda"), "should mention cuda: {msg}");
        assert!(msg.contains("cargo feature"), "should mention feature: {msg}");
    }

    #[cfg(not(feature = "tensorrt"))]
    #[test]
    fn tensorrt_without_feature_returns_error() {
        let err = ExecutionProvider::TensorRt.to_ort_providers().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("tensorrt"), "should mention tensorrt: {msg}");
    }

    #[cfg(not(feature = "coreml"))]
    #[test]
    fn coreml_without_feature_returns_error() {
        let err = ExecutionProvider::CoreMl.to_ort_providers().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("coreml"), "should mention coreml: {msg}");
    }
}
