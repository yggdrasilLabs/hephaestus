//! Forward passes whose key/value cache stays on the session's device.
//!
//! `Session::run` hands every output back in host memory. On a GPU that
//! copies the whole cache to the host after each step and back before the
//! next one. A [`DeviceRun`] runs the same graph through an ONNX Runtime
//! I/O binding instead:
//!
//! - the `present.*` outputs and `logits` are bound to the device, so they
//!   are allocated there and never leave it;
//! - the cache tensors of the previous step are bound as inputs as they
//!   are, so nothing is copied to feed them back;
//! - the small token, mask and position tensors are bound from host
//!   memory, as before.
//!
//! Two things still have to happen to tensors that the host no longer
//! sees, and each is a tiny graph of its own, built here with the model
//! editor and run on the same device:
//!
//! - **the last position of the logits** (`Gather` on the sequence axis,
//!   then `Cast` to `float32`) is all that comes to the host:
//!   `rows x vocab` values instead of `rows x positions x vocab`;
//! - **moving cache rows** when a generation joins or leaves a batch
//!   (`Concat` of two caches, `Gather` of rows, `Where` to zero padding):
//!   see [`DeviceRun::gather_rows`]. The moved cache is written into
//!   memory of the model session's own arena, so the helper session
//!   holds no cache memory of its own.
//!
//! The device is whatever the execution provider computes on. With the
//! CPU provider it is host memory and everything above runs the same
//! way, which is how this path is tested without a GPU.

use std::sync::{Arc, Mutex, PoisonError};

use ort::editor::{Graph, Model, Node, ONNX_DOMAIN, Opset};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::operator::Attribute;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, RunOptions, Session};
use ort::value::{
    DynTensor, DynValue, Outlet, Shape, SymbolicDimensions, Tensor, TensorElementType, TensorRef,
    Value, ValueType, ValueTypeMarker,
};

use crate::ep::{self, ExecutionProvider};
use crate::error::CoreError;

use super::float::FloatType;
use super::text_generation::PastSlot;

/// Opset of the helper graphs: `Gather`, `Concat`, `Cast` and `Where` in
/// versions every execution provider has had for years.
const HELPER_OPSET: u32 = 17;

/// Where a session computes and keeps its outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ComputeDevice {
    device: AllocationDevice,
    id: i32,
}

impl ComputeDevice {
    /// The device of an execution provider; `None` for a provider this
    /// path has not been written for.
    pub(super) fn of(ep: &ExecutionProvider) -> Option<Self> {
        match ep {
            ExecutionProvider::Cpu => Some(Self {
                device: AllocationDevice::CPU,
                id: 0,
            }),
            ExecutionProvider::Cuda => Some(Self {
                device: AllocationDevice::CUDA,
                id: ep::cuda_settings().device_id,
            }),
            ExecutionProvider::TensorRt | ExecutionProvider::CoreMl => None,
        }
    }

    /// Whether the device is host memory.
    fn is_host(self) -> bool {
        self.device == AllocationDevice::CPU
    }

    /// Name for logs and error messages (`Cpu`, `Cuda:0`).
    pub(super) fn name(self) -> String {
        if self.is_host() {
            self.device.as_str().to_string()
        } else {
            format!("{}:{}", self.device.as_str(), self.id)
        }
    }

    fn memory_info(self) -> Result<MemoryInfo<'static>, CoreError> {
        let memory_type = if self.is_host() {
            MemoryType::CPUOutput
        } else {
            MemoryType::Default
        };
        MemoryInfo::new(self.device, self.id, AllocatorType::Device, memory_type)
            .map_err(|e| inference("cannot describe the compute device", &e))
    }

    /// Value of the run option that returns unused arena memory of this
    /// device after a run.
    fn arena_to_shrink(self) -> String {
        if self.is_host() {
            "cpu:0".to_string()
        } else {
            format!("gpu:{}", self.id)
        }
    }
}

fn inference(what: &str, error: &dyn std::fmt::Display) -> CoreError {
    CoreError::Inference(format!("{what}: {error}"))
}

fn load(what: &str, error: &dyn std::fmt::Display) -> CoreError {
    CoreError::ModelLoad(format!("{what}: {error}"))
}

/// An I/O binding the pipeline can own.
///
/// A pipeline is shared between threads behind a lock, so it must be
/// `Sync`; an `IoBinding` is only `Send`. It is only ever used through
/// `&mut`, which the mutex hands out without locking.
struct Binding(Mutex<IoBinding>);

impl Binding {
    fn of(session: &Session) -> Result<Self, ort::Error> {
        session
            .create_binding()
            .map(|binding| Self(Mutex::new(binding)))
    }

    fn get(&mut self) -> &mut IoBinding {
        self.0.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The allocator of the model session on its compute device.
///
/// Tensors created from it must be released before it is, so everything
/// that holds such a tensor holds one of these too (see
/// [`DeviceRun::gather_rows`]). An allocator is `Send` only; the mutex
/// makes it shareable, as for [`Binding`].
pub(super) struct DeviceAllocator(Mutex<Allocator>);

impl DeviceAllocator {
    /// A tensor of `element_type` and shape `dims` on the device.
    fn tensor(
        &self,
        element_type: TensorElementType,
        dims: &[i64],
    ) -> Result<DynTensor, ort::Error> {
        let allocator = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        DynTensor::new(&allocator, element_type, Shape::new(dims.iter().copied()))
    }
}

/// A helper graph with its own binding.
struct Helper {
    session: Session,
    binding: Binding,
}

/// A tensor input or output of a helper graph: every dimension dynamic.
fn outlet(name: &str, element_type: TensorElementType, rank: usize) -> Outlet {
    Outlet::new(
        name,
        ValueType::Tensor {
            ty: element_type,
            shape: Shape::new(std::iter::repeat_n(-1_i64, rank)),
            dimension_symbols: SymbolicDimensions::empty(rank),
        },
    )
}

fn node(
    operator: &str,
    inputs: &[&str],
    output: &str,
    attributes: Vec<Attribute>,
) -> Result<Node, CoreError> {
    Node::new(operator, ONNX_DOMAIN, output, inputs, [output], attributes)
        .map_err(|e| load("cannot build a helper graph node", &e))
}

fn attribute(name: &str, value: i64) -> Result<Attribute, CoreError> {
    Attribute::new(name, value).map_err(|e| load("cannot build a helper graph attribute", &e))
}

impl Helper {
    /// Turn `graph` into a session on the device of `ep`.
    ///
    /// On a device other than the host the session may not place any node
    /// on the CPU: a helper that quietly computed on the host would copy
    /// the cache there and back, which is what this module exists to
    /// avoid. Such a graph fails here, at startup.
    fn new(what: &str, graph: Graph, ep: &ExecutionProvider) -> Result<Self, CoreError> {
        let failed = |e: &dyn std::fmt::Display| load(&format!("cannot build the {what} graph"), e);
        let opset = Opset::new(ONNX_DOMAIN, HELPER_OPSET).map_err(|e| failed(&e))?;
        let mut model = Model::new([opset]).map_err(|e| failed(&e))?;
        model.add_graph(graph).map_err(|e| failed(&e))?;

        // One operation per graph: nothing to optimise or to thread.
        let mut builder = Session::builder()
            .map_err(|e| failed(&e))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| failed(&e))?
            .with_intra_threads(1)
            .map_err(|e| failed(&e))?
            .with_inter_threads(1)
            .map_err(|e| failed(&e))?
            .with_memory_pattern(false)
            .map_err(|e| failed(&e))?;
        let providers = ep.helper_ort_providers()?;
        if !providers.is_empty() {
            builder = builder
                .with_execution_providers(providers)
                .map_err(|e| failed(&e))?
                .with_disable_cpu_fallback()
                .map_err(|e| failed(&e))?;
        }
        let session = model.into_session(&builder).map_err(|e| {
            load(
                &format!(
                    "cannot run the {what} graph on execution provider '{ep}' (set \
                     TEXT_GENERATION_DEVICE_CACHE=off to keep the cache in host memory)"
                ),
                &e,
            )
        })?;
        let binding = Binding::of(&session).map_err(|e| failed(&e))?;
        Ok(Self { session, binding })
    }

    /// `last = Cast(Gather(logits, [-1], axis = 1), float32)`: the
    /// next-token logits of every row, `(rows, 1, vocab)`.
    fn last_position(logits: FloatType, ep: &ExecutionProvider) -> Result<Self, CoreError> {
        let failed = |e: &dyn std::fmt::Display| load("cannot build the last-position graph", e);
        let mut graph = Graph::new().map_err(|e| failed(&e))?;
        graph
            .set_inputs([outlet("logits", logits.element_type(), 3)])
            .map_err(|e| failed(&e))?;
        graph
            .set_outputs([outlet("last", TensorElementType::Float32, 3)])
            .map_err(|e| failed(&e))?;
        let mut index =
            Tensor::<i64>::new(&Allocator::default(), [1_usize]).map_err(|e| failed(&e))?;
        index.extract_tensor_mut().1[0] = -1;
        graph
            .add_initializer("index", index, false)
            .map_err(|e| failed(&e))?;
        let axis = || Ok::<_, CoreError>(vec![attribute("axis", 1)?]);
        match logits {
            FloatType::F32 => {
                graph
                    .add_node(node("Gather", &["logits", "index"], "last", axis()?)?)
                    .map_err(|e| failed(&e))?;
            }
            FloatType::F16 => {
                graph
                    .add_node(node("Gather", &["logits", "index"], "half", axis()?)?)
                    .map_err(|e| failed(&e))?;
                // 1 is TensorProto.FLOAT.
                let to = vec![attribute("to", 1)?];
                graph
                    .add_node(node("Cast", &["half"], "last", to)?)
                    .map_err(|e| failed(&e))?;
            }
        }
        Self::new("last-position", graph, ep)
    }

    /// `out = Where(keep, Gather(Concat(a, b, axis = 0), rows, axis = 0), 0)`:
    /// chosen rows of two caches, with chosen entries zeroed.
    fn cache_rows(cache: FloatType, ep: &ExecutionProvider) -> Result<Self, CoreError> {
        let failed = |e: &dyn std::fmt::Display| load("cannot build the cache-rows graph", e);
        let mut graph = Graph::new().map_err(|e| failed(&e))?;
        graph
            .set_inputs([
                outlet("a", cache.element_type(), 4),
                outlet("b", cache.element_type(), 4),
                outlet("rows", TensorElementType::Int64, 1),
                outlet("keep", TensorElementType::Bool, 4),
            ])
            .map_err(|e| failed(&e))?;
        graph
            .set_outputs([outlet("out", cache.element_type(), 4)])
            .map_err(|e| failed(&e))?;
        let allocator = Allocator::default();
        match cache {
            FloatType::F32 => {
                let mut zero = Tensor::<f32>::new(&allocator, [1_usize]).map_err(|e| failed(&e))?;
                zero.extract_tensor_mut().1[0] = 0.0;
                graph.add_initializer("zero", zero, false)
            }
            FloatType::F16 => {
                let mut zero =
                    Tensor::<half::f16>::new(&allocator, [1_usize]).map_err(|e| failed(&e))?;
                zero.extract_tensor_mut().1[0] = half::f16::ZERO;
                graph.add_initializer("zero", zero, false)
            }
        }
        .map_err(|e| failed(&e))?;
        let axis = || Ok::<_, CoreError>(vec![attribute("axis", 0)?]);
        graph
            .add_node(node("Concat", &["a", "b"], "both", axis()?)?)
            .map_err(|e| failed(&e))?;
        graph
            .add_node(node("Gather", &["both", "rows"], "picked", axis()?)?)
            .map_err(|e| failed(&e))?;
        graph
            .add_node(node(
                "Where",
                &["keep", "picked", "zero"],
                "out",
                Vec::new(),
            )?)
            .map_err(|e| failed(&e))?;
        Self::new("cache-rows", graph, ep)
    }
}

/// Runs a text generation graph with its cache kept on the device; see
/// the module documentation.
pub(super) struct DeviceRun {
    /// Binding of the model session.
    binding: Binding,
    /// Allocates in the model session's arena on the device.
    allocator: Arc<DeviceAllocator>,
    cache_type: FloatType,
    device: ComputeDevice,
    last: Helper,
    rows: Helper,
    /// Whether a forward pass has shown the cache to be on `device`.
    verified: bool,
}

impl DeviceRun {
    /// Prepare bound runs of `session`, which was created for `ep` and
    /// computes on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelLoad`] if a binding or a helper graph
    /// cannot be created on the device.
    pub(super) fn new(
        session: &Session,
        ep: &ExecutionProvider,
        device: ComputeDevice,
        cache: FloatType,
        logits: FloatType,
    ) -> Result<Self, CoreError> {
        Ok(Self {
            binding: Binding::of(session).map_err(|e| load("cannot create an I/O binding", &e))?,
            allocator: Allocator::new(session, device.memory_info()?)
                .map(|allocator| Arc::new(DeviceAllocator(Mutex::new(allocator))))
                .map_err(|e| load("cannot get the session's device allocator", &e))?,
            cache_type: cache,
            device,
            last: Helper::last_position(logits, ep)?,
            rows: Helper::cache_rows(cache, ep)?,
            verified: false,
        })
    }

    /// The device the cache is kept on.
    pub(super) fn device(&self) -> ComputeDevice {
        self.device
    }

    /// Start a forward pass: forget the inputs of the previous one.
    pub(super) fn begin(&mut self) {
        self.binding.get().clear();
    }

    /// Feed `value` to input `name` of the pass started with
    /// [`begin`](Self::begin).
    ///
    /// A value in host memory is copied to the device; one that is
    /// already there is used where it is. A value that borrows host
    /// memory must stay alive until [`run`](Self::run) returns.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if the graph does not take it.
    pub(super) fn feed<T: ValueTypeMarker + ?Sized>(
        &mut self,
        name: &str,
        value: &Value<T>,
    ) -> Result<(), CoreError> {
        self.binding
            .get()
            .bind_input(name, value)
            .map_err(|e| inference(&format!("cannot bind input '{name}'"), &e))
    }

    /// Run the pass.
    ///
    /// Returns the next-token logits of every row as a `float32` host
    /// tensor of shape `(rows, 1, vocab)`, and the cache the pass
    /// produced, in slot order, on the device.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if the graph fails, does not
    /// return an output, or returned its cache somewhere other than on
    /// the device (the session is not computing there).
    pub(super) fn run(
        &mut self,
        session: &mut Session,
        slots: &[PastSlot],
    ) -> Result<(DynValue, Vec<DynValue>), CoreError> {
        let result = self.run_bound(session, slots);
        // The binding must not outlive the host tensors it points at.
        self.binding.get().clear();
        let (logits, present) = result?;
        if !self.verified {
            for value in present.iter().chain([&logits]) {
                let found = value.memory_info().allocation_device();
                if found != self.device.device {
                    return Err(CoreError::Inference(format!(
                        "text generation outputs are in '{}' memory, not on '{}': the session \
                         is not computing on the requested device",
                        found.as_str(),
                        self.device.name(),
                    )));
                }
            }
            self.verified = true;
        }
        let last = self.last_position(&logits)?;
        Ok((last, present))
    }

    fn run_bound(
        &mut self,
        session: &mut Session,
        slots: &[PastSlot],
    ) -> Result<(DynValue, Vec<DynValue>), CoreError> {
        // Outputs are bound anew for every pass: a binding keeps the
        // tensors of the pass before, and their shapes no longer fit.
        let target = self.device.memory_info()?;
        let bind_failed = |name: &str, e: &ort::Error| {
            inference(&format!("cannot bind output '{name}' to the device"), e)
        };
        let binding = self.binding.get();
        binding
            .bind_output_to_device("logits", &target)
            .map_err(|e| bind_failed("logits", &e))?;
        for slot in slots {
            binding
                .bind_output_to_device(slot.output.as_str(), &target)
                .map_err(|e| bind_failed(&slot.output, &e))?;
        }
        let mut outputs = session
            .run_binding(binding)
            .map_err(|e| CoreError::Inference(format!("text generation run failed: {e}")))?;
        let mut take = |name: &str| {
            outputs.remove(name).ok_or_else(|| {
                CoreError::Inference(format!("graph did not return output '{name}'"))
            })
        };
        let logits = take("logits")?;
        let present = slots
            .iter()
            .map(|slot| take(slot.output.as_str()))
            .collect::<Result<Vec<_>, CoreError>>()?;
        Ok((logits, present))
    }

    /// The last position of `logits` (on the device), in host memory.
    fn last_position(&mut self, logits: &DynValue) -> Result<DynValue, CoreError> {
        let Helper { session, binding } = &mut self.last;
        let binding = binding.get();
        let failed = |e: &ort::Error| inference("cannot read the next-token logits", e);
        binding.clear();
        let result = (|| {
            binding.bind_input("logits", logits)?;
            binding.bind_output_to_device("last", &host_memory_info()?)?;
            let mut outputs = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            Ok::<_, ort::Error>(outputs.remove("last"))
        })();
        binding.clear();
        result.map_err(|e| failed(&e))?.ok_or_else(|| {
            CoreError::Inference("the last-position graph returned nothing".to_string())
        })
    }

    /// Chosen rows of one or two cache tensors, as a new tensor on the
    /// device: row `i` of the result is row `rows[i]` of `a` followed by
    /// `b`, and where `keep` (shape `(rows.len(), 1, columns, 1)`, row
    /// major) is `false` the entry is zero instead.
    ///
    /// With a single source `b` is `None`. The graph has two inputs
    /// either way, so `a` is then fed twice: the copy is made on the
    /// device and released when the call returns.
    ///
    /// The result is allocated from the model session's arena and the
    /// graph writes into it, so a moved cache takes the place the cache
    /// before it had; the helper session only holds what it needs while
    /// it runs, and gives that back on its next run. The returned handle
    /// must be kept for as long as the tensor is.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if `a` is not a tensor of four
    /// dimensions or the graph fails.
    pub(super) fn gather_rows(
        &mut self,
        a: &DynValue,
        b: Option<&DynValue>,
        rows: &[i64],
        keep: &[bool],
        columns: usize,
    ) -> Result<(DynValue, Arc<DeviceAllocator>), CoreError> {
        let &[_, heads, _, head_dim] = &a.shape()[..] else {
            return Err(CoreError::Inference(format!(
                "cache tensor has shape {:?}, expected (rows, heads, columns, head_dim)",
                a.shape(),
            )));
        };
        let dims = [rows.len() as i64, heads, columns as i64, head_dim];
        let shrink = self.device.arena_to_shrink();
        let element_type = self.cache_type.element_type();
        let allocator = Arc::clone(&self.allocator);
        let Helper { session, binding } = &mut self.rows;
        let binding = binding.get();
        let failed = |e: &ort::Error| inference("cannot move key/value cache rows", e);
        binding.clear();
        let result = (|| {
            // Give back what the previous move left unused in the helper
            // session's arena.
            let mut options = RunOptions::new()?;
            options.set("memory.enable_memory_arena_shrinkage", shrink.as_str())?;
            let rows_tensor = TensorRef::from_array_view(([rows.len()], rows))?;
            let keep_tensor = TensorRef::from_array_view(([rows.len(), 1, columns, 1], keep))?;
            binding.bind_input("a", a)?;
            binding.bind_input("b", b.unwrap_or(a))?;
            binding.bind_input("rows", &rows_tensor)?;
            binding.bind_input("keep", &keep_tensor)?;
            binding.bind_output("out", allocator.tensor(element_type, &dims)?)?;
            let mut outputs = session.run_binding_with_options(binding, &options)?;
            Ok::<_, ort::Error>(outputs.remove("out"))
        })();
        binding.clear();
        let moved = result.map_err(|e| failed(&e))?.ok_or_else(|| {
            CoreError::Inference("the cache-rows graph returned nothing".to_string())
        })?;
        Ok((moved, allocator))
    }
}

/// Host memory, as the target of an output that the host reads.
fn host_memory_info() -> Result<MemoryInfo<'static>, ort::Error> {
    MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )
}
