//! The key/value cache of a batch of running generations.
//!
//! [`KvCache`] is everything the batched decode loop knows about cache
//! tensors: it takes them from a forward pass, feeds them to the next one
//! untouched, and builds a new cache from chosen rows of others when a
//! generation joins or leaves the batch ([`KvCache::gather`]).
//!
//! Only `gather` moves cache contents, and it does so where the cache
//! lives ([`CacheHome`]):
//!
//! - a cache in host memory is copied by the host, as `float32` or
//!   `float16`;
//! - a cache on the session's device is copied there, by the cache-rows
//!   graph of [`DeviceRun`]. The host sends row numbers and a mask and
//!   never sees the tensors.
//!
//! Both produce the same values: rows are copied, nothing is computed.

use std::fmt::Debug;

use ort::session::SessionOutputs;
use ort::value::{DynValue, PrimitiveTensorElementType, Tensor};

use crate::error::CoreError;

use std::sync::Arc;

use super::device_run::{DeviceAllocator, DeviceRun};
use super::float::FloatType;
use super::text_generation::PastSlot;

/// Where cache tensors live, and so who copies their rows.
pub(super) enum CacheHome<'a> {
    /// Host memory, holding this float type.
    Host(FloatType),
    /// The device of a bound run.
    Device(&'a mut DeviceRun),
}

/// One row to copy into a gathered cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RowPick {
    /// Row of the source cache.
    pub(super) row: usize,
    /// Leading columns to write as zeros instead of copying: the row's
    /// padding, whose cache entries are whatever the graph computed for
    /// padding tokens.
    pub(super) zero_columns: usize,
}

impl RowPick {
    /// The row as it is.
    pub(super) fn copy(row: usize) -> Self {
        Self {
            row,
            zero_columns: 0,
        }
    }
}

/// Cache tensors of `rows` sequences over `columns` positions.
///
/// One tensor per cache slot, in slot order, each of shape
/// `(rows, heads, columns, head_dim)`.
pub(super) struct KvCache {
    // Declared before `_allocator`: the tensors are released first.
    tensors: Vec<DynValue>,
    rows: usize,
    columns: usize,
    /// The device allocator the tensors were created from, if they were
    /// moved on a device; it has to outlive them.
    _allocator: Option<Arc<DeviceAllocator>>,
}

impl KvCache {
    /// The cache fed before anything is cached: `rows` rows, no columns.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if a tensor cannot be created.
    pub(super) fn empty(
        slots: &[PastSlot],
        rows: usize,
        float: FloatType,
    ) -> Result<Self, CoreError> {
        let tensors = slots
            .iter()
            .map(|slot| {
                let [_, heads, _, head_dim] = slot.empty_dims;
                float.zeros(&[rows, heads, 0, head_dim])
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        Ok(Self {
            tensors,
            rows,
            columns: 0,
            _allocator: None,
        })
    }

    /// Take the cache a forward pass returned for `rows` rows, now
    /// covering `columns` positions.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if the graph did not return one of
    /// the cache outputs.
    pub(super) fn from_outputs(
        outputs: &mut SessionOutputs<'_>,
        slots: &[PastSlot],
        rows: usize,
        columns: usize,
    ) -> Result<Self, CoreError> {
        let tensors = slots
            .iter()
            .map(|slot| {
                outputs.remove(slot.output.as_str()).ok_or_else(|| {
                    CoreError::Inference(format!("graph did not return output '{}'", slot.output))
                })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        Ok(Self {
            tensors,
            rows,
            columns,
            _allocator: None,
        })
    }

    /// The cache a bound forward pass returned for `rows` rows, now
    /// covering `columns` positions: one tensor per slot, in slot order.
    pub(super) fn from_tensors(tensors: Vec<DynValue>, rows: usize, columns: usize) -> Self {
        Self {
            tensors,
            rows,
            columns,
            _allocator: None,
        }
    }

    /// Sequences in the cache.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// The tensors to feed, in slot order.
    pub(super) fn tensors(&self) -> &[DynValue] {
        &self.tensors
    }

    /// A new cache made of chosen rows of caches that cover the same
    /// columns, in the order given. The caches live in `home`, and so
    /// does the result.
    ///
    /// This is the only place a generation's cache moves: to another row
    /// when a neighbour leaves, or next to others when it joins. Columns
    /// never move.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if `parts` picks no row, the
    /// caches differ in shape, a pick is out of range, a host tensor is
    /// not of `home`'s float type and of shape `(rows, heads, columns,
    /// head_dim)`, or (on a device) more than two caches are given or the
    /// cache-rows graph fails.
    pub(super) fn gather(
        parts: &[(&KvCache, &[RowPick])],
        home: CacheHome<'_>,
    ) -> Result<Self, CoreError> {
        let rows: usize = parts.iter().map(|(_, picks)| picks.len()).sum();
        let parts: Vec<(&KvCache, &[RowPick])> = parts
            .iter()
            .copied()
            .filter(|(_, picks)| !picks.is_empty())
            .collect();
        let Some((first, _)) = parts.first() else {
            return Err(CoreError::Inference(
                "cannot gather a key/value cache of no rows".to_string(),
            ));
        };
        let columns = first.columns;
        let slots = first.tensors.len();
        for (cache, picks) in &parts {
            if cache.tensors.len() != slots {
                return Err(CoreError::Inference(
                    "key/value caches differ in their tensors".to_string(),
                ));
            }
            if cache.columns != columns {
                return Err(CoreError::Inference(format!(
                    "cannot gather caches of {} and {columns} columns",
                    cache.columns,
                )));
            }
            if let Some(pick) = picks
                .iter()
                .find(|pick| pick.row >= cache.rows || pick.zero_columns > columns)
            {
                return Err(CoreError::Inference(format!(
                    "cache row {} with {} padded columns is outside a cache of {} rows \
                     and {columns} columns",
                    pick.row, pick.zero_columns, cache.rows,
                )));
            }
        }
        let (tensors, allocator) = match home {
            CacheHome::Host(FloatType::F32) => {
                (gather_on_host::<f32>(&parts, rows, columns)?, None)
            }
            CacheHome::Host(FloatType::F16) => {
                (gather_on_host::<half::f16>(&parts, rows, columns)?, None)
            }
            CacheHome::Device(device) => gather_on_device(&parts, columns, device)?,
        };
        Ok(Self {
            tensors,
            rows,
            columns,
            _allocator: allocator,
        })
    }
}

/// [`KvCache::gather`] for tensors of element type `T` in host memory.
/// `parts` has been checked: no empty part, picks in range.
fn gather_on_host<T>(
    parts: &[(&KvCache, &[RowPick])],
    rows: usize,
    columns: usize,
) -> Result<Vec<DynValue>, CoreError>
where
    T: PrimitiveTensorElementType + Copy + Default + Debug + 'static,
{
    let slots = parts.first().map_or(0, |(cache, _)| cache.tensors.len());
    let mut tensors = Vec::with_capacity(slots);
    for slot in 0..slots {
        let mut gathered: Vec<T> = Vec::new();
        let mut heads_and_size = None;
        for (cache, picks) in parts {
            let (shape, data) = cache.tensors[slot]
                .try_extract_tensor::<T>()
                .map_err(|e| CoreError::Inference(format!("cannot read cache tensor: {e}")))?;
            let dims: Vec<usize> = shape
                .iter()
                .map(|&dim| usize::try_from(dim).unwrap_or(usize::MAX))
                .collect();
            let [source_rows, heads, source_columns, head_dim] = dims.as_slice() else {
                return Err(CoreError::Inference(format!(
                    "cache tensor has shape {shape:?}, expected (rows, heads, columns, head_dim)",
                )));
            };
            if *source_rows != cache.rows
                || *source_columns != columns
                || *heads_and_size.get_or_insert((*heads, *head_dim)) != (*heads, *head_dim)
            {
                return Err(CoreError::Inference(format!(
                    "cache tensor has shape {shape:?}, expected {} rows and {columns} columns",
                    cache.rows,
                )));
            }
            let head_len = columns * head_dim;
            let row_len = heads * head_len;
            gathered.reserve(picks.len() * row_len);
            for pick in *picks {
                let start = gathered.len();
                gathered.extend_from_slice(&data[pick.row * row_len..(pick.row + 1) * row_len]);
                if pick.zero_columns > 0 {
                    for head in 0..*heads {
                        let from = start + head * head_len;
                        gathered[from..from + pick.zero_columns * head_dim].fill(T::default());
                    }
                }
            }
        }
        let (heads, head_dim) = heads_and_size.unwrap_or((0, 0));
        tensors.push(
            Tensor::<T>::from_array(([rows, heads, columns, head_dim], gathered))
                .map(Tensor::into_dyn)
                .map_err(|e| {
                    CoreError::Inference(format!("cannot create a tensor for gathered cache: {e}"))
                })?,
        );
    }
    Ok(tensors)
}

/// [`KvCache::gather`] for tensors on the device of `device`. `parts`
/// has been checked: no empty part, picks in range.
fn gather_on_device(
    parts: &[(&KvCache, &[RowPick])],
    columns: usize,
    device: &mut DeviceRun,
) -> Result<(Vec<DynValue>, Option<Arc<DeviceAllocator>>), CoreError> {
    let (first, second) = match parts {
        [first] => (first, None),
        [first, second] => (first, Some(second)),
        _ => {
            return Err(CoreError::Inference(format!(
                "cannot gather {} key/value caches on a device at once",
                parts.len(),
            )));
        }
    };
    // Rows of the second cache follow those of the first.
    let mut rows = Vec::new();
    let mut keep = Vec::new();
    for (offset, (_, picks)) in [(0, first)]
        .into_iter()
        .chain(second.map(|second| (first.0.rows, second)))
    {
        for pick in *picks {
            rows.push(i64::try_from(offset + pick.row).map_err(|_| {
                CoreError::Inference(format!("cache row {} does not fit an index", pick.row))
            })?);
            keep.extend(std::iter::repeat_n(false, pick.zero_columns));
            keep.extend(std::iter::repeat_n(true, columns - pick.zero_columns));
        }
    }
    let mut tensors = Vec::with_capacity(first.0.tensors.len());
    let mut allocator = None;
    for slot in 0..first.0.tensors.len() {
        let (tensor, owner) = device.gather_rows(
            &first.0.tensors[slot],
            second.map(|(cache, _)| &cache.tensors[slot]),
            &rows,
            &keep,
            columns,
        )?;
        tensors.push(tensor);
        allocator = Some(owner);
    }
    Ok((tensors, allocator))
}

/// An owned `float32` tensor of shape `dims` as a dynamic value.
#[cfg(test)]
fn tensor(dims: [usize; 4], data: Vec<f32>, what: &str) -> Result<DynValue, CoreError> {
    Tensor::<f32>::from_array((dims, data))
        .map(Tensor::into_dyn)
        .map_err(|e| CoreError::Inference(format!("cannot create a tensor for {what}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOST: CacheHome<'static> = CacheHome::Host(FloatType::F32);

    /// A one-slot cache whose entry at (row, head, column, i) is
    /// `1000 * row + 100 * head + 10 * column + i`.
    fn numbered(rows: usize, heads: usize, columns: usize, head_dim: usize) -> KvCache {
        let mut data = Vec::new();
        for row in 0..rows {
            for head in 0..heads {
                for column in 0..columns {
                    for i in 0..head_dim {
                        data.push((1000 * row + 100 * head + 10 * column + i) as f32);
                    }
                }
            }
        }
        KvCache {
            tensors: vec![tensor([rows, heads, columns, head_dim], data, "test").expect("tensor")],
            rows,
            columns,
            _allocator: None,
        }
    }

    fn contents(cache: &KvCache) -> (Vec<i64>, Vec<f32>) {
        let (shape, data) = cache.tensors[0].try_extract_tensor::<f32>().expect("f32");
        (shape.iter().copied().collect(), data.to_vec())
    }

    #[test]
    fn gather_keeps_rows_whole_and_in_the_order_picked() {
        // Arrange
        let cache = numbered(3, 2, 2, 2);

        // Act -- rows 2 and 0: row 1 leaves.
        let kept = KvCache::gather(&[(&cache, &[RowPick::copy(2), RowPick::copy(0)])], HOST)
            .expect("gather");

        // Assert
        let (shape, data) = contents(&kept);
        assert_eq!(shape, vec![2, 2, 2, 2]);
        assert_eq!(kept.rows(), 2);
        assert_eq!(
            data,
            vec![
                2000.0, 2001.0, 2010.0, 2011.0, 2100.0, 2101.0, 2110.0, 2111.0, //
                0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0,
            ]
        );
    }

    #[test]
    fn gather_joins_caches_and_zeroes_only_the_padding_of_a_row() {
        // Arrange
        let running = numbered(1, 2, 3, 1);
        let joining = numbered(2, 2, 3, 1);
        let picks = [RowPick {
            row: 1,
            zero_columns: 2,
        }];

        // Act
        let merged = KvCache::gather(&[(&running, &[RowPick::copy(0)]), (&joining, &picks)], HOST)
            .expect("gather");

        // Assert -- the joining row keeps its last column in both heads.
        let (shape, data) = contents(&merged);
        assert_eq!(shape, vec![2, 2, 3, 1]);
        assert_eq!(
            data,
            vec![
                0.0, 10.0, 20.0, 100.0, 110.0, 120.0, 0.0, 0.0, 1020.0, 0.0, 0.0, 1120.0
            ]
        );
    }

    #[test]
    fn gather_refuses_caches_that_do_not_fit_together() {
        let narrow = numbered(1, 2, 2, 1);
        let wide = numbered(1, 2, 3, 1);
        let row = [RowPick::copy(0)];

        assert!(KvCache::gather(&[(&narrow, &row), (&wide, &row)], HOST).is_err());
        assert!(KvCache::gather(&[(&narrow, &[RowPick::copy(1)])], HOST).is_err());
        assert!(KvCache::gather(&[(&narrow, &[])], HOST).is_err());
    }

    /// `numbered`, in the given float type.
    fn numbered_as(float: FloatType, rows: usize, columns: usize) -> KvCache {
        let cache = numbered(rows, 2, columns, 3);
        if float == FloatType::F32 {
            return cache;
        }
        let (_, data) = contents(&cache);
        let half: Vec<half::f16> = data.into_iter().map(half::f16::from_f32).collect();
        let tensor = Tensor::<half::f16>::from_array(([rows, 2, columns, 3], half))
            .expect("tensor")
            .into_dyn();
        KvCache::from_tensors(vec![tensor], rows, columns)
    }

    /// Shape and values of a cache tensor, widened to float32.
    fn values(cache: &KvCache) -> (Vec<i64>, Vec<f32>) {
        let (shape, data) = super::super::float::float_tensor(&cache.tensors[0]).expect("float");
        let mut widened = Vec::new();
        let all = data.widen(0..data.len(), &mut widened).to_vec();
        (shape, all)
    }

    #[test]
    fn gather_copies_float16_caches_like_float32_ones() {
        // Arrange -- what float32 gathers, rounded to float16 (which
        // holds whole numbers exactly only up to 2048).
        let picks = [
            RowPick {
                row: 1,
                zero_columns: 2,
            },
            RowPick::copy(0),
        ];
        let expected = {
            let running = numbered_as(FloatType::F32, 3, 3);
            let joining = numbered_as(FloatType::F32, 2, 3);
            let keep = [RowPick::copy(2), RowPick::copy(0)];
            let (shape, data) = values(
                &KvCache::gather(&[(&running, &keep), (&joining, &picks)], HOST).expect("f32"),
            );
            let rounded = data
                .into_iter()
                .map(|value| half::f16::from_f32(value).to_f32())
                .collect::<Vec<f32>>();
            (shape, rounded)
        };
        let running = numbered_as(FloatType::F16, 3, 3);
        let joining = numbered_as(FloatType::F16, 2, 3);
        let keep = [RowPick::copy(2), RowPick::copy(0)];

        // Act
        let merged = KvCache::gather(
            &[(&running, &keep), (&joining, &picks)],
            CacheHome::Host(FloatType::F16),
        )
        .expect("f16");

        // Assert
        assert_eq!(
            merged.tensors[0].data_type(),
            &ort::value::TensorElementType::Float16
        );
        assert_eq!(values(&merged), expected);
        assert_eq!(expected.0, vec![4, 2, 3, 3]);
        // A float16 cache is not read as float32, nor the other way round.
        assert!(KvCache::gather(&[(&running, &keep)], HOST).is_err());
        let full = numbered_as(FloatType::F32, 3, 3);
        assert!(KvCache::gather(&[(&full, &keep)], CacheHome::Host(FloatType::F16)).is_err());
    }

    #[test]
    fn gather_on_the_device_gives_what_the_host_gathers() {
        use crate::ep::ExecutionProvider;
        use crate::pipeline::device_run::ComputeDevice;

        // Arrange -- any session will do for the binding; the cache-rows
        // graph is DeviceRun's own. On the CPU the device is host memory.
        let graph = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/text_generation/model_batch.onnx");
        let session = ort::session::Session::builder()
            .expect("builder")
            .commit_from_file(graph)
            .expect("session");
        let cpu = ExecutionProvider::Cpu;
        let device = ComputeDevice::of(&cpu).expect("the CPU is a device");
        let moves: [(&[RowPick], Option<&[RowPick]>); 6] = [
            // Rows leave: from the end, the middle, the front.
            (&[RowPick::copy(0), RowPick::copy(1)], None),
            (&[RowPick::copy(0), RowPick::copy(2)], None),
            (&[RowPick::copy(2)], None),
            // Rows join an empty batch, one of them padded.
            (
                &[
                    RowPick::copy(1),
                    RowPick {
                        row: 0,
                        zero_columns: 3,
                    },
                ],
                None,
            ),
            // Rows join running ones: padded, unpadded, all padding.
            (
                &[RowPick::copy(0), RowPick::copy(1), RowPick::copy(2)],
                Some(&[
                    RowPick {
                        row: 1,
                        zero_columns: 2,
                    },
                    RowPick::copy(0),
                ]),
            ),
            (
                &[RowPick::copy(1)],
                Some(&[RowPick {
                    row: 0,
                    zero_columns: 4,
                }]),
            ),
        ];

        for float in [FloatType::F32, FloatType::F16] {
            let mut run =
                DeviceRun::new(&session, &cpu, device, float, FloatType::F32).expect("device run");
            let first = numbered_as(float, 3, 4);
            let second = numbered_as(float, 2, 4);
            for (picks, joining) in moves {
                let mut parts: Vec<(&KvCache, &[RowPick])> = vec![(&first, picks)];
                parts.extend(joining.map(|joining| (&second, joining)));

                // Act
                let on_host = KvCache::gather(&parts, CacheHome::Host(float)).expect("host");
                let on_device =
                    KvCache::gather(&parts, CacheHome::Device(&mut run)).expect("device");

                // Assert
                assert_eq!(on_device.rows(), on_host.rows());
                assert_eq!(on_device.tensors[0].data_type(), &float.element_type());
                assert_eq!(
                    values(&on_device),
                    values(&on_host),
                    "{float:?} {picks:?} {joining:?}"
                );
            }

            // More than two caches at once is not something a batch does.
            let row = [RowPick::copy(0)];
            let three: [(&KvCache, &[RowPick]); 3] =
                [(&first, &row), (&second, &row), (&first, &row)];
            assert!(KvCache::gather(&three, CacheHome::Device(&mut run)).is_err());
            assert!(KvCache::gather(&three, CacheHome::Host(float)).is_ok());
        }
    }
}
