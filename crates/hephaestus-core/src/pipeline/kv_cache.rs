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
//!   and cache-columns graphs of [`DeviceRun`]. The host sends row and
//!   column numbers and a mask and never sees the tensors.
//!
//! Both produce the same values: entries are copied or zeroed, nothing is
//! computed. A row stays in its columns; the gathered cache may have more
//! columns than a source (zeros are added on the right) or fewer (the
//! right-most are dropped).

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
///
/// Columns `from..to` of the row are copied, as far as the source and the
/// gathered cache reach; every other entry of the gathered row is zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RowPick {
    /// Row of the source cache.
    pub(super) row: usize,
    /// First column that is copied. The columns before it are the row's
    /// padding in a layout that pads on the left: their cache entries are
    /// whatever the graph computed for padding tokens.
    pub(super) from: usize,
    /// End of the columns that are copied. From there on the source holds
    /// nothing of the row in a layout whose rows start at column 0.
    pub(super) to: usize,
}

impl RowPick {
    /// The row as it is.
    pub(super) fn copy(row: usize) -> Self {
        Self {
            row,
            from: 0,
            to: usize::MAX,
        }
    }

    /// The row without its first `padding` columns.
    pub(super) fn after_padding(row: usize, padding: usize) -> Self {
        Self {
            row,
            from: padding,
            to: usize::MAX,
        }
    }

    /// The first `length` columns of the row.
    pub(super) fn first(row: usize, length: usize) -> Self {
        Self {
            row,
            from: 0,
            to: length,
        }
    }

    /// The columns to copy from a source `source` columns wide into a
    /// cache `columns` wide; empty as `(0, 0)`.
    fn range(self, source: usize, columns: usize) -> (usize, usize) {
        let to = self.to.min(source).min(columns);
        if self.from < to {
            (self.from, to)
        } else {
            (0, 0)
        }
    }

    /// Whether the pick copies all of a row `columns` wide.
    pub(super) fn is_whole(self, columns: usize) -> bool {
        self.from == 0 && self.to >= columns
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

    /// Positions per sequence.
    pub(super) fn columns(&self) -> usize {
        self.columns
    }

    /// The tensors to feed, in slot order.
    pub(super) fn tensors(&self) -> &[DynValue] {
        &self.tensors
    }

    /// A new cache `columns` wide made of chosen rows of other caches,
    /// in the order given. The caches live in `home`, and so does the
    /// result.
    ///
    /// This is the only place a generation's cache moves: to another row
    /// when a neighbour leaves, or next to others when it joins. An entry
    /// never moves to another column: a source narrower than `columns`
    /// gets zeros on the right, a wider one loses its right-most columns.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if `parts` picks no row, the
    /// caches differ in their tensors, a pick is out of range, a host
    /// tensor is not of `home`'s float type and of shape `(rows, heads,
    /// columns, head_dim)`, or (on a device) a helper graph fails.
    pub(super) fn gather(
        parts: &[(&KvCache, &[RowPick])],
        columns: usize,
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
        let slots = first.tensors.len();
        for (cache, picks) in &parts {
            if cache.tensors.len() != slots {
                return Err(CoreError::Inference(
                    "key/value caches differ in their tensors".to_string(),
                ));
            }
            if let Some(pick) = picks.iter().find(|pick| pick.row >= cache.rows) {
                return Err(CoreError::Inference(format!(
                    "cache row {} is outside a cache of {} rows",
                    pick.row, cache.rows,
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
                || *source_columns != cache.columns
                || *heads_and_size.get_or_insert((*heads, *head_dim)) != (*heads, *head_dim)
            {
                return Err(CoreError::Inference(format!(
                    "cache tensor has shape {shape:?}, expected {} rows and {} columns",
                    cache.rows, cache.columns,
                )));
            }
            let source_head_len = source_columns * head_dim;
            let head_len = columns * head_dim;
            for pick in *picks {
                let start = gathered.len();
                gathered.resize(start + heads * head_len, T::default());
                let (from, to) = pick.range(*source_columns, columns);
                for head in 0..*heads {
                    let source = (pick.row * heads + head) * source_head_len;
                    let target = start + head * head_len;
                    gathered[target + from * head_dim..target + to * head_dim]
                        .copy_from_slice(&data[source + from * head_dim..source + to * head_dim]);
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
///
/// A source that is not `columns` wide is first copied to that width
/// (cache-columns graph). The cache-rows graph then takes two caches at a
/// time, so more than two are merged one after another.
fn gather_on_device(
    parts: &[(&KvCache, &[RowPick])],
    columns: usize,
    device: &mut DeviceRun,
) -> Result<(Vec<DynValue>, Option<Arc<DeviceAllocator>>), CoreError> {
    // Which entries of a gathered row are copied, not zero.
    let keep_of = |cache: &KvCache, picks: &[RowPick], keep: &mut Vec<bool>| {
        for pick in picks {
            let (from, to) = pick.range(cache.columns, columns);
            keep.extend((0..columns).map(|column| from <= column && column < to));
        }
    };
    let index = |row: usize| {
        i64::try_from(row)
            .map_err(|_| CoreError::Inference(format!("cache row {row} does not fit an index")))
    };
    let slots = parts.first().map_or(0, |(cache, _)| cache.tensors.len());
    let mut tensors = Vec::with_capacity(slots);
    let mut allocator = None;
    for slot in 0..slots {
        // The rows gathered so far, on the device.
        let mut merged: Option<(DynValue, Arc<DeviceAllocator>, usize)> = None;
        let mut rest = parts.iter();
        while let Some((cache, picks)) = rest.next() {
            // The source at the width of the result; its columns beyond
            // the source's own are zeroed through `keep`.
            let widened = if cache.columns == columns {
                None
            } else {
                Some(device.gather_columns(&cache.tensors[slot], cache.columns, columns)?)
            };
            let source = widened
                .as_ref()
                .map_or(&cache.tensors[slot], |(tensor, _)| tensor);
            let mut rows = Vec::new();
            let mut keep = Vec::new();
            let next = match &merged {
                // Rows of the source follow the rows gathered before.
                Some((before, _, count)) => {
                    for row in 0..*count {
                        rows.push(index(row)?);
                    }
                    keep.resize(count * columns, true);
                    for pick in *picks {
                        rows.push(index(count + pick.row)?);
                    }
                    keep_of(cache, picks, &mut keep);
                    let (tensor, owner) =
                        device.gather_rows(before, Some(source), &rows, &keep, columns)?;
                    (tensor, owner, count + picks.len())
                }
                None => {
                    // The first source, and with it the second when it
                    // needs no widening: one pass for the usual join.
                    let second = rest
                        .clone()
                        .next()
                        .filter(|(second, _)| second.columns == columns);
                    for pick in *picks {
                        rows.push(index(pick.row)?);
                    }
                    keep_of(cache, picks, &mut keep);
                    let mut count = picks.len();
                    if let Some((second, second_picks)) = second {
                        rest.next();
                        for pick in *second_picks {
                            rows.push(index(cache.rows + pick.row)?);
                        }
                        keep_of(second, second_picks, &mut keep);
                        count += second_picks.len();
                    }
                    let (tensor, owner) = device.gather_rows(
                        source,
                        second.map(|(second, _)| &second.tensors[slot]),
                        &rows,
                        &keep,
                        columns,
                    )?;
                    (tensor, owner, count)
                }
            };
            merged = Some(next);
        }
        if let Some((tensor, owner, _)) = merged {
            tensors.push(tensor);
            allocator = Some(owner);
        }
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
        let kept = KvCache::gather(&[(&cache, &[RowPick::copy(2), RowPick::copy(0)])], 2, HOST)
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
        let picks = [RowPick::after_padding(1, 2)];

        // Act
        let merged = KvCache::gather(
            &[(&running, &[RowPick::copy(0)]), (&joining, &picks)],
            3,
            HOST,
        )
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
    fn gather_refuses_rows_that_are_not_there() {
        let narrow = numbered(1, 2, 2, 1);

        assert!(KvCache::gather(&[(&narrow, &[RowPick::copy(1)])], 2, HOST).is_err());
        assert!(KvCache::gather(&[(&narrow, &[])], 2, HOST).is_err());
    }

    #[test]
    fn gather_brings_caches_of_different_widths_to_one_width() {
        // Arrange -- rows that start at column 0: one 2 columns wide, one
        // 4 wide of which the row has used 3.
        let narrow = numbered(1, 2, 2, 1);
        let wide = numbered(2, 2, 4, 1);
        let parts: [(&KvCache, &[RowPick]); 2] = [
            (&narrow, &[RowPick::copy(0)]),
            (&wide, &[RowPick::first(1, 3)]),
        ];

        // Act -- to the width of the longer row, and to less than that.
        let widened = KvCache::gather(&parts, 3, HOST).expect("widen");
        let cut = KvCache::gather(&parts, 1, HOST).expect("cut");

        // Assert -- entries stay in their columns; zeros on the right.
        assert_eq!(widened.columns(), 3);
        assert_eq!(
            contents(&widened),
            (
                vec![2, 2, 3, 1],
                vec![
                    0.0, 10.0, 0.0, 100.0, 110.0, 0.0, //
                    1000.0, 1010.0, 1020.0, 1100.0, 1110.0, 1120.0,
                ]
            )
        );
        assert_eq!(
            contents(&cut),
            (vec![2, 2, 1, 1], vec![0.0, 100.0, 1000.0, 1100.0])
        );
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
        let picks = [RowPick::after_padding(1, 2), RowPick::copy(0)];
        let expected = {
            let running = numbered_as(FloatType::F32, 3, 3);
            let joining = numbered_as(FloatType::F32, 2, 3);
            let keep = [RowPick::copy(2), RowPick::copy(0)];
            let (shape, data) = values(
                &KvCache::gather(&[(&running, &keep), (&joining, &picks)], 3, HOST).expect("f32"),
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
            3,
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
        assert!(KvCache::gather(&[(&running, &keep)], 3, HOST).is_err());
        let full = numbered_as(FloatType::F32, 3, 3);
        assert!(KvCache::gather(&[(&full, &keep)], 3, CacheHome::Host(FloatType::F16)).is_err());
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
        // (rows of a cache 4 wide, rows of a second cache, its width,
        // width of the result).
        type Move<'a> = (&'a [RowPick], Option<&'a [RowPick]>, usize, usize);
        let moves: [Move<'_>; 10] = [
            // Rows leave: from the end, the middle, the front.
            (&[RowPick::copy(0), RowPick::copy(1)], None, 4, 4),
            (&[RowPick::copy(0), RowPick::copy(2)], None, 4, 4),
            (&[RowPick::copy(2)], None, 4, 4),
            // Rows join an empty batch, one of them padded.
            (
                &[RowPick::copy(1), RowPick::after_padding(0, 3)],
                None,
                4,
                4,
            ),
            // Rows join running ones: padded, unpadded, all padding.
            (
                &[RowPick::copy(0), RowPick::copy(1), RowPick::copy(2)],
                Some(&[RowPick::after_padding(1, 2), RowPick::copy(0)]),
                4,
                4,
            ),
            (
                &[RowPick::copy(1)],
                Some(&[RowPick::after_padding(0, 4)]),
                4,
                4,
            ),
            // Rows that start at column 0. The longest leaves and the
            // cache is cut back to the rows that stay.
            (&[RowPick::first(2, 2), RowPick::first(0, 3)], None, 4, 3),
            // A shorter row joins: its cache is widened.
            (
                &[RowPick::first(0, 4), RowPick::first(1, 2)],
                Some(&[RowPick::first(1, 2)]),
                2,
                4,
            ),
            // A longer row joins: the running cache is widened.
            (
                &[RowPick::first(0, 4), RowPick::first(2, 3)],
                Some(&[RowPick::first(0, 6), RowPick::first(1, 5)]),
                6,
                6,
            ),
            // Both at once: one is cut, the other widened.
            (&[RowPick::first(1, 4)], Some(&[RowPick::first(0, 1)]), 1, 3),
        ];

        for float in [FloatType::F32, FloatType::F16] {
            let mut run =
                DeviceRun::new(&session, &cpu, device, float, FloatType::F32).expect("device run");
            let first = numbered_as(float, 3, 4);
            for (picks, joining, second_width, columns) in moves {
                let second = numbered_as(float, 2, second_width);
                let mut parts: Vec<(&KvCache, &[RowPick])> = vec![(&first, picks)];
                parts.extend(joining.map(|joining| (&second, joining)));

                // Act
                let on_host =
                    KvCache::gather(&parts, columns, CacheHome::Host(float)).expect("host");
                let on_device =
                    KvCache::gather(&parts, columns, CacheHome::Device(&mut run)).expect("device");

                // Assert
                assert_eq!(on_device.rows(), on_host.rows());
                assert_eq!(on_device.tensors[0].data_type(), &float.element_type());
                assert_eq!(
                    values(&on_device),
                    values(&on_host),
                    "{float:?} {picks:?} {joining:?}"
                );
                assert_eq!(on_device.columns(), columns);
            }

            // Several requests join at once, each read alone: more than
            // two caches, of three widths.
            let second = numbered_as(float, 2, 2);
            let third = numbered_as(float, 1, 5);
            let several: [(&KvCache, &[RowPick]); 4] = [
                (&first, &[RowPick::first(2, 4), RowPick::first(0, 3)]),
                (&second, &[RowPick::first(1, 2)]),
                (&third, &[RowPick::first(0, 5)]),
                (&second, &[RowPick::first(0, 2)]),
            ];
            let on_host = KvCache::gather(&several, 5, CacheHome::Host(float)).expect("host");
            let on_device =
                KvCache::gather(&several, 5, CacheHome::Device(&mut run)).expect("device");
            assert_eq!(on_device.rows(), 5);
            assert_eq!(values(&on_device), values(&on_host), "{float:?} several");
        }
    }
}
