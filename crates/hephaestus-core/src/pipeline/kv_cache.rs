//! The key/value cache of a batch of running generations.
//!
//! [`KvCache`] is everything the batched decode loop knows about cache
//! tensors: it takes them from a forward pass, feeds them to the next one
//! untouched, and builds a new cache from chosen rows of others when a
//! generation joins or leaves the batch ([`KvCache::gather`]).
//!
//! Only `gather` looks inside the tensors. It copies through host memory
//! and reads `float32`; a cache that stays on a device, or holds another
//! element type, needs a different `gather` and nothing else.

use ort::session::SessionOutputs;
use ort::value::{DynValue, Tensor};

use crate::error::CoreError;

use super::text_generation::PastSlot;

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
    tensors: Vec<DynValue>,
    rows: usize,
    columns: usize,
}

impl KvCache {
    /// The cache fed before anything is cached: `rows` rows, no columns.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if a tensor cannot be created.
    pub(super) fn empty(slots: &[PastSlot], rows: usize) -> Result<Self, CoreError> {
        let tensors = slots
            .iter()
            .map(|slot| {
                let [_, heads, _, head_dim] = slot.empty_dims;
                tensor([rows, heads, 0, head_dim], Vec::new(), &slot.input)
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        Ok(Self {
            tensors,
            rows,
            columns: 0,
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
        })
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
    /// columns, in the order given.
    ///
    /// This is the only place a generation's cache moves: to another row
    /// when a neighbour leaves, or next to others when it joins. Columns
    /// never move.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if `parts` picks no row, the
    /// caches differ in shape, a pick is out of range, or a tensor is not
    /// `float32` of shape `(rows, heads, columns, head_dim)`.
    pub(super) fn gather(parts: &[(&KvCache, &[RowPick])]) -> Result<Self, CoreError> {
        let rows: usize = parts.iter().map(|(_, picks)| picks.len()).sum();
        let Some((first, _)) = parts.iter().find(|(_, picks)| !picks.is_empty()) else {
            return Err(CoreError::Inference(
                "cannot gather a key/value cache of no rows".to_string(),
            ));
        };
        let columns = first.columns;
        let slots = first.tensors.len();
        let mut tensors = Vec::with_capacity(slots);
        for slot in 0..slots {
            let mut gathered: Vec<f32> = Vec::new();
            let mut heads_and_size = None;
            for (cache, picks) in parts {
                if picks.is_empty() {
                    continue;
                }
                let value = cache.tensors.get(slot).ok_or_else(|| {
                    CoreError::Inference("key/value caches differ in their tensors".to_string())
                })?;
                let (shape, data) = value
                    .try_extract_tensor::<f32>()
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
                    if pick.row >= cache.rows || pick.zero_columns > columns {
                        return Err(CoreError::Inference(format!(
                            "cache row {} with {} padded columns is outside a cache of {} rows \
                             and {columns} columns",
                            pick.row, pick.zero_columns, cache.rows,
                        )));
                    }
                    let start = gathered.len();
                    gathered.extend_from_slice(&data[pick.row * row_len..(pick.row + 1) * row_len]);
                    if pick.zero_columns > 0 {
                        for head in 0..*heads {
                            let from = start + head * head_len;
                            gathered[from..from + pick.zero_columns * head_dim].fill(0.0);
                        }
                    }
                }
            }
            let (heads, head_dim) = heads_and_size.unwrap_or((0, 0));
            tensors.push(tensor(
                [rows, heads, columns, head_dim],
                gathered,
                "gathered cache",
            )?);
        }
        Ok(Self {
            tensors,
            rows,
            columns,
        })
    }
}

/// An owned `float32` tensor of shape `dims` as a dynamic value.
fn tensor(dims: [usize; 4], data: Vec<f32>, what: &str) -> Result<DynValue, CoreError> {
    Tensor::<f32>::from_array((dims, data))
        .map(Tensor::into_dyn)
        .map_err(|e| CoreError::Inference(format!("cannot create a tensor for {what}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let kept =
            KvCache::gather(&[(&cache, &[RowPick::copy(2), RowPick::copy(0)])]).expect("gather");

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
        let merged = KvCache::gather(&[(&running, &[RowPick::copy(0)]), (&joining, &picks)])
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

        assert!(KvCache::gather(&[(&narrow, &row), (&wide, &row)]).is_err());
        assert!(KvCache::gather(&[(&narrow, &[RowPick::copy(1)])]).is_err());
        assert!(KvCache::gather(&[(&narrow, &[])]).is_err());
    }
}
