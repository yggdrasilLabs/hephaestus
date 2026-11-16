//! The two float element types a generation graph may use.
//!
//! Full-precision exports compute in `float32`; half-precision ones
//! (`model_fp16.onnx`, `model_q4f16.onnx`) declare `float16` cache tensors
//! and often `float16` logits. Everything that builds or reads such a
//! tensor on the host goes through this module, so the decode loops do
//! not care which of the two a graph uses. Tokens are always picked from
//! `float32` logits: `float16` ones are widened first, which is exact.

use std::ops::Range;

use half::f16;
use half::slice::HalfFloatSliceExt;
use ort::value::{DynValue, Tensor, TensorElementType};

use crate::error::CoreError;

/// Element type of a cache or logits tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FloatType {
    /// `float32`.
    F32,
    /// `float16`.
    F16,
}

impl FloatType {
    /// The float type of a declared element type; `None` if it is not one
    /// this crate can generate with.
    pub(super) fn of(element_type: Option<TensorElementType>) -> Option<Self> {
        match element_type {
            Some(TensorElementType::Float32) => Some(Self::F32),
            Some(TensorElementType::Float16) => Some(Self::F16),
            _ => None,
        }
    }

    /// The ONNX Runtime element type.
    pub(super) fn element_type(self) -> TensorElementType {
        match self {
            Self::F32 => TensorElementType::Float32,
            Self::F16 => TensorElementType::Float16,
        }
    }

    /// A host tensor of shape `dims` holding zeros.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Inference`] if the tensor cannot be created.
    pub(super) fn zeros(self, dims: &[usize]) -> Result<DynValue, CoreError> {
        let len: usize = dims.iter().product();
        match self {
            Self::F32 => {
                Tensor::<f32>::from_array((dims, vec![0.0_f32; len])).map(Tensor::into_dyn)
            }
            Self::F16 => {
                Tensor::<f16>::from_array((dims, vec![f16::ZERO; len])).map(Tensor::into_dyn)
            }
        }
        .map_err(|e| CoreError::Inference(format!("cannot create a {self:?} tensor: {e}")))
    }
}

/// The values of a float tensor in host memory.
#[derive(Debug, Clone, Copy)]
pub(super) enum FloatData<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

impl<'a> FloatData<'a> {
    /// Number of values.
    pub(super) fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::F16(data) => data.len(),
        }
    }

    /// The values in `range` as `float32`.
    ///
    /// `float32` data is returned as it is; `float16` data is widened
    /// into `scratch`, which is only touched in that case.
    ///
    /// # Panics
    ///
    /// Panics if `range` is out of bounds.
    pub(super) fn widen<'s>(&self, range: Range<usize>, scratch: &'s mut Vec<f32>) -> &'s [f32]
    where
        'a: 's,
    {
        match self {
            Self::F32(data) => &data[range],
            Self::F16(data) => {
                let half = &data[range];
                scratch.clear();
                scratch.resize(half.len(), 0.0);
                half.convert_to_f32_slice(scratch);
                scratch
            }
        }
    }
}

/// Shape and values of a `float32` or `float16` tensor in host memory.
///
/// # Errors
///
/// Returns [`CoreError::Inference`] if `value` is not such a tensor.
pub(super) fn float_tensor(value: &DynValue) -> Result<(Vec<i64>, FloatData<'_>), CoreError> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Ok((shape.iter().copied().collect(), FloatData::F32(data)));
    }
    let (shape, data) = value
        .try_extract_tensor::<f16>()
        .map_err(|e| CoreError::Inference(format!("expected a float32 or float16 tensor: {e}")))?;
    Ok((shape.iter().copied().collect(), FloatData::F16(data)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float16_values_widen_exactly_and_float32_values_are_not_copied() {
        // Arrange
        let values = [0.0_f32, 1.5, -2.25, 1024.0, -0.000_061_035_156];
        let half: Vec<f16> = values.iter().map(|&value| f16::from_f32(value)).collect();
        let mut scratch = vec![9.0_f32; 2];

        // Act / Assert -- float16: every value is representable, so exact.
        let data = FloatData::F16(&half);
        assert_eq!(data.len(), 5);
        assert_eq!(data.widen(1..4, &mut scratch), &values[1..4]);
        assert_eq!(data.widen(0..5, &mut scratch), &values[..]);

        // float32: the same memory, scratch untouched.
        let data = FloatData::F32(&values);
        let before = scratch.clone();
        let widened = data.widen(2..4, &mut scratch);
        assert!(std::ptr::eq(widened.as_ptr(), values[2..].as_ptr()));
        assert_eq!(scratch, before);
    }

    #[test]
    fn zeros_have_the_requested_type_and_shape() {
        for float in [FloatType::F32, FloatType::F16] {
            let empty = float.zeros(&[3, 2, 0, 4]).expect("empty");
            assert_eq!(empty.data_type(), &float.element_type());
            let (shape, data) = float_tensor(&empty).expect("read");
            assert_eq!(shape, vec![3, 2, 0, 4]);
            assert_eq!(data.len(), 0);

            let filled = float.zeros(&[2, 3]).expect("zeros");
            let (shape, data) = float_tensor(&filled).expect("read");
            assert_eq!(shape, vec![2, 3]);
            let mut scratch = Vec::new();
            assert_eq!(data.widen(0..6, &mut scratch), &[0.0; 6]);
            assert_eq!(FloatType::of(Some(float.element_type())), Some(float));
        }
        assert_eq!(FloatType::of(Some(TensorElementType::Float64)), None);
        assert_eq!(FloatType::of(None), None);
    }
}
