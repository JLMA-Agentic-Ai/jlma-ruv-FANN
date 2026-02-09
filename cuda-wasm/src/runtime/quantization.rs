//! INT8/INT4 Quantization for inference acceleration
//!
//! Provides quantization and dequantization primitives used in neural network
//! inference to reduce memory bandwidth and leverage integer arithmetic units.
//! Supports symmetric and asymmetric quantization schemes.
//!
//! Reference: "Quantization and Training of Neural Networks for Efficient
//! Integer-Arithmetic-Only Inference" — Jacob et al., CVPR 2018

use std::fmt;

/// Quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantScheme {
    /// Symmetric: zero_point = 0, range = [-scale*127, scale*127]
    Symmetric,
    /// Asymmetric: zero_point ≠ 0, full [0, 255] range used
    Asymmetric,
}

/// Quantization bit width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantBits {
    /// 8-bit integers (INT8).
    Int8,
    /// 4-bit integers (INT4), packed 2 per byte.
    Int4,
}

/// Quantization parameters computed from calibration.
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
    pub bits: QuantBits,
    pub scheme: QuantScheme,
    /// Per-channel scales (if per-channel quantization).
    pub per_channel_scales: Option<Vec<f32>>,
}

impl QuantParams {
    /// Compute quantization parameters from data range.
    pub fn from_range(min_val: f32, max_val: f32, bits: QuantBits, scheme: QuantScheme) -> Self {
        let (qmin, qmax) = match bits {
            QuantBits::Int8 => (-128i32, 127i32),
            QuantBits::Int4 => (-8i32, 7i32),
        };

        match scheme {
            QuantScheme::Symmetric => {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = abs_max / qmax as f32;
                Self {
                    scale: if scale == 0.0 { 1.0 } else { scale },
                    zero_point: 0,
                    bits,
                    scheme,
                    per_channel_scales: None,
                }
            }
            QuantScheme::Asymmetric => {
                let range = max_val - min_val;
                let scale = range / (qmax - qmin) as f32;
                let zero_point = (qmin as f32 - min_val / scale).round() as i32;
                Self {
                    scale: if scale == 0.0 { 1.0 } else { scale },
                    zero_point: zero_point.clamp(qmin, qmax),
                    bits,
                    scheme,
                    per_channel_scales: None,
                }
            }
        }
    }

    /// Compute parameters from data using min/max calibration.
    pub fn calibrate(data: &[f32], bits: QuantBits, scheme: QuantScheme) -> Self {
        if data.is_empty() {
            return Self { scale: 1.0, zero_point: 0, bits, scheme, per_channel_scales: None };
        }
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Self::from_range(min_val, max_val, bits, scheme)
    }
}

/// Quantize an f32 tensor to INT8.
pub fn quantize_int8(data: &[f32], params: &QuantParams) -> Vec<i8> {
    data.iter().map(|&x| {
        let q = (x / params.scale).round() as i32 + params.zero_point;
        q.clamp(-128, 127) as i8
    }).collect()
}

/// Dequantize INT8 to f32.
pub fn dequantize_int8(data: &[i8], params: &QuantParams) -> Vec<f32> {
    data.iter().map(|&q| {
        (q as i32 - params.zero_point) as f32 * params.scale
    }).collect()
}

/// Quantize an f32 tensor to INT4 (packed, 2 values per byte).
pub fn quantize_int4(data: &[f32], params: &QuantParams) -> Vec<u8> {
    let mut packed = Vec::with_capacity((data.len() + 1) / 2);
    for chunk in data.chunks(2) {
        let lo = {
            let q = (chunk[0] / params.scale).round() as i32 + params.zero_point;
            (q.clamp(-8, 7) & 0x0F) as u8
        };
        let hi = if chunk.len() > 1 {
            let q = (chunk[1] / params.scale).round() as i32 + params.zero_point;
            ((q.clamp(-8, 7) & 0x0F) as u8) << 4
        } else {
            0
        };
        packed.push(lo | hi);
    }
    packed
}

/// Dequantize INT4 (packed) to f32.
pub fn dequantize_int4(data: &[u8], count: usize, params: &QuantParams) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    for &byte in data {
        if result.len() >= count { break; }
        // Low nibble (sign-extend from 4 bits)
        let lo = (byte & 0x0F) as i8;
        let lo = if lo & 0x08 != 0 { lo | !0x0F_u8 as i8 } else { lo }; // sign extend
        result.push((lo as i32 - params.zero_point) as f32 * params.scale);

        if result.len() >= count { break; }
        // High nibble
        let hi = ((byte >> 4) & 0x0F) as i8;
        let hi = if hi & 0x08 != 0 { hi | !0x0F_u8 as i8 } else { hi };
        result.push((hi as i32 - params.zero_point) as f32 * params.scale);
    }
    result
}

/// INT8 matrix multiply with f32 accumulation: C = A · B
/// A: (m × k) as i8, B: (k × n) as i8, C: (m × n) as i32 → f32
pub fn quantized_gemm_int8(
    a: &[i8], b: &[i8],
    m: usize, k: usize, n: usize,
    a_params: &QuantParams, b_params: &QuantParams,
) -> Vec<f32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p] as i32 - a_params.zero_point;
            for j in 0..n {
                let b_val = b[p * n + j] as i32 - b_params.zero_point;
                c[i * n + j] += a_val * b_val;
            }
        }
    }
    // Dequantize result
    let output_scale = a_params.scale * b_params.scale;
    c.iter().map(|&v| v as f32 * output_scale).collect()
}

/// Compute quantization error (MSE) between original and quantized-dequantized.
pub fn quantization_error(original: &[f32], params: &QuantParams) -> QuantError {
    let quantized = quantize_int8(original, params);
    let dequantized = dequantize_int8(&quantized, params);

    let mse: f64 = original.iter().zip(dequantized.iter())
        .map(|(&o, &d)| ((o - d) as f64).powi(2))
        .sum::<f64>() / original.len() as f64;

    let max_error = original.iter().zip(dequantized.iter())
        .map(|(&o, &d)| (o - d).abs())
        .fold(0.0f32, f32::max);

    let signal_power: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / original.len() as f64;
    let snr = if mse > 0.0 { 10.0 * (signal_power / mse).log10() } else { f64::INFINITY };

    QuantError {
        mse: mse as f32,
        max_error,
        snr_db: snr as f32,
        compression_ratio: match params.bits {
            QuantBits::Int8 => 4.0,  // f32 → i8
            QuantBits::Int4 => 8.0,  // f32 → i4
        },
    }
}

/// Quantization error statistics.
#[derive(Debug, Clone)]
pub struct QuantError {
    pub mse: f32,
    pub max_error: f32,
    pub snr_db: f32,
    pub compression_ratio: f32,
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantError: MSE={:.6}, MaxErr={:.4}, SNR={:.1}dB, {}x compression",
            self.mse, self.max_error, self.snr_db, self.compression_ratio)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_int8_roundtrip() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let params = QuantParams::calibrate(&data, QuantBits::Int8, QuantScheme::Symmetric);
        let quantized = quantize_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);

        for i in 0..data.len() {
            assert!((data[i] - dequantized[i]).abs() < 0.02,
                "Mismatch at {}: original={}, dequantized={}", i, data[i], dequantized[i]);
        }
    }

    #[test]
    fn test_asymmetric_int8() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let params = QuantParams::calibrate(&data, QuantBits::Int8, QuantScheme::Asymmetric);
        let quantized = quantize_int8(&data, &params);
        let dequantized = dequantize_int8(&quantized, &params);

        for i in 0..data.len() {
            assert!((data[i] - dequantized[i]).abs() < 0.02,
                "Asymmetric mismatch at {}: {} vs {}", i, data[i], dequantized[i]);
        }
    }

    #[test]
    fn test_int4_quantization() {
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let params = QuantParams::calibrate(&data, QuantBits::Int4, QuantScheme::Symmetric);
        let packed = quantize_int4(&data, &params);
        let dequantized = dequantize_int4(&packed, data.len(), &params);

        assert_eq!(dequantized.len(), data.len());
        // INT4 has less precision, allow larger tolerance
        for i in 0..data.len() {
            assert!((data[i] - dequantized[i]).abs() < 0.5,
                "INT4 mismatch at {}: {} vs {}", i, data[i], dequantized[i]);
        }
    }

    #[test]
    fn test_int4_packing() {
        let data = vec![0.0, 0.0, 0.0, 0.0]; // 4 values → 2 bytes
        let params = QuantParams::from_range(-1.0, 1.0, QuantBits::Int4, QuantScheme::Symmetric);
        let packed = quantize_int4(&data, &params);
        assert_eq!(packed.len(), 2);
    }

    #[test]
    fn test_quantized_gemm() {
        // A: 2×2, B: 2×2
        let a_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_f32 = vec![5.0f32, 6.0, 7.0, 8.0];

        let a_params = QuantParams::calibrate(&a_f32, QuantBits::Int8, QuantScheme::Symmetric);
        let b_params = QuantParams::calibrate(&b_f32, QuantBits::Int8, QuantScheme::Symmetric);

        let a_q = quantize_int8(&a_f32, &a_params);
        let b_q = quantize_int8(&b_f32, &b_params);

        let c = quantized_gemm_int8(&a_q, &b_q, 2, 2, 2, &a_params, &b_params);
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert!((c[0] - 19.0).abs() < 1.0, "Got {}", c[0]);
        assert!((c[1] - 22.0).abs() < 1.0, "Got {}", c[1]);
        assert!((c[2] - 43.0).abs() < 1.5, "Got {}", c[2]);
        assert!((c[3] - 50.0).abs() < 1.5, "Got {}", c[3]);
    }

    #[test]
    fn test_quantization_error() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let params = QuantParams::calibrate(&data, QuantBits::Int8, QuantScheme::Symmetric);
        let error = quantization_error(&data, &params);

        assert!(error.mse < 0.001, "MSE too high: {}", error.mse);
        assert!(error.snr_db > 30.0, "SNR too low: {}dB", error.snr_db);
        assert_eq!(error.compression_ratio, 4.0);
    }

    #[test]
    fn test_quantization_error_int4() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        let params = QuantParams::calibrate(&data, QuantBits::Int4, QuantScheme::Symmetric);
        let error = quantization_error(&data, &params);
        assert_eq!(error.compression_ratio, 8.0);
        // INT4 will have higher error than INT8
    }

    #[test]
    fn test_zero_range_calibration() {
        let data = vec![0.0, 0.0, 0.0];
        let params = QuantParams::calibrate(&data, QuantBits::Int8, QuantScheme::Symmetric);
        assert_eq!(params.scale, 1.0); // Should not be zero
    }

    #[test]
    fn test_empty_calibration() {
        let params = QuantParams::calibrate(&[], QuantBits::Int8, QuantScheme::Symmetric);
        assert_eq!(params.scale, 1.0);
    }

    #[test]
    fn test_quant_error_display() {
        let error = QuantError { mse: 0.001, max_error: 0.01, snr_db: 40.0, compression_ratio: 4.0 };
        let s = format!("{}", error);
        assert!(s.contains("MSE"));
        assert!(s.contains("4x"));
    }
}
