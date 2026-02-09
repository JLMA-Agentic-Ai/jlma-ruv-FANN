//! BFloat16 (bf16) floating-point support
//!
//! Implements the Google Brain bfloat16 format used extensively in ML training.
//! BF16 has the same exponent range as f32 (8 bits) but reduced mantissa (7 bits),
//! making it ideal for training where range matters more than precision.
//!
//! Layout: 1 sign bit, 8 exponent bits, 7 mantissa bits.
//! Range: same as f32 (±3.4×10³⁸), precision: ~2 decimal digits.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};

/// BFloat16 — Google Brain's 16-bit floating-point format.
///
/// Unlike IEEE fp16 (Half), bf16 shares f32's exponent range, making it
/// a drop-in replacement for f32 in training loops where dynamic range
/// is more important than mantissa precision.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BFloat16 {
    bits: u16,
}

impl BFloat16 {
    pub const ZERO: Self = Self { bits: 0x0000 };
    pub const ONE: Self = Self { bits: 0x3F80 };
    pub const NEG_ONE: Self = Self { bits: 0xBF80 };
    pub const INFINITY: Self = Self { bits: 0x7F80 };
    pub const NEG_INFINITY: Self = Self { bits: 0xFF80 };
    pub const NAN: Self = Self { bits: 0x7FC0 };
    pub const MAX: Self = Self { bits: 0x7F7F }; // ~3.39×10³⁸
    pub const MIN_POSITIVE: Self = Self { bits: 0x0080 }; // smallest normal
    pub const EPSILON: Self = Self { bits: 0x3C00 }; // 2^-7 ≈ 0.0078125

    /// Create from raw u16 bits.
    pub fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    /// Get raw u16 bits.
    pub fn to_bits(self) -> u16 {
        self.bits
    }

    /// Convert from f32 (truncation, matching hardware behavior).
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        // Round to nearest even: check bit 16 (round bit) and bits 0-15 (sticky)
        let round_bit = (bits >> 15) & 1;
        let sticky = if bits & 0x7FFF != 0 { 1u32 } else { 0 };
        let lsb = (bits >> 16) & 1;

        // Round to nearest, ties to even
        let rounded = (bits >> 16) + (round_bit & (sticky | lsb));

        // Handle overflow to infinity
        if (rounded & 0x7F80) == 0x7F80 && (bits & 0x7F800000) != 0x7F800000 {
            // Rounding overflowed to inf, but original was finite
            Self { bits: ((bits >> 16) & 0xFF80) as u16 | 0x7F }
        } else {
            Self { bits: rounded as u16 }
        }
    }

    /// Convert to f32 (lossless — just pad lower 16 bits with zeros).
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.bits as u32) << 16)
    }

    /// Check if NaN.
    pub fn is_nan(self) -> bool {
        (self.bits & 0x7F80) == 0x7F80 && (self.bits & 0x007F) != 0
    }

    /// Check if infinite.
    pub fn is_infinite(self) -> bool {
        (self.bits & 0x7FFF) == 0x7F80
    }

    /// Check if finite (not NaN or infinite).
    pub fn is_finite(self) -> bool {
        (self.bits & 0x7F80) != 0x7F80
    }

    /// Check if zero (positive or negative).
    pub fn is_zero(self) -> bool {
        (self.bits & 0x7FFF) == 0
    }

    /// Check if the sign bit is set.
    pub fn is_sign_negative(self) -> bool {
        self.bits & 0x8000 != 0
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self { bits: self.bits & 0x7FFF }
    }

    /// Fused multiply-add: a * b + c (computed in f32).
    pub fn fma(a: BFloat16, b: BFloat16, c: BFloat16) -> BFloat16 {
        BFloat16::from_f32(a.to_f32().mul_add(b.to_f32(), c.to_f32()))
    }

    /// Square root.
    pub fn sqrt(self) -> Self {
        BFloat16::from_f32(self.to_f32().sqrt())
    }

    /// Reciprocal (1/x).
    pub fn recip(self) -> Self {
        BFloat16::from_f32(1.0 / self.to_f32())
    }

    /// Minimum of two values (NaN-propagating).
    pub fn min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::NAN;
        }
        if self.to_f32() <= other.to_f32() { self } else { other }
    }

    /// Maximum of two values (NaN-propagating).
    pub fn max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::NAN;
        }
        if self.to_f32() >= other.to_f32() { self } else { other }
    }

    /// Clamp value to [lo, hi].
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
}

// ── Arithmetic ops ─────────────────────────────────────────────────

impl Add for BFloat16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        BFloat16::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub for BFloat16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        BFloat16::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for BFloat16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        BFloat16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for BFloat16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        BFloat16::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Neg for BFloat16 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { bits: self.bits ^ 0x8000 }
    }
}

impl PartialOrd for BFloat16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl fmt::Debug for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bf16({:.4})", self.to_f32())
    }
}

impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.to_f32())
    }
}

impl From<f32> for BFloat16 {
    fn from(v: f32) -> Self { BFloat16::from_f32(v) }
}

impl From<BFloat16> for f32 {
    fn from(v: BFloat16) -> f32 { v.to_f32() }
}

// ── Batch operations ───────────────────────────────────────────────

/// Convert an f32 slice to bf16.
pub fn f32_to_bf16_slice(input: &[f32]) -> Vec<BFloat16> {
    input.iter().map(|&v| BFloat16::from_f32(v)).collect()
}

/// Convert a bf16 slice to f32.
pub fn bf16_to_f32_slice(input: &[BFloat16]) -> Vec<f32> {
    input.iter().map(|v| v.to_f32()).collect()
}

/// Dot product of two bf16 slices, accumulated in f32.
pub fn bf16_dot(a: &[BFloat16], b: &[BFloat16]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x.to_f32() * y.to_f32()).sum()
}

/// Matrix-vector multiply: y = A * x, with bf16 inputs and f32 accumulation.
/// A is (rows × cols) row-major, x is (cols,), y is (rows,).
pub fn bf16_gemv(a: &[BFloat16], x: &[BFloat16], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|r| {
        let row_start = r * cols;
        (0..cols).map(|c| {
            a[row_start + c].to_f32() * x[c].to_f32()
        }).sum()
    }).collect()
}

/// Mixed-precision GEMM: C = A * B with bf16 inputs and f32 accumulation.
/// A is (m × k), B is (k × n), C is (m × n).
pub fn bf16_gemm(a: &[BFloat16], b: &[BFloat16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p].to_f32();
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j].to_f32();
            }
        }
    }
    c
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -0.125, 3.14];
        for &v in &values {
            let bf = BFloat16::from_f32(v);
            let back = bf.to_f32();
            assert!((back - v).abs() < 0.05, "Roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_bf16_constants() {
        assert_eq!(BFloat16::ZERO.to_f32(), 0.0);
        assert_eq!(BFloat16::ONE.to_f32(), 1.0);
        assert_eq!(BFloat16::NEG_ONE.to_f32(), -1.0);
        assert!(BFloat16::INFINITY.is_infinite());
        assert!(BFloat16::NAN.is_nan());
        assert!(BFloat16::MAX.to_f32() > 1e38);
    }

    #[test]
    fn test_bf16_arithmetic() {
        let a = BFloat16::from_f32(2.0);
        let b = BFloat16::from_f32(3.0);
        assert!((a + b).to_f32() - 5.0 < 0.1);
        assert!((a - b).to_f32() - (-1.0) < 0.1);
        assert!((a * b).to_f32() - 6.0 < 0.1);
        assert!(((a / b).to_f32() - 0.6667).abs() < 0.02);
    }

    #[test]
    fn test_bf16_neg() {
        let a = BFloat16::from_f32(42.0);
        assert!((-a).to_f32() < 0.0);
        assert!(((-a).to_f32() + 42.0).abs() < 0.5);
    }

    #[test]
    fn test_bf16_comparison() {
        let a = BFloat16::from_f32(1.0);
        let b = BFloat16::from_f32(2.0);
        assert!(a < b);
        assert!(b > a);
        assert!(BFloat16::NAN.partial_cmp(&a).is_none());
    }

    #[test]
    fn test_bf16_special_values() {
        assert!(BFloat16::NAN.is_nan());
        assert!(!BFloat16::NAN.is_finite());
        assert!(BFloat16::INFINITY.is_infinite());
        assert!(!BFloat16::INFINITY.is_finite());
        assert!(BFloat16::ZERO.is_zero());
        assert!(BFloat16::from_bits(0x8000).is_zero()); // -0
    }

    #[test]
    fn test_bf16_fma() {
        let a = BFloat16::from_f32(2.0);
        let b = BFloat16::from_f32(3.0);
        let c = BFloat16::from_f32(1.0);
        let result = BFloat16::fma(a, b, c);
        assert!((result.to_f32() - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_bf16_sqrt() {
        let a = BFloat16::from_f32(4.0);
        assert!((a.sqrt().to_f32() - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_bf16_clamp() {
        let lo = BFloat16::from_f32(0.0);
        let hi = BFloat16::from_f32(1.0);
        let v = BFloat16::from_f32(1.5);
        assert!((v.clamp(lo, hi).to_f32() - 1.0).abs() < 0.01);
        let v2 = BFloat16::from_f32(-0.5);
        assert!((v2.clamp(lo, hi).to_f32()).abs() < 0.01);
    }

    #[test]
    fn test_bf16_batch_convert() {
        let f32s = vec![1.0f32, 2.0, 3.0, 4.0];
        let bf16s = f32_to_bf16_slice(&f32s);
        let back = bf16_to_f32_slice(&bf16s);
        for i in 0..f32s.len() {
            assert!((back[i] - f32s[i]).abs() < 0.05);
        }
    }

    #[test]
    fn test_bf16_dot() {
        let a = f32_to_bf16_slice(&[1.0, 2.0, 3.0]);
        let b = f32_to_bf16_slice(&[4.0, 5.0, 6.0]);
        let result = bf16_dot(&a, &b);
        assert!((result - 32.0).abs() < 0.5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_bf16_gemv() {
        let a = f32_to_bf16_slice(&[1.0, 2.0, 3.0, 4.0]); // 2x2
        let x = f32_to_bf16_slice(&[1.0, 1.0]);
        let y = bf16_gemv(&a, &x, 2, 2);
        assert!((y[0] - 3.0).abs() < 0.1); // 1+2
        assert!((y[1] - 7.0).abs() < 0.1); // 3+4
    }

    #[test]
    fn test_bf16_gemm() {
        // 2x2 * 2x2
        let a = f32_to_bf16_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = f32_to_bf16_slice(&[5.0, 6.0, 7.0, 8.0]);
        let c = bf16_gemm(&a, &b, 2, 2, 2);
        assert!((c[0] - 19.0).abs() < 0.5); // 1*5+2*7
        assert!((c[1] - 22.0).abs() < 0.5); // 1*6+2*8
        assert!((c[2] - 43.0).abs() < 0.5); // 3*5+4*7
        assert!((c[3] - 50.0).abs() < 0.5); // 3*6+4*8
    }

    #[test]
    fn test_bf16_same_range_as_f32() {
        // bf16 should handle very large values that fp16 cannot
        let big = BFloat16::from_f32(1e30);
        assert!(big.to_f32() > 1e29);
        assert!(big.is_finite());

        let small = BFloat16::from_f32(1e-30);
        assert!(small.to_f32() > 0.0);
        assert!(small.is_finite());
    }

    #[test]
    fn test_bf16_display() {
        let v = BFloat16::from_f32(3.14);
        let s = format!("{}", v);
        assert!(s.contains("3.1"), "Expected ~3.14, got {}", s);
    }
}
