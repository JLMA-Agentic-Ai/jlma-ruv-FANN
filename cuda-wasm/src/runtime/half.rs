//! Half-precision (fp16) floating-point support
//!
//! Provides a software `Half` type that emulates IEEE 754 half-precision
//! (binary16) arithmetic, mirroring CUDA's `__half` type. All operations
//! go through f32 internally, which matches the behavior of CUDA's
//! half-precision on hardware without native fp16 ALUs.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};

/// IEEE 754 half-precision floating-point (binary16)
///
/// Layout: 1 sign bit, 5 exponent bits, 10 mantissa bits.
/// Range: ±65504, smallest normal: 6.1×10⁻⁵, precision: ~3 decimal digits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Half {
    bits: u16,
}

impl Half {
    /// Zero constant
    pub const ZERO: Self = Self { bits: 0x0000 };
    /// One constant
    pub const ONE: Self = Self { bits: 0x3C00 };
    /// Negative one
    pub const NEG_ONE: Self = Self { bits: 0xBC00 };
    /// Positive infinity
    pub const INFINITY: Self = Self { bits: 0x7C00 };
    /// Negative infinity
    pub const NEG_INFINITY: Self = Self { bits: 0xFC00 };
    /// Not a number (NaN)
    pub const NAN: Self = Self { bits: 0x7E00 };
    /// Maximum finite value (65504)
    pub const MAX: Self = Self { bits: 0x7BFF };
    /// Minimum positive normal value
    pub const MIN_POSITIVE: Self = Self { bits: 0x0400 };
    /// Machine epsilon (2⁻¹⁰ ≈ 9.77×10⁻⁴)
    pub const EPSILON: Self = Self { bits: 0x1400 };

    /// Create from raw bits
    pub const fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    /// Get the raw bits
    pub const fn to_bits(self) -> u16 {
        self.bits
    }

    /// Convert from f32 to half-precision
    pub fn from_f32(value: f32) -> Self {
        Self { bits: f32_to_f16(value) }
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        f16_to_f32(self.bits)
    }

    /// Convert from f64 to half-precision
    pub fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }

    /// Convert to f64
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Check if NaN
    pub fn is_nan(self) -> bool {
        (self.bits & 0x7C00) == 0x7C00 && (self.bits & 0x03FF) != 0
    }

    /// Check if infinite
    pub fn is_infinite(self) -> bool {
        (self.bits & 0x7FFF) == 0x7C00
    }

    /// Check if finite
    pub fn is_finite(self) -> bool {
        (self.bits & 0x7C00) != 0x7C00
    }

    /// Check if normal (not zero, denormal, infinity, or NaN)
    pub fn is_normal(self) -> bool {
        let exp = self.bits & 0x7C00;
        exp != 0 && exp != 0x7C00
    }

    /// Check if zero (positive or negative)
    pub fn is_zero(self) -> bool {
        (self.bits & 0x7FFF) == 0
    }

    /// Check if sign bit is set
    pub fn is_sign_negative(self) -> bool {
        (self.bits & 0x8000) != 0
    }

    /// Absolute value
    pub fn abs(self) -> Self {
        Self { bits: self.bits & 0x7FFF }
    }

    /// Fused multiply-add: a * b + c
    pub fn fma(a: Self, b: Self, c: Self) -> Self {
        Self::from_f32(a.to_f32().mul_add(b.to_f32(), c.to_f32()))
    }

    /// Square root
    pub fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    /// Reciprocal (1/x)
    pub fn recip(self) -> Self {
        Self::from_f32(1.0 / self.to_f32())
    }

    /// Minimum of two values
    pub fn min(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().min(other.to_f32()))
    }

    /// Maximum of two values
    pub fn max(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().max(other.to_f32()))
    }

    /// Clamp between min and max
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self::from_f32(self.to_f32().clamp(min.to_f32(), max.to_f32()))
    }
}

// -- Conversion functions (IEEE 754 bit manipulation) -------------------------

/// Convert f32 to f16 bits
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exp == 0xFF {
        // Infinity or NaN
        if mantissa == 0 {
            return sign | 0x7C00; // Infinity
        } else {
            return sign | 0x7C00 | ((mantissa >> 13) as u16).max(1); // NaN
        }
    }

    let unbiased_exp = exp - 127;

    if unbiased_exp > 15 {
        // Overflow -> infinity
        return sign | 0x7C00;
    }

    if unbiased_exp < -24 {
        // Underflow -> zero
        return sign;
    }

    if unbiased_exp < -14 {
        // Denormalized
        let shift = -1 - unbiased_exp;
        let m = (mantissa | 0x00800000) >> (shift + 13);
        return sign | m as u16;
    }

    // Normal
    let f16_exp = ((unbiased_exp + 15) as u16) << 10;
    let f16_mantissa = (mantissa >> 13) as u16;
    sign | f16_exp | f16_mantissa
}

/// Convert f16 bits to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;

    if exp == 0x1F {
        // Infinity or NaN
        let f32_bits = sign | 0x7F800000 | (mantissa << 13);
        return f32::from_bits(f32_bits);
    }

    if exp == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign);
        }
        // Denormalized -> normalize
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f32_exp = ((e + 127) as u32) << 23;
        let f32_bits = sign | f32_exp | (m << 13);
        return f32::from_bits(f32_bits);
    }

    // Normal
    let f32_exp = ((exp as i32 - 15 + 127) as u32) << 23;
    let f32_bits = sign | f32_exp | (mantissa << 13);
    f32::from_bits(f32_bits)
}

// -- Operator implementations -------------------------------------------------

impl Add for Half {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub for Half {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for Half {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for Half {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Neg for Half {
    type Output = Self;
    fn neg(self) -> Self {
        Self { bits: self.bits ^ 0x8000 }
    }
}

impl PartialOrd for Half {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl Default for Half {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({})", self.to_f32())
    }
}

impl fmt::Display for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f32> for Half {
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<f64> for Half {
    fn from(v: f64) -> Self {
        Self::from_f64(v)
    }
}

impl From<Half> for f32 {
    fn from(v: Half) -> Self {
        v.to_f32()
    }
}

impl From<Half> for f64 {
    fn from(v: Half) -> Self {
        v.to_f64()
    }
}

/// Convert a slice of f32 to half-precision
pub fn f32_to_half_slice(src: &[f32]) -> Vec<Half> {
    src.iter().map(|&v| Half::from_f32(v)).collect()
}

/// Convert a slice of half-precision to f32
pub fn half_to_f32_slice(src: &[Half]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

/// Dot product in half-precision (accumulated in f32 for precision)
pub fn half_dot(a: &[Half], b: &[Half]) -> Half {
    let acc: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.to_f32() * y.to_f32())
        .sum();
    Half::from_f32(acc)
}

/// GEMV (General Matrix-Vector multiply) in half-precision
pub fn half_gemv(
    m: usize,
    n: usize,
    alpha: Half,
    a: &[Half],      // m x n matrix (row-major)
    x: &[Half],      // n-element vector
    beta: Half,
    y: &mut [Half],   // m-element vector
) {
    let alpha_f = alpha.to_f32();
    let beta_f = beta.to_f32();

    for i in 0..m {
        let mut sum: f32 = 0.0;
        for j in 0..n {
            sum += a[i * n + j].to_f32() * x[j].to_f32();
        }
        let result = alpha_f * sum + beta_f * y[i].to_f32();
        y[i] = Half::from_f32(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_zero() {
        assert_eq!(Half::ZERO.to_f32(), 0.0);
        assert!(Half::ZERO.is_zero());
    }

    #[test]
    fn test_half_one() {
        assert_eq!(Half::ONE.to_f32(), 1.0);
    }

    #[test]
    fn test_half_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0, -100.0, 0.001];
        for &v in &values {
            let h = Half::from_f32(v);
            let back = h.to_f32();
            assert!((back - v).abs() < 0.01, "Roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_half_infinity() {
        assert!(Half::INFINITY.is_infinite());
        assert!(!Half::INFINITY.is_finite());
        assert!(Half::NEG_INFINITY.is_infinite());
    }

    #[test]
    fn test_half_nan() {
        assert!(Half::NAN.is_nan());
        assert!(!Half::NAN.is_finite());
        assert!(!Half::NAN.is_normal());
    }

    #[test]
    fn test_half_arithmetic() {
        let a = Half::from_f32(2.0);
        let b = Half::from_f32(3.0);

        assert_eq!((a + b).to_f32(), 5.0);
        assert_eq!((b - a).to_f32(), 1.0);
        assert_eq!((a * b).to_f32(), 6.0);
        let div_result = (b / a).to_f32();
        assert!((div_result - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_half_negation() {
        let a = Half::from_f32(5.0);
        assert_eq!((-a).to_f32(), -5.0);
        assert_eq!((-(-a)).to_f32(), 5.0);
    }

    #[test]
    fn test_half_comparison() {
        let a = Half::from_f32(1.0);
        let b = Half::from_f32(2.0);

        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
        assert!(a >= a);
    }

    #[test]
    fn test_half_abs() {
        let neg = Half::from_f32(-3.5);
        let pos = neg.abs();
        assert!((pos.to_f32() - 3.5).abs() < 0.01);
    }

    #[test]
    fn test_half_fma() {
        let a = Half::from_f32(2.0);
        let b = Half::from_f32(3.0);
        let c = Half::from_f32(1.0);

        let result = Half::fma(a, b, c);
        assert!((result.to_f32() - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_half_sqrt() {
        let a = Half::from_f32(4.0);
        assert!((a.sqrt().to_f32() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_half_min_max() {
        let a = Half::from_f32(1.0);
        let b = Half::from_f32(3.0);

        assert_eq!(a.min(b).to_f32(), 1.0);
        assert_eq!(a.max(b).to_f32(), 3.0);
    }

    #[test]
    fn test_half_clamp() {
        let v = Half::from_f32(5.0);
        let lo = Half::from_f32(0.0);
        let hi = Half::from_f32(3.0);

        assert_eq!(v.clamp(lo, hi).to_f32(), 3.0);
    }

    #[test]
    fn test_half_overflow() {
        let big = Half::from_f32(100000.0);
        assert!(big.is_infinite());
    }

    #[test]
    fn test_half_underflow() {
        let tiny = Half::from_f32(1e-10);
        assert!(tiny.is_zero() || !tiny.is_normal());
    }

    #[test]
    fn test_f32_to_half_slice() {
        let src = vec![1.0f32, 2.0, 3.0];
        let halves = f32_to_half_slice(&src);
        let back = half_to_f32_slice(&halves);
        assert_eq!(back, src);
    }

    #[test]
    fn test_half_dot_product() {
        let a = f32_to_half_slice(&[1.0, 2.0, 3.0]);
        let b = f32_to_half_slice(&[4.0, 5.0, 6.0]);

        let result = half_dot(&a, &b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((result.to_f32() - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_half_gemv() {
        // 2x3 matrix [[1,2,3],[4,5,6]]
        let a = f32_to_half_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = f32_to_half_slice(&[1.0, 1.0, 1.0]);
        let mut y = f32_to_half_slice(&[0.0, 0.0]);

        half_gemv(2, 3, Half::ONE, &a, &x, Half::ZERO, &mut y);

        assert!((y[0].to_f32() - 6.0).abs() < 0.1);  // 1+2+3
        assert!((y[1].to_f32() - 15.0).abs() < 0.1); // 4+5+6
    }

    #[test]
    fn test_half_display() {
        let h = Half::from_f32(3.14);
        let s = format!("{}", h);
        assert!(s.starts_with("3.1"));
    }

    #[test]
    fn test_half_from_f64() {
        let h = Half::from_f64(2.5);
        assert!((h.to_f64() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_half_recip() {
        let a = Half::from_f32(4.0);
        assert!((a.recip().to_f32() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_half_max_value() {
        let max = Half::MAX;
        assert!((max.to_f32() - 65504.0).abs() < 1.0);
        assert!(max.is_finite());
    }

    #[test]
    fn test_half_is_sign_negative() {
        assert!(!Half::from_f32(1.0).is_sign_negative());
        assert!(Half::from_f32(-1.0).is_sign_negative());
        assert!(!Half::ZERO.is_sign_negative());
    }
}
