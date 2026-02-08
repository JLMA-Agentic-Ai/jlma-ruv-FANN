//! SIMD-accelerated vector operations for CPU fallback paths
//!
//! Provides vectorized element-wise add, multiply, scale, dot product, and
//! reduction operations. Architecture-specific implementations are selected
//! at compile time via `cfg(target_arch)`, with a scalar fallback for
//! unsupported platforms.

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Element-wise addition: `c[i] = a[i] + b[i]`
///
/// # Panics
/// Panics if `a`, `b`, and `c` do not all have the same length.
pub fn vector_add_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "vector_add_f32: a.len() != b.len()");
    assert_eq!(a.len(), c.len(), "vector_add_f32: a.len() != c.len()");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: length equality checked above; AVX2 detected at runtime.
            unsafe { avx2::vector_add_f32_avx2(a, b, c) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Safety: NEON is mandatory on aarch64; length equality checked above.
        unsafe { neon::vector_add_f32_neon(a, b, c) };
        return;
    }

    // Scalar fallback (also used on wasm32 and other architectures)
    #[allow(unreachable_code)]
    scalar::vector_add_f32_scalar(a, b, c);
}

/// Element-wise multiplication: `c[i] = a[i] * b[i]`
///
/// # Panics
/// Panics if `a`, `b`, and `c` do not all have the same length.
pub fn vector_mul_f32(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "vector_mul_f32: a.len() != b.len()");
    assert_eq!(a.len(), c.len(), "vector_mul_f32: a.len() != c.len()");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::vector_mul_f32_avx2(a, b, c) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vector_mul_f32_neon(a, b, c) };
        return;
    }

    #[allow(unreachable_code)]
    scalar::vector_mul_f32_scalar(a, b, c);
}

/// Scale every element: `c[i] = a[i] * scalar`
///
/// # Panics
/// Panics if `a` and `c` do not have the same length.
pub fn vector_scale_f32(a: &[f32], scalar: f32, c: &mut [f32]) {
    assert_eq!(a.len(), c.len(), "vector_scale_f32: a.len() != c.len()");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::vector_scale_f32_avx2(a, scalar, c) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vector_scale_f32_neon(a, scalar, c) };
        return;
    }

    #[allow(unreachable_code)]
    scalar::vector_scale_f32_scalar(a, scalar, c);
}

/// Dot product: `sum(a[i] * b[i])`
///
/// # Panics
/// Panics if `a` and `b` do not have the same length.
pub fn vector_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector_dot_f32: a.len() != b.len()");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vector_dot_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::vector_dot_f32_neon(a, b) };
    }

    #[allow(unreachable_code)]
    scalar::vector_dot_f32_scalar(a, b)
}

/// Sum reduction: `sum(a[i])`
pub fn vector_reduce_sum_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vector_reduce_sum_f32_avx2(a) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::vector_reduce_sum_f32_neon(a) };
    }

    #[allow(unreachable_code)]
    scalar::vector_reduce_sum_f32_scalar(a)
}

// ---------------------------------------------------------------------------
// Scalar fallback implementation
// ---------------------------------------------------------------------------
mod scalar {
    pub fn vector_add_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..a.len() {
            c[i] = a[i] + b[i];
        }
    }

    pub fn vector_mul_f32_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..a.len() {
            c[i] = a[i] * b[i];
        }
    }

    pub fn vector_scale_f32_scalar(a: &[f32], scalar: f32, c: &mut [f32]) {
        for i in 0..a.len() {
            c[i] = a[i] * scalar;
        }
    }

    pub fn vector_dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    pub fn vector_reduce_sum_f32_scalar(a: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for &val in a {
            sum += val;
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const AVX2_F32_LANES: usize = 8;

    /// AVX2 vector addition: processes 8 f32s per iteration.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available and all slices have the same length.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vector_add_f32_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
        let n = a.len();
        let chunks = n / AVX2_F32_LANES;
        let remainder = n % AVX2_F32_LANES;

        for i in 0..chunks {
            let offset = i * AVX2_F32_LANES;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(c.as_mut_ptr().add(offset), vc);
        }

        // Handle remaining elements
        let tail_start = chunks * AVX2_F32_LANES;
        for i in 0..remainder {
            c[tail_start + i] = a[tail_start + i] + b[tail_start + i];
        }
    }

    /// AVX2 vector multiplication.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vector_mul_f32_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
        let n = a.len();
        let chunks = n / AVX2_F32_LANES;
        let remainder = n % AVX2_F32_LANES;

        for i in 0..chunks {
            let offset = i * AVX2_F32_LANES;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            let vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(c.as_mut_ptr().add(offset), vc);
        }

        let tail_start = chunks * AVX2_F32_LANES;
        for i in 0..remainder {
            c[tail_start + i] = a[tail_start + i] * b[tail_start + i];
        }
    }

    /// AVX2 scalar multiplication.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vector_scale_f32_avx2(a: &[f32], scalar: f32, c: &mut [f32]) {
        let n = a.len();
        let chunks = n / AVX2_F32_LANES;
        let remainder = n % AVX2_F32_LANES;
        let vs = _mm256_set1_ps(scalar);

        for i in 0..chunks {
            let offset = i * AVX2_F32_LANES;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vc = _mm256_mul_ps(va, vs);
            _mm256_storeu_ps(c.as_mut_ptr().add(offset), vc);
        }

        let tail_start = chunks * AVX2_F32_LANES;
        for i in 0..remainder {
            c[tail_start + i] = a[tail_start + i] * scalar;
        }
    }

    /// AVX2 dot product.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vector_dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / AVX2_F32_LANES;
        let remainder = n % AVX2_F32_LANES;

        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * AVX2_F32_LANES;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            // Use FMA if available for better precision and performance
            acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        }

        // Horizontal sum of the 8-wide accumulator
        let sum = hsum_avx2(acc);

        // Tail
        let tail_start = chunks * AVX2_F32_LANES;
        let mut tail_sum = 0.0f32;
        for i in 0..remainder {
            tail_sum += a[tail_start + i] * b[tail_start + i];
        }

        sum + tail_sum
    }

    /// AVX2 sum reduction.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vector_reduce_sum_f32_avx2(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / AVX2_F32_LANES;
        let remainder = n % AVX2_F32_LANES;

        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * AVX2_F32_LANES;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            acc = _mm256_add_ps(acc, va);
        }

        let sum = hsum_avx2(acc);

        let tail_start = chunks * AVX2_F32_LANES;
        let mut tail_sum = 0.0f32;
        for i in 0..remainder {
            tail_sum += a[tail_start + i];
        }

        sum + tail_sum
    }

    /// Horizontal sum of an __m256 register (8 x f32 -> single f32).
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_avx2(v: __m256) -> f32 {
        // Add high 128 to low 128
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        // Horizontal add within 128 bits
        let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
        let sums = _mm_add_ps(sum128, shuf); // [0+1, _, 2+3, _]
        let shuf2 = _mm_movehl_ps(sums, sums); // [2+3, _, _, _]
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
}

// ---------------------------------------------------------------------------
// NEON implementation (aarch64)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    const NEON_F32_LANES: usize = 4;

    /// NEON vector addition: processes 4 f32s per iteration.
    ///
    /// # Safety
    /// Caller must ensure all slices have the same length. NEON is mandatory on aarch64.
    pub unsafe fn vector_add_f32_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
        let n = a.len();
        let chunks = n / NEON_F32_LANES;
        let remainder = n % NEON_F32_LANES;

        for i in 0..chunks {
            let offset = i * NEON_F32_LANES;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let vc = vaddq_f32(va, vb);
            vst1q_f32(c.as_mut_ptr().add(offset), vc);
        }

        let tail_start = chunks * NEON_F32_LANES;
        for i in 0..remainder {
            c[tail_start + i] = a[tail_start + i] + b[tail_start + i];
        }
    }

    /// NEON vector multiplication.
    pub unsafe fn vector_mul_f32_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
        let n = a.len();
        let chunks = n / NEON_F32_LANES;
        let remainder = n % NEON_F32_LANES;

        for i in 0..chunks {
            let offset = i * NEON_F32_LANES;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            let vc = vmulq_f32(va, vb);
            vst1q_f32(c.as_mut_ptr().add(offset), vc);
        }

        let tail_start = chunks * NEON_F32_LANES;
        for i in 0..remainder {
            c[tail_start + i] = a[tail_start + i] * b[tail_start + i];
        }
    }

    /// NEON scalar multiplication.
    pub unsafe fn vector_scale_f32_neon(a: &[f32], scalar: f32, c: &mut [f32]) {
        let n = a.len();
        let chunks = n / NEON_F32_LANES;
        let remainder = n % NEON_F32_LANES;
        let vs = vdupq_n_f32(scalar);

        for i in 0..chunks {
            let offset = i * NEON_F32_LANES;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vc = vmulq_f32(va, vs);
            vst1q_f32(c.as_mut_ptr().add(offset), vc);
        }

        let tail_start = chunks * NEON_F32_LANES;
        for i in 0..remainder {
            c[tail_start + i] = a[tail_start + i] * scalar;
        }
    }

    /// NEON dot product.
    pub unsafe fn vector_dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / NEON_F32_LANES;
        let remainder = n % NEON_F32_LANES;

        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * NEON_F32_LANES;
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            acc = vfmaq_f32(acc, va, vb);
        }

        let sum = vaddvq_f32(acc);

        let tail_start = chunks * NEON_F32_LANES;
        let mut tail_sum = 0.0f32;
        for i in 0..remainder {
            tail_sum += a[tail_start + i] * b[tail_start + i];
        }

        sum + tail_sum
    }

    /// NEON sum reduction.
    pub unsafe fn vector_reduce_sum_f32_neon(a: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / NEON_F32_LANES;
        let remainder = n % NEON_F32_LANES;

        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * NEON_F32_LANES;
            let va = vld1q_f32(a.as_ptr().add(offset));
            acc = vaddq_f32(acc, va);
        }

        let sum = vaddvq_f32(acc);

        let tail_start = chunks * NEON_F32_LANES;
        let mut tail_sum = 0.0f32;
        for i in 0..remainder {
            tail_sum += a[tail_start + i];
        }

        sum + tail_sum
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_vector_add_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0; 9];

        vector_add_f32(&a, &b, &mut c);

        for val in &c {
            assert!(approx_eq(*val, 10.0), "Expected 10.0, got {val}");
        }
    }

    #[test]
    fn test_vector_mul_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 4];

        vector_mul_f32(&a, &b, &mut c);

        assert!(approx_eq(c[0], 2.0));
        assert!(approx_eq(c[1], 6.0));
        assert!(approx_eq(c[2], 12.0));
        assert!(approx_eq(c[3], 20.0));
    }

    #[test]
    fn test_vector_scale_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut c = vec![0.0; 5];

        vector_scale_f32(&a, 3.0, &mut c);

        assert!(approx_eq(c[0], 3.0));
        assert!(approx_eq(c[1], 6.0));
        assert!(approx_eq(c[2], 9.0));
        assert!(approx_eq(c[3], 12.0));
        assert!(approx_eq(c[4], 15.0));
    }

    #[test]
    fn test_vector_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = vector_dot_f32(&a, &b);
        assert!(approx_eq(result, 10.0));
    }

    #[test]
    fn test_vector_reduce_sum_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = vector_reduce_sum_f32(&a);
        assert!(approx_eq(result, 55.0));
    }

    #[test]
    fn test_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];

        vector_add_f32(&a, &b, &mut c);
        vector_mul_f32(&a, &b, &mut c);
        vector_scale_f32(&a, 2.0, &mut c);
        assert!(approx_eq(vector_dot_f32(&a, &b), 0.0));
        assert!(approx_eq(vector_reduce_sum_f32(&a), 0.0));
    }

    #[test]
    fn test_large_vector() {
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut c = vec![0.0; n];

        vector_add_f32(&a, &b, &mut c);

        for val in &c {
            assert!(approx_eq(*val, n as f32));
        }
    }

    #[test]
    #[should_panic(expected = "a.len() != b.len()")]
    fn test_mismatched_lengths_add() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let mut c = vec![0.0; 2];
        vector_add_f32(&a, &b, &mut c);
    }
}
