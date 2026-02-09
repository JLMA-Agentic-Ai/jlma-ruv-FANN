//! SIMD-accelerated matrix operations
//!
//! Provides optimized matrix multiplication using tiled algorithms with
//! SIMD inner loops. Matrices are stored in row-major order as flat slices.

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Matrix multiply: C = A * B
///
/// Dimensions: A is (m x k), B is (k x n), C is (m x n).
/// All matrices are stored in row-major order as flat `[f32]` slices.
///
/// # Panics
/// Panics if slice lengths do not match the declared dimensions.
pub fn matrix_multiply_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k, "matrix_multiply_f32: a.len() != m * k");
    assert_eq!(b.len(), k * n, "matrix_multiply_f32: b.len() != k * n");
    assert_eq!(c.len(), m * n, "matrix_multiply_f32: c.len() != m * n");

    // Zero out C
    for val in c.iter_mut() {
        *val = 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::matrix_multiply_f32_avx2(a, b, c, m, n, k) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::matrix_multiply_f32_neon(a, b, c, m, n, k) };
        return;
    }

    #[allow(unreachable_code)]
    scalar::matrix_multiply_f32_scalar(a, b, c, m, n, k);
}

// ---------------------------------------------------------------------------
// Tiling constants
// ---------------------------------------------------------------------------

/// Tile size for the blocked/tiled matrix multiply.
/// Chosen to fit well in L1 cache for typical desktop CPUs.
const TILE_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------
mod scalar {
    use super::TILE_SIZE;

    /// Tiled scalar matrix multiply for better cache behaviour.
    pub fn matrix_multiply_f32_scalar(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        // Tiled loop ordering: tiles over (i, j, p) with inner micro-kernel
        let ti_count = (m + TILE_SIZE - 1) / TILE_SIZE;
        let tj_count = (n + TILE_SIZE - 1) / TILE_SIZE;
        let tp_count = (k + TILE_SIZE - 1) / TILE_SIZE;

        for ti in 0..ti_count {
            let i_start = ti * TILE_SIZE;
            let i_end = (i_start + TILE_SIZE).min(m);

            for tj in 0..tj_count {
                let j_start = tj * TILE_SIZE;
                let j_end = (j_start + TILE_SIZE).min(n);

                for tp in 0..tp_count {
                    let p_start = tp * TILE_SIZE;
                    let p_end = (p_start + TILE_SIZE).min(k);

                    // Inner micro-kernel
                    for i in i_start..i_end {
                        for p in p_start..p_end {
                            let a_ip = a[i * k + p];
                            for j in j_start..j_end {
                                c[i * n + j] += a_ip * b[p * n + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 implementation (x86_64)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    use super::TILE_SIZE;

    const AVX2_F32_LANES: usize = 8;

    /// AVX2 tiled matrix multiply. Processes 8 f32s in the inner j-loop.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available and dimensions match slice lengths.
    #[target_feature(enable = "avx2")]
    pub unsafe fn matrix_multiply_f32_avx2(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let ti_count = (m + TILE_SIZE - 1) / TILE_SIZE;
        let tj_count = (n + TILE_SIZE - 1) / TILE_SIZE;
        let tp_count = (k + TILE_SIZE - 1) / TILE_SIZE;

        for ti in 0..ti_count {
            let i_start = ti * TILE_SIZE;
            let i_end = (i_start + TILE_SIZE).min(m);

            for tj in 0..tj_count {
                let j_start = tj * TILE_SIZE;
                let j_end = (j_start + TILE_SIZE).min(n);

                for tp in 0..tp_count {
                    let p_start = tp * TILE_SIZE;
                    let p_end = (p_start + TILE_SIZE).min(k);

                    for i in i_start..i_end {
                        for p in p_start..p_end {
                            let a_val = _mm256_set1_ps(a[i * k + p]);

                            // SIMD inner loop: process 8 columns at a time
                            let mut j = j_start;
                            while j + AVX2_F32_LANES <= j_end {
                                let b_vec = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
                                let c_vec = _mm256_loadu_ps(c.as_ptr().add(i * n + j));
                                let result = _mm256_add_ps(c_vec, _mm256_mul_ps(a_val, b_vec));
                                _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                                j += AVX2_F32_LANES;
                            }

                            // Scalar tail for remaining columns
                            let a_ip = a[i * k + p];
                            while j < j_end {
                                c[i * n + j] += a_ip * b[p * n + j];
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON implementation (aarch64)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    use super::TILE_SIZE;

    const NEON_F32_LANES: usize = 4;

    /// NEON tiled matrix multiply. Processes 4 f32s in the inner j-loop.
    ///
    /// # Safety
    /// Caller must ensure dimensions match slice lengths. NEON is mandatory on aarch64.
    pub unsafe fn matrix_multiply_f32_neon(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        let ti_count = (m + TILE_SIZE - 1) / TILE_SIZE;
        let tj_count = (n + TILE_SIZE - 1) / TILE_SIZE;
        let tp_count = (k + TILE_SIZE - 1) / TILE_SIZE;

        for ti in 0..ti_count {
            let i_start = ti * TILE_SIZE;
            let i_end = (i_start + TILE_SIZE).min(m);

            for tj in 0..tj_count {
                let j_start = tj * TILE_SIZE;
                let j_end = (j_start + TILE_SIZE).min(n);

                for tp in 0..tp_count {
                    let p_start = tp * TILE_SIZE;
                    let p_end = (p_start + TILE_SIZE).min(k);

                    for i in i_start..i_end {
                        for p in p_start..p_end {
                            let a_val = vdupq_n_f32(a[i * k + p]);

                            let mut j = j_start;
                            while j + NEON_F32_LANES <= j_end {
                                let b_vec = vld1q_f32(b.as_ptr().add(p * n + j));
                                let c_vec = vld1q_f32(c.as_ptr().add(i * n + j));
                                let result = vfmaq_f32(c_vec, a_val, b_vec);
                                vst1q_f32(c.as_mut_ptr().add(i * n + j), result);
                                j += NEON_F32_LANES;
                            }

                            let a_ip = a[i * k + p];
                            while j < j_end {
                                c[i * n + j] += a_ip * b[p * n + j];
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_identity_multiply() {
        // 3x3 identity * vector
        let a: Vec<f32> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let b: Vec<f32> = vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ];
        let mut c = vec![0.0; 6];

        matrix_multiply_f32(&a, &b, &mut c, 3, 2, 3);

        assert!(approx_eq(c[0], 1.0));
        assert!(approx_eq(c[1], 2.0));
        assert!(approx_eq(c[2], 3.0));
        assert!(approx_eq(c[3], 4.0));
        assert!(approx_eq(c[4], 5.0));
        assert!(approx_eq(c[5], 6.0));
    }

    #[test]
    fn test_2x2_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        matrix_multiply_f32(&a, &b, &mut c, 2, 2, 2);

        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert!(approx_eq(c[0], 19.0));
        assert!(approx_eq(c[1], 22.0));
        assert!(approx_eq(c[2], 43.0));
        assert!(approx_eq(c[3], 50.0));
    }

    #[test]
    fn test_large_matrix() {
        let m = 64;
        let n = 64;
        let k = 64;

        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();
        let mut c_simd = vec![0.0; m * n];
        let mut c_ref = vec![0.0; m * n];

        matrix_multiply_f32(&a, &b, &mut c_simd, m, n, k);

        // Reference naive multiply
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }

        for idx in 0..m * n {
            assert!(
                approx_eq(c_simd[idx], c_ref[idx]),
                "Mismatch at index {idx}: got {}, expected {}",
                c_simd[idx],
                c_ref[idx]
            );
        }
    }

    #[test]
    fn test_non_square_matrix() {
        // A: 2x3, B: 3x4, C: 2x4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = vec![0.0; 8];

        matrix_multiply_f32(&a, &b, &mut c, 2, 4, 3);

        // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
        //       = [38, 44, 50, 56]
        // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
        //       = [83, 98, 113, 128]
        assert!(approx_eq(c[0], 38.0));
        assert!(approx_eq(c[1], 44.0));
        assert!(approx_eq(c[2], 50.0));
        assert!(approx_eq(c[3], 56.0));
        assert!(approx_eq(c[4], 83.0));
        assert!(approx_eq(c[5], 98.0));
        assert!(approx_eq(c[6], 113.0));
        assert!(approx_eq(c[7], 128.0));
    }

    #[test]
    #[should_panic(expected = "a.len() != m * k")]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        matrix_multiply_f32(&a, &b, &mut c, 2, 2, 2);
    }
}
