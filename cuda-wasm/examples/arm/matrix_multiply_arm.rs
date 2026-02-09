//! ARM-optimized matrix multiplication example
//!
//! Demonstrates NEON-accelerated tiled matrix multiplication on ARM/AArch64
//! platforms and compares it with a naive implementation.
//!
//! # Optimization Techniques
//! - Tiled (blocked) access pattern for cache efficiency
//! - NEON SIMD intrinsics for 4-wide f32 multiply-accumulate
//! - Loop unrolling for reduced branch overhead
//!
//! # Supported Platforms
//! - Apple Silicon (M1/M2/M3/M4)
//! - ARM Linux (AWS Graviton, Ampere Altra, Raspberry Pi 4+)
//!
//! # Running
//! ```bash
//! cargo run --example matrix_multiply_arm --release
//! ```

use std::time::Instant;

/// Tile size for cache-blocked matrix multiplication.
/// 64 is chosen to fit within L1 cache on most ARM processors.
const TILE_SIZE: usize = 64;

/// Naive matrix multiplication: C = A * B
///
/// O(n^3) with poor cache behavior for large matrices.
fn matmul_naive(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Tiled (cache-blocked) matrix multiplication: C = A * B
///
/// Improves cache utilization by processing TILE_SIZE x TILE_SIZE sub-blocks.
fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // Zero the output
    for val in c.iter_mut() {
        *val = 0.0;
    }

    let tile = TILE_SIZE.min(n);

    for i0 in (0..n).step_by(tile) {
        for j0 in (0..n).step_by(tile) {
            for k0 in (0..n).step_by(tile) {
                let i_end = (i0 + tile).min(n);
                let j_end = (j0 + tile).min(n);
                let k_end = (k0 + tile).min(n);

                for i in i0..i_end {
                    for k in k0..k_end {
                        let a_ik = a[i * n + k];
                        for j in j0..j_end {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// NEON-accelerated tiled matrix multiplication for AArch64
///
/// Combines cache-blocking with NEON SIMD to process 4 f32 columns
/// simultaneously using `vfmaq_f32` (fused multiply-accumulate).
#[cfg(target_arch = "aarch64")]
fn matmul_neon_tiled(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    // Zero the output
    for val in c.iter_mut() {
        *val = 0.0;
    }

    let tile = TILE_SIZE.min(n);
    let simd_width = 4; // NEON 128-bit = 4 x f32

    for i0 in (0..n).step_by(tile) {
        for j0 in (0..n).step_by(tile) {
            for k0 in (0..n).step_by(tile) {
                let i_end = (i0 + tile).min(n);
                let j_end = (j0 + tile).min(n);
                let k_end = (k0 + tile).min(n);

                for i in i0..i_end {
                    for k in k0..k_end {
                        let a_ik = a[i * n + k];

                        // NEON SIMD path for aligned chunks of 4
                        let j_simd_end = j0 + ((j_end - j0) / simd_width) * simd_width;

                        unsafe {
                            use std::arch::aarch64::*;
                            let va = vdupq_n_f32(a_ik);

                            let mut j = j0;
                            while j < j_simd_end {
                                let vb = vld1q_f32(b.as_ptr().add(k * n + j));
                                let vc = vld1q_f32(c.as_ptr().add(i * n + j));
                                let vr = vfmaq_f32(vc, va, vb);
                                vst1q_f32(c.as_mut_ptr().add(i * n + j), vr);
                                j += simd_width;
                            }
                        }

                        // Scalar remainder
                        for j in j_simd_end..j_end {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Verify that two matrices are approximately equal
fn verify_results(expected: &[f32], actual: &[f32], tolerance: f32) -> (bool, f32) {
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(e, a)| (e - a).abs())
        .fold(0.0f32, f32::max);

    (max_diff <= tolerance, max_diff)
}

/// Run a timed benchmark of a matmul function
fn benchmark_matmul<F>(name: &str, a: &[f32], b: &[f32], c: &mut [f32], n: usize, f: F, iterations: usize)
where
    F: Fn(&[f32], &[f32], &mut [f32], usize),
{
    // Warm up
    f(a, b, c, n);

    let start = Instant::now();
    for _ in 0..iterations {
        f(a, b, c, n);
    }
    let elapsed = start.elapsed();

    // 2 * n^3 FLOPs for matrix multiply (n^3 multiplies + n^3 additions)
    let flops_per_iter = 2.0 * (n as f64).powi(3);
    let total_flops = flops_per_iter * iterations as f64;
    let gflops = total_flops / elapsed.as_secs_f64() / 1e9;
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!("  {:<25} {:>8.3} ms/iter  ({:.3} GFLOP/s)", name, ms_per_iter, gflops);
}

fn main() {
    println!("=== cuda-wasm ARM Matrix Multiply Example ===\n");

    // Detect platform
    let arch = std::env::consts::ARCH;
    println!("Platform: {} / {}", std::env::consts::OS, arch);

    let is_arm = cfg!(target_arch = "aarch64");
    if is_arm {
        println!("ARM NEON SIMD: AVAILABLE");
        println!("Tile size: {}x{}", TILE_SIZE, TILE_SIZE);
    } else {
        println!("ARM NEON SIMD: NOT AVAILABLE (running on {})", arch);
        println!("  -> Tiled scalar will be used. Run on ARM for NEON acceleration.");
    }
    println!();

    // Test with multiple matrix sizes
    let sizes: Vec<usize> = vec![64, 128, 256, 512];

    for &n in &sizes {
        let total_elements = n * n;
        let mem_per_matrix = total_elements * std::mem::size_of::<f32>();

        println!(
            "Matrix size: {}x{} ({} elements, {:.1} KB per matrix)",
            n, n, total_elements,
            mem_per_matrix as f64 / 1024.0
        );

        // Initialize matrices: A with small values, B as identity-ish for easy verification
        let a: Vec<f32> = (0..total_elements)
            .map(|idx| ((idx % n) as f32 + 1.0) * 0.01)
            .collect();
        let b: Vec<f32> = (0..total_elements)
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                if row == col { 1.0 } else { 0.001 }
            })
            .collect();

        let mut c_naive = vec![0.0f32; total_elements];
        let mut c_tiled = vec![0.0f32; total_elements];
        #[allow(unused_mut)]
        let mut c_neon = vec![0.0f32; total_elements];

        // Choose iteration count based on matrix size
        let iterations = if n <= 128 { 50 } else if n <= 256 { 10 } else { 3 };

        // Naive
        benchmark_matmul("Naive", &a, &b, &mut c_naive, n, matmul_naive, iterations);

        // Tiled scalar
        benchmark_matmul("Tiled (scalar)", &a, &b, &mut c_tiled, n, matmul_tiled, iterations);

        // Verify tiled matches naive
        let (ok, max_diff) = verify_results(&c_naive, &c_tiled, 1e-3);
        if ok {
            println!("  Tiled vs Naive:           MATCH (max diff: {:.2e})", max_diff);
        } else {
            println!("  Tiled vs Naive:           MISMATCH (max diff: {:.2e})", max_diff);
        }

        // NEON tiled (only on ARM)
        #[cfg(target_arch = "aarch64")]
        {
            benchmark_matmul("NEON Tiled", &a, &b, &mut c_neon, n, matmul_neon_tiled, iterations);

            let (ok, max_diff) = verify_results(&c_naive, &c_neon, 1e-3);
            if ok {
                println!("  NEON vs Naive:            MATCH (max diff: {:.2e})", max_diff);
            } else {
                println!("  NEON vs Naive:            MISMATCH (max diff: {:.2e})", max_diff);
            }

            // Calculate speedup
            let naive_time = {
                let start = Instant::now();
                for _ in 0..iterations {
                    matmul_naive(&a, &b, &mut c_naive, n);
                }
                start.elapsed().as_secs_f64()
            };
            let neon_time = {
                let start = Instant::now();
                for _ in 0..iterations {
                    matmul_neon_tiled(&a, &b, &mut c_neon, n);
                }
                start.elapsed().as_secs_f64()
            };

            if neon_time > 0.0 {
                println!("  Speedup (NEON vs Naive):  {:.2}x", naive_time / neon_time);
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            println!("  NEON Tiled                (skipped - not ARM)");

            // Show tiled vs naive speedup on this platform
            let naive_time = {
                let start = Instant::now();
                for _ in 0..iterations {
                    matmul_naive(&a, &b, &mut c_naive, n);
                }
                start.elapsed().as_secs_f64()
            };
            let tiled_time = {
                let start = Instant::now();
                for _ in 0..iterations {
                    matmul_tiled(&a, &b, &mut c_tiled, n);
                }
                start.elapsed().as_secs_f64()
            };

            if tiled_time > 0.0 {
                println!("  Speedup (Tiled vs Naive): {:.2}x", naive_time / tiled_time);
            }
        }

        println!();
    }

    println!("Notes:");
    println!("  - Run with --release for meaningful performance numbers");
    println!("  - NEON SIMD requires AArch64 (ARM 64-bit) targets");
    println!("  - Tiled algorithm improves cache utilization on all platforms");
    println!("  - For production, consider using BLAS libraries (OpenBLAS, Apple Accelerate)");
    println!("\nDone.");
}
