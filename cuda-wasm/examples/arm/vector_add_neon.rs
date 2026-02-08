//! ARM NEON SIMD vector addition example
//!
//! Demonstrates using NEON SIMD intrinsics for accelerated vector addition
//! on ARM/AArch64 platforms. Falls back to scalar operations on non-ARM targets.
//!
//! # Supported Platforms
//! - Apple Silicon (M1/M2/M3/M4)
//! - ARM Linux (AWS Graviton, Ampere Altra)
//! - Any AArch64 system with NEON support
//!
//! # Running
//! ```bash
//! cargo run --example vector_add_neon
//! ```

use std::time::Instant;

/// Scalar vector addition (fallback for non-ARM platforms)
fn vector_add_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

/// NEON-accelerated vector addition for AArch64 targets
///
/// Processes 4 f32 elements per NEON instruction using 128-bit SIMD registers.
/// Falls back to scalar for any remainder elements.
#[cfg(target_arch = "aarch64")]
fn vector_add_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());

    let n = a.len();
    let simd_width = 4; // NEON processes 4 x f32 in a 128-bit register
    let simd_end = n - (n % simd_width);

    // NEON SIMD path: process 4 elements at a time
    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i < simd_end {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vaddq_f32(va, vb);
            vst1q_f32(c.as_mut_ptr().add(i), vc);
            i += simd_width;
        }
    }

    // Scalar remainder
    for i in simd_end..n {
        c[i] = a[i] + b[i];
    }
}

/// Run a timed benchmark of a vector add function
fn benchmark_add<F>(name: &str, a: &[f32], b: &[f32], c: &mut [f32], f: F, iterations: usize)
where
    F: Fn(&[f32], &[f32], &mut [f32]),
{
    // Warm up
    for _ in 0..10 {
        f(a, b, c);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f(a, b, c);
    }
    let elapsed = start.elapsed();

    let total_ops = a.len() as f64 * iterations as f64;
    let gflops = total_ops / elapsed.as_secs_f64() / 1e9;
    let throughput_gb = (a.len() * std::mem::size_of::<f32>() * 3) as f64
        * iterations as f64
        / elapsed.as_secs_f64()
        / 1e9;

    println!("  {:<20} {:>10.3} ms  ({:.2} GFLOP/s, {:.2} GB/s)",
        name,
        elapsed.as_secs_f64() * 1000.0,
        gflops,
        throughput_gb,
    );
}

fn main() {
    println!("=== cuda-wasm ARM NEON Vector Addition Example ===\n");

    // Detect platform
    let arch = std::env::consts::ARCH;
    println!("Platform: {} / {}", std::env::consts::OS, arch);

    let is_arm = cfg!(target_arch = "aarch64");
    if is_arm {
        println!("ARM NEON SIMD: AVAILABLE");
    } else {
        println!("ARM NEON SIMD: NOT AVAILABLE (running on {})", arch);
        println!("  -> Using scalar fallback. Run on an ARM platform for NEON acceleration.");
    }
    println!();

    // Test with various sizes
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let iterations = 1000;

    for &size in &sizes {
        println!("Vector size: {} elements ({} KB per vector)",
            size,
            size * std::mem::size_of::<f32>() / 1024
        );

        // Initialize test data
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();
        let mut c_scalar = vec![0.0f32; size];
        #[allow(unused_mut)]
        let mut c_neon = vec![0.0f32; size];

        // Benchmark scalar
        benchmark_add("Scalar", &a, &b, &mut c_scalar, vector_add_scalar, iterations);

        // Benchmark NEON (only on ARM)
        #[cfg(target_arch = "aarch64")]
        {
            benchmark_add("NEON SIMD", &a, &b, &mut c_neon, vector_add_neon, iterations);

            // Verify NEON results match scalar
            let max_diff: f32 = c_scalar
                .iter()
                .zip(c_neon.iter())
                .map(|(s, n)| (s - n).abs())
                .fold(0.0f32, f32::max);

            if max_diff < 1e-6 {
                println!("  Verification: PASSED (max diff: {:.2e})", max_diff);
            } else {
                println!("  Verification: FAILED (max diff: {:.2e})", max_diff);
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            println!("  NEON SIMD          (skipped - not ARM)");
        }

        println!();
    }

    // Show sample results
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let mut c = vec![0.0f32; 8];

    #[cfg(target_arch = "aarch64")]
    vector_add_neon(&a, &b, &mut c);

    #[cfg(not(target_arch = "aarch64"))]
    vector_add_scalar(&a, &b, &mut c);

    println!("Sample result:");
    println!("  a = {:?}", &a);
    println!("  b = {:?}", &b);
    println!("  c = {:?}", &c);

    println!("\nDone.");
}
