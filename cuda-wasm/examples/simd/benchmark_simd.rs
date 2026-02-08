//! SIMD benchmark example for cuda-wasm
//!
//! Detects available SIMD features at runtime and benchmarks vector addition
//! across multiple data sizes, comparing SIMD-accelerated vs. scalar performance.
//!
//! # Supported SIMD ISAs
//! - **x86_64**: SSE2, AVX2, AVX-512 (via `is_x86_feature_detected!`)
//! - **AArch64**: NEON (always available on AArch64)
//!
//! # Running
//! ```bash
//! cargo run --example benchmark_simd --release
//! ```
//!
//! Use `RUSTFLAGS="-C target-cpu=native"` for best results on your hardware.

use std::time::Instant;

// ---- SIMD Feature Detection ----

/// Detected SIMD capabilities on the current platform
struct SimdCapabilities {
    has_sse2: bool,
    has_avx2: bool,
    has_avx512f: bool,
    has_neon: bool,
    best_width: usize, // Number of f32 elements per SIMD register
    best_name: &'static str,
}

fn detect_simd() -> SimdCapabilities {
    let mut caps = SimdCapabilities {
        has_sse2: false,
        has_avx2: false,
        has_avx512f: false,
        has_neon: false,
        best_width: 1, // scalar fallback
        best_name: "Scalar",
    };

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            caps.has_sse2 = true;
            caps.best_width = 4; // 128-bit / 32-bit = 4
            caps.best_name = "SSE2";
        }
        if is_x86_feature_detected!("avx2") {
            caps.has_avx2 = true;
            caps.best_width = 8; // 256-bit / 32-bit = 8
            caps.best_name = "AVX2";
        }
        if is_x86_feature_detected!("avx512f") {
            caps.has_avx512f = true;
            caps.best_width = 16; // 512-bit / 32-bit = 16
            caps.best_name = "AVX-512";
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64
        caps.has_neon = true;
        caps.best_width = 4; // 128-bit / 32-bit = 4
        caps.best_name = "NEON";
    }

    caps
}

fn print_simd_capabilities(caps: &SimdCapabilities) {
    println!("SIMD Feature Detection:");
    println!("  Architecture: {}", std::env::consts::ARCH);

    #[cfg(target_arch = "x86_64")]
    {
        println!("  SSE2:     {}", if caps.has_sse2 { "YES" } else { "NO" });
        println!("  AVX2:     {}", if caps.has_avx2 { "YES" } else { "NO" });
        println!("  AVX-512:  {}", if caps.has_avx512f { "YES" } else { "NO" });
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  NEON:     {}", if caps.has_neon { "YES" } else { "NO" });
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("  (No SIMD detection for this architecture)");
    }

    println!("  Best ISA: {} ({} x f32 per register)", caps.best_name, caps.best_width);
    println!();
}

// ---- Vector Add Implementations ----

/// Scalar vector addition (baseline)
fn vector_add_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

/// SSE2-accelerated vector addition (x86_64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_add_sse2(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let simd_end = n - (n % 4);

    let mut i = 0;
    while i < simd_end {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
        i += 4;
    }
    for i in simd_end..n {
        c[i] = a[i] + b[i];
    }
}

/// AVX2-accelerated vector addition (x86_64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_add_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let simd_end = n - (n % 8);

    let mut i = 0;
    while i < simd_end {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
        i += 8;
    }
    for i in simd_end..n {
        c[i] = a[i] + b[i];
    }
}

/// AVX-512 accelerated vector addition (x86_64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_add_avx512(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = a.len();
    let simd_end = n - (n % 16);

    let mut i = 0;
    while i < simd_end {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let vc = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
        i += 16;
    }
    for i in simd_end..n {
        c[i] = a[i] + b[i];
    }
}

/// NEON-accelerated vector addition (AArch64)
#[cfg(target_arch = "aarch64")]
fn vector_add_neon(a: &[f32], b: &[f32], c: &mut [f32]) {
    let n = a.len();
    let simd_end = n - (n % 4);

    unsafe {
        use std::arch::aarch64::*;
        let mut i = 0;
        while i < simd_end {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vaddq_f32(va, vb);
            vst1q_f32(c.as_mut_ptr().add(i), vc);
            i += 4;
        }
    }
    for i in simd_end..n {
        c[i] = a[i] + b[i];
    }
}

// ---- Benchmarking ----

struct BenchResult {
    name: String,
    time_ms: f64,
    gflops: f64,
    bandwidth_gb: f64,
}

fn bench_vector_add<F>(name: &str, a: &[f32], b: &[f32], c: &mut [f32], f: F, iterations: usize) -> BenchResult
where
    F: Fn(&[f32], &[f32], &mut [f32]),
{
    // Warmup
    for _ in 0..20 {
        f(a, b, c);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f(a, b, c);
    }
    let elapsed = start.elapsed();

    let n = a.len() as f64;
    let total_ops = n * iterations as f64;
    let gflops = total_ops / elapsed.as_secs_f64() / 1e9;

    // 3 arrays * n elements * 4 bytes per element (2 reads + 1 write)
    let bytes_per_iter = n * 3.0 * 4.0;
    let bandwidth_gb = bytes_per_iter * iterations as f64 / elapsed.as_secs_f64() / 1e9;

    BenchResult {
        name: name.to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        gflops,
        bandwidth_gb,
    }
}

fn print_results_table(size: usize, results: &[BenchResult]) {
    let scalar_time = results.first().map(|r| r.time_ms).unwrap_or(1.0);

    println!("  {:<15} {:>12} {:>12} {:>12} {:>10}",
        "Method", "Time (ms)", "GFLOP/s", "BW (GB/s)", "Speedup");
    println!("  {}", "-".repeat(65));

    for result in results {
        let speedup = if result.time_ms > 0.0 {
            scalar_time / result.time_ms
        } else {
            0.0
        };

        println!("  {:<15} {:>12.3} {:>12.2} {:>12.2} {:>9.2}x",
            result.name, result.time_ms, result.gflops, result.bandwidth_gb, speedup);
    }
    println!();
}

fn main() {
    println!("=== cuda-wasm SIMD Benchmark ===\n");

    let caps = detect_simd();
    print_simd_capabilities(&caps);

    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let iterations = 5000;

    for &size in &sizes {
        println!("Vector size: {} elements ({} KB)",
            size,
            size * std::mem::size_of::<f32>() / 1024
        );

        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.001).collect();
        let mut c = vec![0.0f32; size];

        let adjusted_iters = if size >= 1_000_000 { iterations / 10 } else { iterations };

        let mut results = Vec::new();

        // Scalar baseline (always available)
        results.push(bench_vector_add("Scalar", &a, &b, &mut c, vector_add_scalar, adjusted_iters));

        // x86_64 SIMD variants
        #[cfg(target_arch = "x86_64")]
        {
            if caps.has_sse2 {
                results.push(bench_vector_add("SSE2", &a, &b, &mut c, |a, b, c| {
                    unsafe { vector_add_sse2(a, b, c) }
                }, adjusted_iters));
            }
            if caps.has_avx2 {
                results.push(bench_vector_add("AVX2", &a, &b, &mut c, |a, b, c| {
                    unsafe { vector_add_avx2(a, b, c) }
                }, adjusted_iters));
            }
            if caps.has_avx512f {
                results.push(bench_vector_add("AVX-512", &a, &b, &mut c, |a, b, c| {
                    unsafe { vector_add_avx512(a, b, c) }
                }, adjusted_iters));
            }
        }

        // AArch64 NEON
        #[cfg(target_arch = "aarch64")]
        {
            if caps.has_neon {
                results.push(bench_vector_add("NEON", &a, &b, &mut c, vector_add_neon, adjusted_iters));
            }
        }

        print_results_table(size, &results);

        // Verify correctness of the best SIMD implementation
        let mut c_scalar = vec![0.0f32; size];
        vector_add_scalar(&a, &b, &mut c_scalar);

        let max_diff: f32 = c_scalar
            .iter()
            .zip(c.iter())
            .map(|(s, n)| (s - n).abs())
            .fold(0.0f32, f32::max);

        if max_diff < 1e-6 {
            println!("  Correctness: PASSED (max diff: {:.2e})\n", max_diff);
        } else {
            println!("  Correctness: FAILED (max diff: {:.2e})\n", max_diff);
        }
    }

    // Summary
    println!("=== Summary ===");
    println!("  Best available SIMD: {} ({}-wide f32)", caps.best_name, caps.best_width);
    println!("  Theoretical speedup: {}x over scalar", caps.best_width);
    println!();
    println!("Tips:");
    println!("  - Build with RUSTFLAGS=\"-C target-cpu=native\" for auto-vectorization");
    println!("  - Use --release for meaningful benchmarks");
    println!("  - On x86_64, AVX-512 gives best throughput when available");
    println!("  - On ARM, NEON is always available on AArch64");
    println!("\nDone.");
}
