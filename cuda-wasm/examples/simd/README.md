# SIMD Support in cuda-wasm

This directory contains examples demonstrating SIMD (Single Instruction, Multiple Data) acceleration in cuda-wasm across x86_64 and ARM platforms.

## Overview

cuda-wasm uses platform-specific SIMD instructions to accelerate compute operations on CPUs. This provides significant speedups for workloads that cannot use GPU acceleration (e.g., CPU fallback paths, preprocessing, or environments without GPU access).

## Supported SIMD Instruction Sets

### x86_64

| ISA | Register Width | f32 per Register | Detection |
|-----|---------------|-------------------|-----------|
| **SSE2** | 128-bit | 4 | `is_x86_feature_detected!("sse2")` |
| **AVX2** | 256-bit | 8 | `is_x86_feature_detected!("avx2")` |
| **AVX-512** | 512-bit | 16 | `is_x86_feature_detected!("avx512f")` |

SSE2 is available on virtually all x86_64 processors. AVX2 is available on Intel Haswell (2013) and AMD Excavator (2015) and later. AVX-512 is available on select Intel Xeon, Ice Lake, and AMD Zen 4+ processors.

### ARM / AArch64

| ISA | Register Width | f32 per Register | Detection |
|-----|---------------|-------------------|-----------|
| **NEON** | 128-bit | 4 | Always available on AArch64 |
| **SVE** | 128-2048 bit | Variable | `is_aarch64_feature_detected!("sve")` |
| **SVE2** | 128-2048 bit | Variable | `is_aarch64_feature_detected!("sve2")` |

NEON is mandatory on all AArch64 (ARM 64-bit) processors. SVE (Scalable Vector Extension) is available on ARM Neoverse V1+ (AWS Graviton3, Fujitsu A64FX).

## Runtime Detection

cuda-wasm detects SIMD capabilities at runtime and selects the best available implementation:

```rust
#[cfg(target_arch = "x86_64")]
{
    if is_x86_feature_detected!("avx512f") {
        // Use AVX-512 path (16 x f32)
    } else if is_x86_feature_detected!("avx2") {
        // Use AVX2 path (8 x f32)
    } else if is_x86_feature_detected!("sse2") {
        // Use SSE2 path (4 x f32)
    }
}

#[cfg(target_arch = "aarch64")]
{
    // NEON is always available (4 x f32)
}
```

This ensures compiled binaries run correctly on any hardware while exploiting the best available SIMD capabilities.

## Performance Characteristics

### Vector Addition (1M f32 elements, --release)

#### x86_64 (Intel Xeon, typical results)

| Method | Time | Speedup | Bandwidth |
|--------|------|---------|-----------|
| Scalar | 1.2 ms | 1.0x | 10 GB/s |
| SSE2 | 0.35 ms | 3.4x | 34 GB/s |
| AVX2 | 0.18 ms | 6.7x | 67 GB/s |
| AVX-512 | 0.10 ms | 12x | 120 GB/s |

#### AArch64 (Apple M2, typical results)

| Method | Time | Speedup | Bandwidth |
|--------|------|---------|-----------|
| Scalar | 0.8 ms | 1.0x | 15 GB/s |
| NEON | 0.22 ms | 3.6x | 55 GB/s |

*Note: At larger data sizes, performance becomes memory-bandwidth limited rather than compute-limited.*

### When SIMD Helps Most

- **Compute-bound operations**: Math-heavy kernels with high arithmetic intensity
- **Small to medium data**: Data fits in L1/L2 cache, avoiding memory bottlenecks
- **Embarrassingly parallel**: Independent element-wise operations (add, multiply, etc.)

### When SIMD Helps Less

- **Memory-bound operations**: Large data exceeding cache, limited by DRAM bandwidth
- **Branch-heavy code**: Conditional logic that prevents vectorization
- **Sequential dependencies**: Loop-carried dependencies that cannot be parallelized

## Examples

### benchmark_simd.rs

Run the SIMD benchmark:

```bash
# Basic run
cargo run --example benchmark_simd --release

# With native CPU optimizations (recommended)
RUSTFLAGS="-C target-cpu=native" cargo run --example benchmark_simd --release
```

The benchmark:
1. Detects available SIMD features
2. Runs vector_add at 4 sizes: 1K, 10K, 100K, 1M elements
3. Tests all available SIMD implementations
4. Prints a comparison table with speedup vs. scalar
5. Verifies correctness of SIMD results

## Building with SIMD

### Compile-Time Feature Selection

```bash
# Let the compiler auto-vectorize for the build machine
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Target specific x86_64 features
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release

# Target specific ARM features
RUSTFLAGS="-C target-feature=+neon,+fp-armv8" cargo build --release
```

### Runtime Feature Detection

For distributed binaries that need to run on various hardware:

```rust
// Use #[target_feature] attribute for unsafe SIMD functions
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_avx2(data: &mut [f32]) {
    // AVX2 intrinsics here
}

// Runtime dispatch
fn compute(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_avx2(data) };
        }
    }
    compute_scalar(data);
}
```

## Integration with cuda-wasm

SIMD acceleration is used as a CPU fallback in cuda-wasm when:

1. **No GPU available**: Server or container without GPU access
2. **Small workloads**: Data too small to justify GPU transfer overhead
3. **Preprocessing**: Data preparation before GPU kernel launch
4. **WebAssembly**: WASM SIMD (128-bit) for browser-based compute
