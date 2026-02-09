# ARM Support for cuda-wasm

This directory contains examples demonstrating ARM platform support in cuda-wasm, including NEON SIMD acceleration, Apple Silicon optimizations, and ARM server deployment guidance.

## NEON SIMD Acceleration

ARM NEON is a 128-bit SIMD (Single Instruction, Multiple Data) extension available on all modern ARM processors. cuda-wasm leverages NEON to accelerate compute-intensive operations when running on ARM targets.

### Key Capabilities

- **128-bit registers**: Process 4x f32 or 2x f64 per instruction
- **Fused multiply-accumulate (FMA)**: Single-cycle multiply-add via `vfmaq_f32`
- **Automatic vectorization**: Rust compiler can auto-vectorize with `target-feature=+neon`
- **Runtime detection**: Detect NEON availability with `cfg!(target_arch = "aarch64")`

### Performance Characteristics

| Operation | Scalar | NEON | Speedup |
|-----------|--------|------|---------|
| Vector Add (1M f32) | ~0.8ms | ~0.2ms | ~4x |
| Matrix Multiply (512x512) | ~250ms | ~60ms | ~4x |
| Dot Product (1M f32) | ~1.2ms | ~0.3ms | ~4x |

*Measured on Apple M2 with `--release`. Actual results vary by hardware.*

## Examples

### vector_add_neon.rs

Demonstrates NEON-accelerated vector addition:

```bash
cargo run --example vector_add_neon --release
```

Features:
- NEON `vld1q_f32` / `vaddq_f32` / `vst1q_f32` for 4-wide f32 processing
- Automatic scalar fallback on non-ARM platforms
- Benchmarks at multiple vector sizes (1K to 1M elements)
- Throughput measurement in GFLOP/s and GB/s

### matrix_multiply_arm.rs

Demonstrates ARM-optimized tiled matrix multiplication:

```bash
cargo run --example matrix_multiply_arm --release
```

Features:
- Cache-blocked (tiled) algorithm with 64x64 tiles
- NEON `vfmaq_f32` for fused multiply-accumulate
- Comparison: Naive vs. Tiled vs. NEON Tiled
- Speedup metrics and correctness verification

## Supported Platforms

### Apple Silicon (M1/M2/M3/M4)

All Apple Silicon chips include NEON and support cuda-wasm natively:

```bash
# Build for native Apple Silicon
cargo build --release --target aarch64-apple-darwin

# Run examples
cargo run --example vector_add_neon --release
cargo run --example matrix_multiply_arm --release
```

Apple Silicon also provides:
- **AMX**: Apple Matrix Extension for large matrix ops (via Accelerate framework)
- **Apple GPU**: Metal/WebGPU backend for GPU compute
- **Unified memory**: Zero-copy CPU-GPU data sharing

### ARM Server Deployment

#### AWS Graviton (Graviton2, Graviton3, Graviton4)

```bash
# Cross-compile for Graviton
cargo build --release --target aarch64-unknown-linux-gnu

# Or build directly on a Graviton instance
cargo build --release
```

Graviton tips:
- Use `RUSTFLAGS="-C target-cpu=neoverse-n1"` for Graviton2
- Use `RUSTFLAGS="-C target-cpu=neoverse-v1"` for Graviton3
- Graviton3 adds SVE (Scalable Vector Extension) support

#### Ampere Altra / Altra Max

```bash
# Build with Neoverse N1 optimizations (Ampere Altra)
RUSTFLAGS="-C target-cpu=neoverse-n1" cargo build --release
```

Ampere Altra features:
- Up to 128 cores per socket
- Consistent single-threaded performance
- Available on Nutanix AHV and major cloud providers

#### NVIDIA Grace (ARM + GPU)

```bash
# Build for Grace Hopper Superchip
RUSTFLAGS="-C target-cpu=neoverse-v2" cargo build --release
```

Grace Hopper combines:
- ARM Neoverse V2 CPU cores
- H100 GPU with NVLink-C2C
- Coherent CPU-GPU memory via NVLink

## WebGPU on ARM GPUs

cuda-wasm supports WebGPU on ARM mobile and embedded GPUs:

### Mali GPUs (ARM)

- Mali-G710, G720 and later support Vulkan 1.1+
- WebGPU via Dawn/wgpu on Linux
- Common in MediaTek and Samsung Exynos SoCs

### Qualcomm Adreno GPUs

- Adreno 600/700 series support Vulkan 1.1+
- WebGPU on Android via Chrome
- Common in Snapdragon SoCs

### Apple GPU

- Full Metal and WebGPU support
- Best-in-class mobile GPU performance
- Unified memory architecture

## Cross-Compilation

### From x86_64 to AArch64

```bash
# Install the cross-compilation target
rustup target add aarch64-unknown-linux-gnu

# Install cross-compilation toolchain (Ubuntu/Debian)
sudo apt install gcc-aarch64-linux-gnu

# Build
cargo build --release --target aarch64-unknown-linux-gnu
```

### Using Docker for Cross-Compilation

```bash
# Build ARM container image
docker buildx build --platform linux/arm64 -t cuda-wasm:arm64 .

# Run on ARM (or with QEMU emulation)
docker run --platform linux/arm64 cuda-wasm:arm64
```

## Integration with cuda-wasm Runtime

The ARM examples use the same cuda-wasm API as x86 workloads. The runtime automatically detects ARM NEON and uses it for supported operations:

```rust
use cuda_rust_wasm::prelude::*;

fn main() -> Result<()> {
    let runtime = Runtime::new()?;

    // Runtime automatically detects ARM NEON
    println!("Backend: {:?}", runtime.device().backend());

    // Same API regardless of platform
    let device = runtime.device();
    let mut buffer = DeviceBuffer::new(1024, device.clone())?;
    // ...
    Ok(())
}
```
