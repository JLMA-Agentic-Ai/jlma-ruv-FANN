# ADR-002: Add SIMD Acceleration Layer for CPU-Side Operations

## Status

**Accepted**

Date: 2025-07-15

## Context

The cuda-rust-wasm project supports multiple execution backends:

1. **WebGPU** (`src/backend/webgpu.rs`, `src/backend/webgpu_optimized.rs`) -- GPU compute via WGSL shaders
2. **Native GPU** (`src/backend/native_gpu.rs`) -- Direct CUDA/OpenCL execution (feature-gated)
3. **WASM/CPU Runtime** (`src/backend/wasm_runtime.rs`) -- CPU fallback for environments without GPU access

The backend selection logic in `src/backend/mod.rs` shows the fallback chain:

```rust
pub fn get_backend() -> Box<dyn Backend> {
    #[cfg(target_arch = "wasm32")]
    { Box::new(webgpu::WebGPUBackend::new()) }

    #[cfg(not(target_arch = "wasm32"))]
    {
        #[cfg(feature = "cuda-backend")]
        { if native_gpu::is_cuda_available() { return Box::new(native_gpu::NativeGPUBackend::new()); } }
        Box::new(wasm_runtime::WasmRuntime::new()) // CPU fallback
    }
}
```

The CPU fallback path (`WasmRuntime`) performs kernel operations using scalar Rust code with no vectorization. This fallback is used in:

- Server-side environments without GPU drivers (CI/CD, containers, cloud functions)
- Development/testing workflows
- WASM environments where WebGPU is not available (Node.js, older browsers)
- ARM devices without GPU compute support (Raspberry Pi, embedded systems)

The project already declares a `wasm-simd` feature flag in `Cargo.toml` but it is not wired to any implementation:

```toml
[features]
wasm-simd = []
```

Key observations:

- **No SIMD intrinsics are used anywhere in the codebase.** All arithmetic operations in the CPU path are scalar.
- **The transpiled kernel code** (see `examples/transpiled/`) generates scalar Rust loops that could benefit from auto-vectorization hints or explicit SIMD.
- **Common GPU workloads** (vector addition, matrix multiplication, reductions, stencils) map directly to SIMD operations.
- **Rust stable supports `std::simd`** (portable SIMD) as of Rust 1.78, and `std::arch` provides access to platform intrinsics.
- **WASM SIMD** (128-bit, corresponding to `v128`) is supported in all major browsers and Node.js 16+.

## Decision

We will implement a SIMD acceleration layer that provides vectorized execution paths for CPU-side kernel operations. The layer will be structured as follows:

### Architecture

```
src/simd/
    mod.rs              -- Public API, runtime feature detection, dispatch
    portable.rs         -- Portable SIMD using std::simd (Rust nightly) or manual u32x4/f32x4
    x86.rs              -- AVX2 and AVX-512 intrinsics via std::arch::x86_64
    arm.rs              -- NEON intrinsics via std::arch::aarch64
    wasm.rs             -- WASM SIMD128 intrinsics via std::arch::wasm32
    operations.rs       -- SIMD-accelerated kernel primitives (reduce, scan, elementwise)
    tests.rs            -- Cross-platform SIMD correctness tests
```

### Runtime Feature Detection

The SIMD module will detect available hardware features at runtime using `std::is_x86_feature_detected!` (x86) and compile-time target features (ARM, WASM):

```rust
pub enum SimdCapability {
    Scalar,         // No SIMD, scalar fallback
    Neon,           // ARM NEON (128-bit, 4xf32)
    Sse42,          // x86 SSE4.2 (128-bit, 4xf32)
    Avx2,           // x86 AVX2 (256-bit, 8xf32)
    Avx512,         // x86 AVX-512 (512-bit, 16xf32)
    WasmSimd128,    // WASM SIMD (128-bit, 4xf32)
}

pub fn detect_simd() -> SimdCapability { ... }
```

### SIMD Operation Primitives

The layer will provide vectorized implementations for common kernel operations:

| Operation | Scalar | NEON (128-bit) | AVX2 (256-bit) | AVX-512 (512-bit) | WASM SIMD |
|-----------|--------|----------------|-----------------|---------------------|-----------|
| Vector add (f32) | 1 elem/cycle | 4 elem/cycle | 8 elem/cycle | 16 elem/cycle | 4 elem/cycle |
| Vector mul (f32) | 1 elem/cycle | 4 elem/cycle | 8 elem/cycle | 16 elem/cycle | 4 elem/cycle |
| Dot product (f32) | 1 elem/cycle | 4 elem/cycle | 8 elem/cycle | 16 elem/cycle | 4 elem/cycle |
| Reduction sum (f32) | serial | 4-wide + hadd | 8-wide + hadd | 16-wide + hadd | 4-wide + hadd |
| Prefix scan (f32) | serial | Blelloch 4-wide | Blelloch 8-wide | Blelloch 16-wide | Blelloch 4-wide |
| Matrix multiply (f32) | O(n^3) scalar | 4-wide tiled | 8-wide tiled | 16-wide tiled | 4-wide tiled |
| Min/Max reduction | serial | vminq/vmaxq | _mm256_min_ps | _mm512_min_ps | f32x4_min |

### Integration with Code Generator

The `CodeGenerator` (`src/transpiler/code_generator.rs`) will be extended to emit SIMD-aware Rust code when generating transpiled kernels for CPU targets. The generated code will call into the SIMD dispatch layer rather than performing scalar operations directly.

### Feature Gating

```toml
[features]
wasm-simd = []       # Existing, will now enable WASM SIMD128 path
simd = []            # Enable all native SIMD paths (AVX2, NEON)
simd-avx512 = []     # Opt-in AVX-512 (requires nightly or target-cpu)
```

### Unsafe Code Policy

SIMD intrinsics require `unsafe` blocks. All unsafe SIMD code will:

1. Be confined to the `src/simd/` module
2. Include `// SAFETY:` comments documenting invariants
3. Have scalar fallback equivalents for correctness validation
4. Be tested with `proptest` for equivalence between scalar and SIMD paths

## Consequences

### Positive

- **2-8x CPU fallback performance.** SIMD vectorization provides 4x (NEON/SSE/WASM), 8x (AVX2), or 16x (AVX-512) throughput improvements for embarrassingly parallel operations.
- **WASM performance improvement.** WASM SIMD128 is widely supported and provides a 4x improvement in browsers without WebGPU.
- **Automatic dispatch.** Runtime feature detection ensures the fastest available path is used without user configuration.
- **Testing parity.** Scalar fallback enables correctness testing of SIMD paths against known-good scalar results.
- **Wires up existing feature flag.** The `wasm-simd` feature flag in `Cargo.toml` will finally have an implementation.

### Negative

- **Platform-specific code paths.** Each SIMD ISA requires a separate implementation, increasing maintenance surface. At minimum: scalar + NEON + AVX2 + WASM SIMD = 4 implementations per operation.
- **Unsafe code.** SIMD intrinsics are inherently unsafe in Rust. This introduces `unsafe` blocks into the codebase, which must be carefully audited.
- **Alignment requirements.** SIMD operations require aligned memory. The memory pool (`src/memory/memory_pool.rs`) must be extended to support alignment guarantees (currently allocates `Vec<u8>` with default alignment).
- **Compiler flag dependencies.** AVX2 requires `-C target-feature=+avx2` or `target-cpu=haswell` at compile time. AVX-512 requires nightly or specific CPU targets. These must be documented clearly.
- **Binary size impact.** SIMD code paths increase binary size. For WASM targets where size matters (see `profile.wasm-size` in `Cargo.toml`), SIMD should be opt-in.

### Risks

- **Auto-vectorization interference.** The Rust compiler (LLVM) may auto-vectorize scalar code. Explicit SIMD may not always outperform well-written scalar code with auto-vectorization hints. Benchmarking is required.
- **WASM SIMD availability.** While broadly supported, some WASM runtimes (especially in embedded contexts) may not support SIMD128. Feature detection at the WASM level requires checking `WebAssembly.validate()` with SIMD opcodes.

## References

- `src/backend/mod.rs` -- Backend selection with CPU fallback
- `src/backend/wasm_runtime.rs` -- Current scalar CPU runtime
- `src/memory/memory_pool.rs` -- Memory pool (needs alignment support)
- `Cargo.toml` -- `wasm-simd` feature flag (currently unused)
- `examples/transpiled/` -- Transpiled kernel examples showing scalar code generation
- Rust `std::arch` documentation: https://doc.rust-lang.org/std/arch/index.html
- WASM SIMD proposal: https://github.com/WebAssembly/simd
