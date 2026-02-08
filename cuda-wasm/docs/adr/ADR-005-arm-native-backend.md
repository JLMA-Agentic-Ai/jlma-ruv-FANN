# ADR-005: Implement ARM-Native GPU Compute via Vulkan/Metal Using wgpu Abstraction

## Status

**Accepted**

Date: 2025-07-15

## Context

The cuda-rust-wasm project's ARM support is currently limited to:

1. **Compilation target**: The project compiles for `aarch64` via Rust's cross-compilation support.
2. **CPU fallback**: On ARM devices without CUDA, the `WasmRuntime` (scalar CPU backend) is used.
3. **No ARM GPU compute**: There is no backend that utilizes ARM-based GPU hardware (Mali, Adreno, Apple GPU) for compute workloads.

The backend selection logic (`src/backend/mod.rs`) shows the gap:

```rust
#[cfg(not(target_arch = "wasm32"))]
{
    #[cfg(feature = "cuda-backend")]
    {
        if native_gpu::is_cuda_available() {
            return Box::new(native_gpu::NativeGPUBackend::new());
        }
    }
    // Falls through to WasmRuntime (CPU) on ARM devices with GPUs
    Box::new(wasm_runtime::WasmRuntime::new())
}
```

This means ARM devices with capable GPU hardware -- including mobile phones, tablets, Apple Silicon Macs, NVIDIA Jetson platforms, and Raspberry Pi 5 (VideoCore VII) -- fall back to scalar CPU execution, wasting available GPU compute resources.

The current GPU-related dependencies in `Cargo.toml` include:

```toml
wgpu = { version = "0.19", features = ["webgl", "webgpu"] }
vulkano = { version = "0.34", optional = true }
```

The `wgpu` crate is already a dependency and provides a cross-platform GPU abstraction that supports:

| Backend  | Platforms                            | GPU Hardware          |
|----------|--------------------------------------|-----------------------|
| Vulkan   | Linux, Android, Windows              | NVIDIA, AMD, Intel, Mali, Adreno |
| Metal    | macOS, iOS                           | Apple GPU              |
| DX12     | Windows                              | NVIDIA, AMD, Intel     |
| OpenGL ES| Android (fallback)                   | Mali, Adreno, PowerVR |

Since `wgpu` abstracts over all these backends, it can provide GPU compute on ARM devices through Vulkan (Linux/Android) or Metal (Apple), without requiring device-specific code.

The existing WebGPU backend (`src/backend/webgpu.rs`) targets the browser WebGPU API through `wasm-bindgen`. It does **not** use `wgpu` for native GPU compute. The `webgpu_optimized.rs` variant also targets browser environments.

## Decision

We will implement an ARM-native GPU compute backend using `wgpu` as the hardware abstraction layer. This backend will work on all platforms supported by `wgpu`, not just ARM, providing a unified native GPU compute path.

### Architecture

```
src/backend/
    mod.rs                  -- Updated backend selection with wgpu-native priority
    backend_trait.rs        -- Existing BackendTrait (unchanged)
    wgpu_native.rs          -- NEW: Native GPU compute via wgpu (Vulkan/Metal/DX12)
    webgpu.rs               -- Existing: Browser WebGPU (unchanged)
    webgpu_optimized.rs     -- Existing: Optimized browser WebGPU (unchanged)
    native_gpu.rs           -- Existing: Direct CUDA/OpenCL (unchanged)
    wasm_runtime.rs         -- Existing: CPU fallback (unchanged)
```

### WgpuNativeBackend Implementation

```rust
pub struct WgpuNativeBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    capabilities: BackendCapabilities,
}

#[async_trait]
impl BackendTrait for WgpuNativeBackend {
    fn name(&self) -> &str { "wgpu-native" }

    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    async fn initialize(&mut self) -> Result<()> {
        // Request adapter with compute-capable features
        // Prefer Vulkan on Linux/Android, Metal on macOS/iOS
        // Fall back to OpenGL ES if needed
    }

    async fn compile_kernel(&self, wgsl_source: &str) -> Result<Vec<u8>> {
        // Compile WGSL to a wgpu::ShaderModule
        // Serialize pipeline state for caching
    }

    async fn launch_kernel(
        &self, kernel: &[u8], grid: (u32, u32, u32),
        block: (u32, u32, u32), args: &[*const u8],
    ) -> Result<()> {
        // Create compute pipeline from cached shader module
        // Create bind groups for kernel arguments
        // Create command encoder, dispatch compute, submit
        // Read back results from GPU buffers
    }

    fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
        // Create wgpu::Buffer with STORAGE | COPY_SRC | COPY_DST usage
    }

    fn free_memory(&self, ptr: *mut u8) -> Result<()> {
        // Destroy associated wgpu::Buffer
    }

    fn copy_memory(&self, dst: *mut u8, src: *const u8, size: usize, kind: MemcpyKind) -> Result<()> {
        // Use wgpu::Queue::write_buffer for HostToDevice
        // Use buffer mapping for DeviceToHost
        // Use CommandEncoder::copy_buffer_to_buffer for DeviceToDevice
    }

    fn synchronize(&self) -> Result<()> {
        // Submit empty command buffer and wait for completion
        // Use device.poll(wgpu::Maintain::Wait)
    }
}
```

### Updated Backend Selection

The backend selection order will be updated to prioritize the wgpu-native backend on non-WASM platforms:

```rust
pub fn get_backend() -> Box<dyn Backend> {
    #[cfg(target_arch = "wasm32")]
    { Box::new(webgpu::WebGPUBackend::new()) }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // 1. Try direct CUDA if available and requested
        #[cfg(feature = "cuda-backend")]
        {
            if native_gpu::is_cuda_available() {
                return Box::new(native_gpu::NativeGPUBackend::new());
            }
        }

        // 2. Try wgpu-native (Vulkan/Metal/DX12)
        if let Ok(backend) = wgpu_native::WgpuNativeBackend::try_new() {
            return Box::new(backend);
        }

        // 3. CPU fallback
        Box::new(wasm_runtime::WasmRuntime::new())
    }
}
```

### Adapter Selection Strategy

The wgpu adapter selection will use a priority-based strategy:

1. **Discrete GPU** (high performance): Prefer discrete NVIDIA/AMD GPUs if available
2. **Integrated GPU** (power efficient): Fall back to integrated Intel/AMD/Apple GPUs
3. **Software renderer** (compatibility): Use software Vulkan (SwiftShader/lavapipe) as last resort

```rust
async fn select_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    // Try high-performance first
    if let Some(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await {
        return Some(adapter);
    }

    // Fall back to low-power
    instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await
}
```

### ARM-Specific Considerations

| Platform | GPU | wgpu Backend | Workgroup Size Limit | Notes |
|----------|-----|-------------|---------------------|-------|
| Android (Qualcomm) | Adreno | Vulkan | 128-1024 | Widely deployed, good compute support |
| Android (ARM) | Mali | Vulkan | 64-256 | Older Mali GPUs have limited compute |
| Android (Samsung) | Xclipse (AMD RDNA2) | Vulkan | 1024 | Full compute support |
| Apple Silicon Mac | Apple GPU | Metal | 1024 | Excellent compute, unified memory |
| Apple iOS | Apple GPU | Metal | 512-1024 | Power-constrained |
| NVIDIA Jetson | NVIDIA (Maxwell-Ampere) | Vulkan | 1024 | Full CUDA-class compute |
| Raspberry Pi 5 | VideoCore VII | Vulkan | 16-64 | Very limited compute |
| Linux (AMD) | RDNA/CDNA | Vulkan | 1024 | Full compute support |

The backend will query `wgpu::Adapter::limits()` to determine actual hardware limits and adjust workgroup sizes accordingly, rather than assuming NVIDIA-class capabilities (e.g., the current hardcoded workgroup size of 64 in the WGSL generator).

### WGSL Compatibility

Since the wgpu-native backend consumes WGSL shaders (the same output as the existing WGSL generator), the transpilation pipeline requires no changes:

```
CUDA Source --> Parser --> AST --> WGSL Generator --> WGSL Shader
                                                          |
                                          +---------------+
                                          |               |
                                   Browser WebGPU    wgpu-native
                                   (webgpu.rs)       (wgpu_native.rs)
```

This reuses the `WgslGenerator` (`src/transpiler/wgsl.rs`) without modification.

### Feature Configuration

```toml
[features]
wgpu-native = []    # Enable native GPU compute via wgpu
                    # No additional dependencies needed; wgpu is already required

[dependencies]
# wgpu is already a dependency, just ensure native features are enabled
wgpu = { version = "0.19", features = ["vulkan", "metal", "dx12"] }
```

## Consequences

### Positive

- **True GPU compute on ARM devices.** Mobile phones, tablets, Apple Silicon Macs, Jetson boards, and other ARM devices can execute transpiled kernels on their GPUs.
- **Unified backend for all GPU vendors.** A single `wgpu_native` backend works with NVIDIA (Vulkan), AMD (Vulkan), Intel (Vulkan), Apple (Metal), ARM Mali (Vulkan), and Qualcomm Adreno (Vulkan).
- **No new dependencies.** `wgpu` is already a dependency. Enabling native backends requires only feature flags.
- **WGSL pipeline reuse.** The existing WGSL generator provides the shader source. No transpiler changes are needed.
- **Automatic fallback.** The backend selection chain tries wgpu-native before CPU, ensuring GPU compute is used when available.
- **Desktop GPU support.** The wgpu-native backend is not ARM-specific; it works on x86 desktops with Vulkan/DX12, providing GPU compute without requiring CUDA drivers.

### Negative

- **wgpu abstraction overhead.** The wgpu abstraction adds a layer between the application and the GPU driver. For CUDA-capable NVIDIA GPUs, direct CUDA access (via `native_gpu.rs`) will always be more performant.
- **WGSL limitations.** WGSL does not support all CUDA features (64-bit atomics, warp primitives, dynamic parallelism, texture operations). The transpiled kernels are limited to what WGSL can express.
- **Workgroup size variation.** ARM GPUs have significantly different optimal workgroup sizes (Mali: 64, Apple: 256-1024) compared to NVIDIA (256-1024). The WGSL generator currently hardcodes `workgroup_size(64, 1, 1)`, which may be suboptimal.
- **Memory model differences.** ARM GPUs use different memory architectures (unified memory on Apple, tiled rendering on Mali/Adreno). Buffer allocation strategies may need per-vendor tuning.
- **Driver quality variation.** Vulkan driver quality on mobile ARM GPUs varies significantly. Mali and Adreno drivers have known bugs with compute shaders.

### Risks

- **wgpu version coupling.** The `wgpu` crate evolves rapidly (currently at 0.19). API changes in future versions may require backend updates.
- **Power consumption on mobile.** GPU compute on mobile devices can drain batteries quickly. The backend should support power-aware scheduling or user-configurable power preferences.
- **Memory pressure on mobile.** Mobile GPUs share memory with the CPU. Large kernel allocations may cause out-of-memory conditions.

## References

- `src/backend/mod.rs` -- Current backend selection logic
- `src/backend/backend_trait.rs` -- `BackendTrait` and `BackendCapabilities`
- `src/backend/webgpu.rs` -- Browser WebGPU backend (not usable natively)
- `src/transpiler/wgsl.rs` -- WGSL code generator (reusable)
- `Cargo.toml` -- wgpu dependency, vulkano optional dependency
- wgpu documentation: https://docs.rs/wgpu/
- wgpu supported backends: https://github.com/gfx-rs/wgpu#supported-platforms
- Vulkan on ARM: https://developer.arm.com/documentation/102249/latest
- Metal Compute: https://developer.apple.com/metal/
